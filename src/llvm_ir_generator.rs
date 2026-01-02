use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::ValueKind;
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

use crate::parser::{
    BinOp, Block, Declaration, Expr, Function, FunctionHeader, Import, Literal, Program, Stmt,
    StructDef, Type, UnOp, VarDecl,
};

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,

    // Symbol tables
    variables: HashMap<String, PointerValue<'ctx>>,
    variable_types: HashMap<String, BasicTypeEnum<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    structs: HashMap<String, BasicTypeEnum<'ctx>>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();

        CodeGen {
            context,
            builder,
            module,
            variables: HashMap::new(),
            variable_types: HashMap::new(),
            functions: HashMap::new(),
            structs: HashMap::new(),
        }
    }

    // ================= Main Entry Point =================

    pub fn compile_program(&mut self, program: &Program) -> Result<(), String> {
        // First pass: declare all structs
        for decl in &program.declarations {
            if let Declaration::Struct(s) = decl {
                self.declare_struct(s)?;
            }
        }

        // Second pass: declare all functions
        for decl in &program.declarations {
            if let Declaration::Function(f) = decl {
                self.declare_function(f)?;
            }
        }

        // Third pass: compile function bodies and global variables
        for decl in &program.declarations {
            match decl {
                Declaration::Function(f) => self.compile_function(f)?,
                Declaration::Var(v) => self.compile_global_var(v)?,
                Declaration::Struct(_) => {} // Already handled
            }
        }

        Ok(())
    }

    // ================= Type Conversion =================

    fn convert_type(&self, ty: &Type) -> Result<BasicTypeEnum<'ctx>, String> {
        match ty {
            Type::Primitive(name) => match name.as_str() {
                "int" => Ok(self.context.i64_type().into()),
                "float" => Ok(self.context.f64_type().into()),
                "bool" => Ok(self.context.bool_type().into()),
                "char" => Ok(self.context.i8_type().into()),
                "string" => Ok(self
                    .context
                    .i8_type()
                    .ptr_type(AddressSpace::default())
                    .into()),
                "void" => Err("void is not a basic type".to_string()),
                _ => Err(format!("Unknown primitive type: {}", name)),
            },
            Type::Array(inner) => {
                let inner_ty = self.convert_type(inner)?;
                Ok(inner_ty.ptr_type(AddressSpace::default()).into())
            }
            Type::Custom(name) => self
                .structs
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Unknown custom type: {}", name)),
            Type::Function(_) => {
                // Function pointers
                Ok(self
                    .context
                    .i8_type()
                    .ptr_type(AddressSpace::default())
                    .into())
            }
        }
    }

    fn is_void_type(&self, ty: &Type) -> bool {
        matches!(ty, Type::Primitive(name) if name == "void")
    }

    // ================= Struct Handling =================

    fn declare_struct(&mut self, struct_def: &StructDef) -> Result<(), String> {
        let mut field_types = Vec::new();

        for (_, field_ty) in &struct_def.fields {
            let ty = self.convert_type(field_ty)?;
            field_types.push(ty);
        }

        let struct_type = self.context.struct_type(&field_types, false);
        self.structs
            .insert(struct_def.name.clone(), struct_type.into());

        Ok(())
    }

    // ================= Function Handling =================

    fn declare_function(&mut self, function: &Function) -> Result<FunctionValue<'ctx>, String> {
        let name = function
            .header
            .name
            .as_ref()
            .ok_or("Function must have a name at top level")?;

        if self.functions.contains_key(name) {
            return Ok(self.functions[name]);
        }

        let fn_type = self.get_function_type(&function.header)?;
        let fn_val = self.module.add_function(name, fn_type, None);

        self.functions.insert(name.clone(), fn_val);
        Ok(fn_val)
    }

    fn get_function_type(
        &self,
        header: &FunctionHeader,
    ) -> Result<inkwell::types::FunctionType<'ctx>, String> {
        let mut param_types = Vec::new();

        for (_, param_ty) in &header.params {
            let ty = self.convert_type(param_ty)?;
            param_types.push(ty.into());
        }

        if self.is_void_type(&header.return_type) {
            Ok(self.context.void_type().fn_type(&param_types, false))
        } else {
            let ret_ty = self.convert_type(&header.return_type)?;
            Ok(ret_ty.fn_type(&param_types, false))
        }
    }

    fn compile_function(&mut self, function: &Function) -> Result<(), String> {
        let name = function
            .header
            .name
            .as_ref()
            .ok_or("Function must have a name")?;

        let fn_val = self.functions[name];
        let entry_bb = self.context.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(entry_bb);

        // Create a new scope for function variables
        let prev_vars = self.variables.clone();
        let prev_types = self.variable_types.clone();
        self.variables.clear();
        self.variable_types.clear();

        // Allocate parameters
        for (i, (param_name, param_ty)) in function.header.params.iter().enumerate() {
            if let Some(name) = param_name {
                let param_val = fn_val
                    .get_nth_param(i as u32)
                    .ok_or_else(|| format!("Missing parameter {}", i))?;

                let param_type = self.convert_type(param_ty)?;
                let alloca = self
                    .builder
                    .build_alloca(param_type, name)
                    .map_err(|e| format!("Failed to allocate parameter: {:?}", e))?;

                self.builder
                    .build_store(alloca, param_val)
                    .map_err(|e| format!("Failed to store parameter: {:?}", e))?;

                self.variables.insert(name.clone(), alloca);
                self.variable_types.insert(name.clone(), param_type);
            }
        }

        // Compile function body
        self.compile_block(&function.body)?;

        // Add return if missing
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            if self.is_void_type(&function.header.return_type) {
                self.builder
                    .build_return(None)
                    .map_err(|e| format!("Failed to build return: {:?}", e))?;
            } else {
                // Return default value
                let ret_ty = self.convert_type(&function.header.return_type)?;
                let zero = self.get_default_value(ret_ty);
                self.builder
                    .build_return(Some(&zero))
                    .map_err(|e| format!("Failed to build return: {:?}", e))?;
            }
        }

        // Restore previous scope
        self.variables = prev_vars;
        self.variable_types = prev_types;

        Ok(())
    }

    fn get_default_value(&self, ty: BasicTypeEnum<'ctx>) -> BasicValueEnum<'ctx> {
        match ty {
            BasicTypeEnum::IntType(int_ty) => int_ty.const_zero().into(),
            BasicTypeEnum::FloatType(float_ty) => float_ty.const_zero().into(),
            BasicTypeEnum::PointerType(ptr_ty) => ptr_ty.const_null().into(),
            BasicTypeEnum::StructType(struct_ty) => struct_ty.const_zero().into(),
            _ => panic!("Unsupported type for default value"),
        }
    }

    // ================= Global Variables =================

    fn compile_global_var(&mut self, var: &VarDecl) -> Result<(), String> {
        let value = self.compile_expr(&var.value)?;
        let global =
            self.module
                .add_global(value.get_type(), Some(AddressSpace::default()), &var.name);
        global.set_initializer(&value);

        Ok(())
    }

    // ================= Block & Statements =================

    fn compile_block(&mut self, block: &Block) -> Result<(), String> {
        for stmt in &block.statements {
            self.compile_stmt(stmt)?;

            // Stop if block is terminated
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_some()
            {
                break;
            }
        }
        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Var(var) => self.compile_var_decl(var),
            Stmt::Return(expr) => self.compile_return(expr),
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                Ok(())
            }
            Stmt::Block(block) => self.compile_block(block),
            Stmt::If {
                cond,
                then_block,
                else_block,
            } => self.compile_if(cond, then_block, else_block.as_ref()),
            Stmt::While { cond, body } => self.compile_while(cond, body),
            Stmt::For { name, iter, body } => self.compile_for(name, iter, body),
        }
    }

    fn compile_var_decl(&mut self, var: &VarDecl) -> Result<(), String> {
        let value = self.compile_expr(&var.value)?;

        let ty = if let Some(ref var_ty) = var.ty {
            self.convert_type(var_ty)?
        } else {
            value.get_type()
        };

        let alloca = self
            .builder
            .build_alloca(ty, &var.name)
            .map_err(|e| format!("Failed to allocate variable: {:?}", e))?;

        self.builder
            .build_store(alloca, value)
            .map_err(|e| format!("Failed to store variable: {:?}", e))?;

        self.variables.insert(var.name.clone(), alloca);
        self.variable_types.insert(var.name.clone(), ty);
        Ok(())
    }

    fn compile_return(&mut self, expr: &Option<Expr>) -> Result<(), String> {
        if let Some(e) = expr {
            let value = self.compile_expr(e)?;
            self.builder
                .build_return(Some(&value))
                .map_err(|e| format!("Failed to build return: {:?}", e))?;
        } else {
            self.builder
                .build_return(None)
                .map_err(|e| format!("Failed to build return: {:?}", e))?;
        }
        Ok(())
    }

    fn compile_if(
        &mut self,
        cond: &Expr,
        then_block: &Block,
        else_block: Option<&Block>,
    ) -> Result<(), String> {
        let cond_val = self.compile_expr(cond)?;
        let cond_val = cond_val.into_int_value();

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        let then_bb = self.context.append_basic_block(parent_fn, "then");
        let else_bb = self.context.append_basic_block(parent_fn, "else");
        let merge_bb = self.context.append_basic_block(parent_fn, "ifcont");

        self.builder
            .build_conditional_branch(cond_val, then_bb, else_bb)
            .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

        // Then block
        self.builder.position_at_end(then_bb);
        self.compile_block(then_block)?;
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| format!("Failed to build branch: {:?}", e))?;
        }

        // Else block
        self.builder.position_at_end(else_bb);
        if let Some(eb) = else_block {
            self.compile_block(eb)?;
        }
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| format!("Failed to build branch: {:?}", e))?;
        }

        self.builder.position_at_end(merge_bb);
        Ok(())
    }

    fn compile_while(&mut self, cond: &Expr, body: &Block) -> Result<(), String> {
        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        let cond_bb = self.context.append_basic_block(parent_fn, "while.cond");
        let body_bb = self.context.append_basic_block(parent_fn, "while.body");
        let end_bb = self.context.append_basic_block(parent_fn, "while.end");

        self.builder
            .build_unconditional_branch(cond_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        // Condition
        self.builder.position_at_end(cond_bb);
        let cond_val = self.compile_expr(cond)?.into_int_value();
        self.builder
            .build_conditional_branch(cond_val, body_bb, end_bb)
            .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

        // Body
        self.builder.position_at_end(body_bb);
        self.compile_block(body)?;
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            self.builder
                .build_unconditional_branch(cond_bb)
                .map_err(|e| format!("Failed to build branch: {:?}", e))?;
        }

        self.builder.position_at_end(end_bb);
        Ok(())
    }

    fn compile_for(&mut self, _name: &str, _iter: &Expr, _body: &Block) -> Result<(), String> {
        // For loop implementation would require iterator protocol
        Err("For loops not yet implemented".to_string())
    }

    // ================= Expressions =================

    fn compile_expr(&mut self, expr: &Expr) -> Result<BasicValueEnum<'ctx>, String> {
        match expr {
            Expr::Literal(lit) => self.compile_literal(lit),
            Expr::Identifier(name) => self.compile_identifier(name),
            Expr::Assign(target, value) => self.compile_assign(target, value),
            Expr::Binary(left, op, right) => self.compile_binary(left, op, right),
            Expr::Unary(op, expr) => self.compile_unary(op, expr),
            Expr::Call(func, args) => self.compile_call(func, args),
            Expr::Field(obj, field) => self.compile_field(obj, field),
            Expr::Index(arr, idx) => self.compile_index(arr, idx),
            Expr::Function(_) => Err("Anonymous functions not yet supported".to_string()),
        }
    }

    fn compile_literal(&self, lit: &Literal) -> Result<BasicValueEnum<'ctx>, String> {
        match lit {
            Literal::Int(s) => {
                let val: i64 = s.parse().map_err(|e| format!("Invalid integer: {}", e))?;
                Ok(self.context.i64_type().const_int(val as u64, true).into())
            }
            Literal::Float(s) => {
                let val: f64 = s.parse().map_err(|e| format!("Invalid float: {}", e))?;
                Ok(self.context.f64_type().const_float(val).into())
            }
            Literal::Bool(b) => Ok(self.context.bool_type().const_int(*b as u64, false).into()),
            Literal::Char(c) => Ok(self.context.i8_type().const_int(*c as u64, false).into()),
            Literal::String(s) => {
                let str_val = self
                    .builder
                    .build_global_string_ptr(s, "str")
                    .map_err(|e| format!("Failed to build string: {:?}", e))?;
                Ok(str_val.as_pointer_value().into())
            }
        }
    }

    fn compile_identifier(&mut self, name: &str) -> Result<BasicValueEnum<'ctx>, String> {
        let ptr = self
            .variables
            .get(name)
            .ok_or_else(|| format!("Undefined variable: {}", name))?;

        // Get the type that was stored with this variable
        let load_type = if let Some(alloca_type) = self.variable_types.get(name) {
            *alloca_type
        } else {
            return Err(format!("Unknown type for variable: {}", name));
        };

        self.builder
            .build_load(load_type, *ptr, name)
            .map_err(|e| format!("Failed to load variable: {:?}", e))
    }

    fn compile_assign(
        &mut self,
        target: &Expr,
        value: &Expr,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let val = self.compile_expr(value)?;

        match target {
            Expr::Identifier(name) => {
                let ptr = self
                    .variables
                    .get(name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))?;

                self.builder
                    .build_store(*ptr, val)
                    .map_err(|e| format!("Failed to store value: {:?}", e))?;

                Ok(val)
            }
            _ => Err("Invalid assignment target".to_string()),
        }
    }

    fn compile_binary(
        &mut self,
        left: &Expr,
        op: &BinOp,
        right: &Expr,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let lhs = self.compile_expr(left)?;
        let rhs = self.compile_expr(right)?;

        match (lhs, rhs) {
            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                let result: BasicValueEnum = match op {
                    BinOp::Add => self.builder.build_int_add(l, r, "add").map(|v| v.into()),
                    BinOp::Sub => self.builder.build_int_sub(l, r, "sub").map(|v| v.into()),
                    BinOp::Mul => self.builder.build_int_mul(l, r, "mul").map(|v| v.into()),
                    BinOp::Div => self
                        .builder
                        .build_int_signed_div(l, r, "div")
                        .map(|v| v.into()),

                    BinOp::Eq => self
                        .builder
                        .build_int_compare(IntPredicate::EQ, l, r, "eq")
                        .map(|v| v.into()),
                    BinOp::Ne => self
                        .builder
                        .build_int_compare(IntPredicate::NE, l, r, "ne")
                        .map(|v| v.into()),
                    BinOp::Lt => self
                        .builder
                        .build_int_compare(IntPredicate::SLT, l, r, "lt")
                        .map(|v| v.into()),
                    BinOp::Gt => self
                        .builder
                        .build_int_compare(IntPredicate::SGT, l, r, "gt")
                        .map(|v| v.into()),
                    BinOp::Le => self
                        .builder
                        .build_int_compare(IntPredicate::SLE, l, r, "le")
                        .map(|v| v.into()),
                    BinOp::Ge => self
                        .builder
                        .build_int_compare(IntPredicate::SGE, l, r, "ge")
                        .map(|v| v.into()),

                    BinOp::And => self.builder.build_and(l, r, "and").map(|v| v.into()),
                    BinOp::Or => self.builder.build_or(l, r, "or").map(|v| v.into()),
                }
                .map_err(|e| format!("Failed to build int op: {:?}", e))?;

                Ok(result)
            }

            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                let result: BasicValueEnum = match op {
                    BinOp::Add => self.builder.build_float_add(l, r, "fadd").map(|v| v.into()),
                    BinOp::Sub => self.builder.build_float_sub(l, r, "fsub").map(|v| v.into()),
                    BinOp::Mul => self.builder.build_float_mul(l, r, "fmul").map(|v| v.into()),
                    BinOp::Div => self.builder.build_float_div(l, r, "fdiv").map(|v| v.into()),

                    BinOp::Eq => self
                        .builder
                        .build_float_compare(FloatPredicate::OEQ, l, r, "feq")
                        .map(|v| v.into()),
                    BinOp::Ne => self
                        .builder
                        .build_float_compare(FloatPredicate::ONE, l, r, "fne")
                        .map(|v| v.into()),
                    BinOp::Lt => self
                        .builder
                        .build_float_compare(FloatPredicate::OLT, l, r, "flt")
                        .map(|v| v.into()),
                    BinOp::Gt => self
                        .builder
                        .build_float_compare(FloatPredicate::OGT, l, r, "fgt")
                        .map(|v| v.into()),
                    BinOp::Le => self
                        .builder
                        .build_float_compare(FloatPredicate::OLE, l, r, "fle")
                        .map(|v| v.into()),
                    BinOp::Ge => self
                        .builder
                        .build_float_compare(FloatPredicate::OGE, l, r, "fge")
                        .map(|v| v.into()),

                    _ => return Err(format!("Invalid float operation: {:?}", op)),
                }
                .map_err(|e| format!("Failed to build float op: {:?}", e))?;

                Ok(result)
            }

            _ => Err("Type mismatch in binary operation".to_string()),
        }
    }

    fn compile_unary(&mut self, op: &UnOp, expr: &Expr) -> Result<BasicValueEnum<'ctx>, String> {
        let val = self.compile_expr(expr)?;

        match op {
            UnOp::Neg => {
                if let BasicValueEnum::IntValue(i) = val {
                    Ok(self
                        .builder
                        .build_int_neg(i, "neg")
                        .map_err(|e| format!("Failed to build neg: {:?}", e))?
                        .into())
                } else if let BasicValueEnum::FloatValue(f) = val {
                    Ok(self
                        .builder
                        .build_float_neg(f, "fneg")
                        .map_err(|e| format!("Failed to build fneg: {:?}", e))?
                        .into())
                } else {
                    Err("Invalid type for negation".to_string())
                }
            }
            UnOp::Not => {
                if let BasicValueEnum::IntValue(i) = val {
                    Ok(self
                        .builder
                        .build_not(i, "not")
                        .map_err(|e| format!("Failed to build not: {:?}", e))?
                        .into())
                } else {
                    Err("Invalid type for not operation".to_string())
                }
            }
        }
    }

    fn compile_call(
        &mut self,
        func_expr: &Expr,
        args: &[Expr],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // Extract function name
        let func_name = if let Expr::Identifier(name) = func_expr {
            name
        } else {
            return Err("Only direct function calls supported".to_string());
        };

        // Copy out the FunctionValue to avoid immutable borrow lingering
        let function = *self
            .functions
            .get(func_name)
            .ok_or_else(|| format!("Undefined function: {}", func_name))?;

        // Compile all arguments
        let mut compiled_args = Vec::new();
        for arg in args {
            let val = self.compile_expr(arg)?; // mutable borrow of self is fine now
            compiled_args.push(val.into()); // convert to BasicMetadataValueEnum
        }

        // Build the call
        let call_site = self
            .builder
            .build_call(function, &compiled_args, "call")
            .map_err(|e| format!("Failed to build call: {:?}", e))?;

        // Match old ValueKind enum
        match call_site.try_as_basic_value() {
            ValueKind::Basic(value) => Ok(value),
            ValueKind::Instruction(_) => Err("Function returned void".to_string()),
        }
    }
    // fn compile_call(
    //     &mut self,
    //     func_expr: &Expr,
    //     args: &[Expr],
    // ) -> Result<BasicValueEnum<'ctx>, String> {
    //     let func_name = if let Expr::Identifier(name) = func_expr {
    //         name
    //     } else {
    //         return Err("Only direct function calls supported".to_string());
    //     };
    //
    //     let function = self
    //         .functions
    //         .get(func_name)
    //         .ok_or_else(|| format!("Undefined function: {}", func_name))?;
    //
    //     let mut compiled_args = Vec::new();
    //     for arg in args {
    //         let val = self.compile_expr(arg)?;
    //         compiled_args.push(val.into());
    //     }
    //
    //     let call_site = self
    //         .builder
    //         .build_call(*function, &compiled_args, "call")
    //         .map_err(|e| format!("Failed to build call: {:?}", e))?;
    //
    //     match call_site.try_as_basic_value() {
    //         ValueKind::Basic(value) => Ok(value),
    //         ValueKind::Instruction(_) => Err("Function returned void".to_string()),
    //     }
    // }

    fn compile_field(&mut self, _obj: &Expr, _field: &str) -> Result<BasicValueEnum<'ctx>, String> {
        Err("Struct field access not yet implemented".to_string())
    }

    fn compile_index(&mut self, _arr: &Expr, _idx: &Expr) -> Result<BasicValueEnum<'ctx>, String> {
        Err("Array indexing not yet implemented".to_string())
    }

    // ================= Output =================

    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }

    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }
}
