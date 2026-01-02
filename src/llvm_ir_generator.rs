use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
use inkwell::values::{
    BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue, PointerValue,
};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

use crate::parser::{
    BinOp, Block, Declaration, Expr, Function, FunctionHeader, Import, Literal, Program, Stmt,
    StructDef, Type, UnOp, VarDecl,
};

// Example AST nodes
// enum Expr {
//     Number(i64),
//     Add(Box<Expr>, Box<Expr>),
//     Multiply(Box<Expr>, Box<Expr>),
// }
//
pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,

    // Symbol tables
    variables: HashMap<String, PointerValue<'ctx>>,
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
            functions: HashMap::new(),
            structs: HashMap::new(),
        }
    }

    // Main Entry Point
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

    // Type Conversion
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

    // Struct Handling
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

    // Function Handling
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
        self.variables.clear();

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

    // Block & Statements
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

    // experssions
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

    // literals
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
    // pub fn compile_expr(&self, expr: &Expr) -> IntValue<'ctx> {
    //     match expr {
    //         Expr::Number(n) => self.context.i64_type().const_int(*n as u64, false),
    //         Expr::Add(left, right) => {
    //             let lhs = self.compile_expr(left);
    //             let rhs = self.compile_expr(right);
    //             self.builder
    //                 .build_int_add(lhs, rhs, "addtmp")
    //                 .expect("Failed to build add")
    //         }
    //         Expr::Multiply(left, right) => {
    //             let lhs = self.compile_expr(left);
    //             let rhs = self.compile_expr(right);
    //             self.builder
    //                 .build_int_mul(lhs, rhs, "multmp")
    //                 .expect("Failed to build mul")
    //         }
    //     }
    // }

    // pub fn compile(&self) {
    //     // Create a function: i64 main()
    //     let i64_type = self.context.i64_type();
    //     let fn_type = i64_type.fn_type(&[], false);
    //     let function = self.module.add_function("main", fn_type, None);
    //     let basic_block = self.context.append_basic_block(function, "entry");
    //
    //     self.builder.position_at_end(basic_block);
    //
    //     // Example: compile (2 + 3) * 4
    //     let ast = Expr::Multiply(
    //         Box::new(Expr::Add(
    //             Box::new(Expr::Number(2)),
    //             Box::new(Expr::Number(3)),
    //         )),
    //         Box::new(Expr::Number(4)),
    //     );
    //
    //     let result = self.compile_expr(&ast);
    //     self.builder
    //         .build_return(Some(&result))
    //         .expect("Failed to build return");
    // }
    //
    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }
}

// fn main() {
//     let context = Context::create();
//     let compiler = Compiler::new(&context);
//
//     compiler.compile();
//     compiler.print_ir();
// }
