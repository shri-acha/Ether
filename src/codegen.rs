use crate::parser::EnumDef;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Linkage;
use inkwell::module::Module;
use inkwell::types::StructType;
use inkwell::types::{BasicType, BasicTypeEnum};
use inkwell::values::ValueKind;
use inkwell::values::{BasicValueEnum, FunctionValue, PointerValue};
use inkwell::{AddressSpace, FloatPredicate, IntPredicate};
use std::collections::HashMap;

use crate::parser::{
    BinOp, Block, Declaration, Expr, Function, FunctionHeader, Literal, MatchArm, Pattern, Program,
    Stmt, StructDef, Type, UnOp, VarDecl,
};

struct EnumInfo<'ctx> {
    enum_type: StructType<'ctx>,
    variants: Vec<(String, Option<BasicTypeEnum<'ctx>>)>,
}
pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,

    // Symbol tables
    variables: HashMap<String, PointerValue<'ctx>>,
    variable_types: HashMap<String, BasicTypeEnum<'ctx>>,
    functions: HashMap<String, FunctionValue<'ctx>>,
    structs: HashMap<String, BasicTypeEnum<'ctx>>,
    enums: HashMap<String, EnumInfo<'ctx>>,
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
            enums: HashMap::new(),
        }
    }

    //  Main Entry Point

    pub fn compile_program(&mut self, program: &Program) -> Result<(), String> {
        // First pass: declare all structs
        for decl in &program.declarations {
            if let Declaration::Struct(s) = decl {
                self.declare_struct(s)?;
            }
            if let Declaration::Enum(e) = decl {
                self.declare_enum(e)?;
            }
        }
        self.declare_runtime_linked_functions();

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
                Declaration::Enum(_) => {}
            }
        }

        Ok(())
    }

    //  Type Conversion

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
                "void" => Ok(self.context.struct_type(&[], false).into()),
                _ => Err(format!("Unknown primitive type: {}", name)),
            },
            Type::Array(inner) => {
                let inner_ty = self.convert_type(inner)?;
                Ok(inner_ty.ptr_type(AddressSpace::default()).into())
            }
            Type::Custom(name) => {
                if let Some(enum_info) = self.enums.get(name) {
                    Ok(enum_info.enum_type.into())
                } else {
                    self.structs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| format!("Unknown custom type: {}", name))
                }
            }
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

    //  Struct Handling

    fn declare_struct(&mut self, struct_def: &StructDef) -> Result<(), String> {
        let field_types: Result<Vec<_>, String> = struct_def
            .fields
            .iter()
            .map(|(_, ty)| self.convert_type(ty))
            .collect();

        let field_types = field_types?;
        let struct_type = self.context.struct_type(&field_types, false);
        self.structs
            .insert(struct_def.name.clone(), struct_type.into());

        Ok(())
    }
    //  Enum Handling
    //
    // This has a bit of complications while trying to implement
    // llvm doesn't have native support for enums/tagged unions
    // so, we'll have to implement { discriminant , payload }
    // so that the discrimant is encoded with number of variants
    // and the payload is able to handle the size of the maximum variant
    // and also the alignment of the value as aparently llvm doesn't support
    // alignments. As cpu while fetching from memory has to align with types.
    //
    // works now!
    //
    fn declare_enum(&mut self, enum_def: &EnumDef) -> Result<(), String> {
        let mut variants = Vec::new();
        let mut max_size = 0u64;
        let mut max_align_type: Option<BasicTypeEnum<'ctx>> = None;

        for (field_name, field_type_opt) in &enum_def.fields {
            if let Some(field_type) = field_type_opt {
                let basic_ty = self.convert_type(field_type)?;

                let size = basic_ty
                    .size_of()
                    .map(|s| s.get_zero_extended_constant())
                    .unwrap_or(Some(0))
                    .unwrap_or(0);

                if size > max_size {
                    max_size = size;
                    max_align_type = Some(basic_ty);
                }

                variants.push((field_name.clone(), Some(basic_ty)));
            } else {
                // Unit variant (no payload)
                variants.push((field_name.clone(), None));
            }
        }

        let num_variants = enum_def.fields.len();
        let discriminant_type = if num_variants <= 256 {
            self.context.i8_type().into()
        } else if num_variants <= 65536 {
            self.context.i16_type().into()
        } else {
            self.context.i32_type().into()
        };

        let payload_type = max_align_type.unwrap_or_else(|| self.context.i8_type().into());

        let enum_struct = self.context.opaque_struct_type(&enum_def.name);
        enum_struct.set_body(&[discriminant_type, payload_type], false);

        let enum_info = EnumInfo {
            enum_type: enum_struct,
            variants,
        };

        self.enums.insert(enum_def.name.clone(), enum_info);

        Ok(())
    }

    fn create_enum_value(
        &mut self,
        enum_name: &str,
        variant_name: &str,
        payload_value: Option<BasicValueEnum<'ctx>>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let enum_info = self
            .enums
            .get(enum_name)
            .ok_or_else(|| format!("Unknown enum: {}", enum_name))?;

        let variant_idx = enum_info
            .variants
            .iter()
            .position(|(name, _)| name == variant_name)
            .ok_or_else(|| format!("Unknown variant: {}", variant_name))?;

        let num_variants = enum_info.variants.len();
        let discriminant_val: BasicValueEnum = if num_variants <= 256 {
            self.context
                .i8_type()
                .const_int(variant_idx as u64, false)
                .into()
        } else if num_variants <= 65536 {
            self.context
                .i16_type()
                .const_int(variant_idx as u64, false)
                .into()
        } else {
            self.context
                .i64_type()
                .const_int(variant_idx as u64, false)
                .into()
        };

        let enum_type = enum_info.enum_type;
        let payload_field_type = enum_type
            .get_field_type_at_index(1)
            .ok_or("Invalid enum structure")?;

        let payload_val = if let Some(val) = payload_value {
            val
        } else {
            self.get_default_value(payload_field_type)
        };

        let mut struct_val = enum_type.get_undef();
        struct_val = self
            .builder
            .build_insert_value(struct_val, discriminant_val, 0, "with_discriminant")
            .map_err(|e| format!("Failed to insert discriminant: {:?}", e))?
            .into_struct_value();
        struct_val = self
            .builder
            .build_insert_value(struct_val, payload_val, 1, "with_payload")
            .map_err(|e| format!("Failed to insert payload: {:?}", e))?
            .into_struct_value();

        Ok(struct_val.into())
    }

    fn extract_enum_discriminant(
        &mut self,
        enum_value: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let struct_val = enum_value.into_struct_value();
        self.builder
            .build_extract_value(struct_val, 0, "discriminant")
            .map_err(|e| format!("Failed to extract discriminant: {:?}", e))
    }

    fn extract_enum_payload(
        &mut self,
        enum_value: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let struct_val = enum_value.into_struct_value();
        self.builder
            .build_extract_value(struct_val, 1, "payload")
            .map_err(|e| format!("Failed to extract payload: {:?}", e))
    }

    //  Function Handling

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
        let fn_val;
        if name == "main" {
            fn_val = self
                .module
                .add_function(name, fn_type, Some(Linkage::External));
            fn_val.set_call_conventions(0);
        } else {
            fn_val = self
                .module
                .add_function(name, fn_type, Some(Linkage::Internal));
        }

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

    // ================= Runtime Linked Functions =================

    fn declare_runtime_linked_functions(&mut self) {
        let i8_type = self.context.i8_type();
        let i64_type = self.context.i64_type();
        let i8_ptr_type = i8_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();

        // Declare __Eth_print_str
        let print_fn_str_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        self.module
            .add_function("__Eth_print_str", print_fn_str_type, None);

        // Declare __Eth_print_i64
        let print_fn_i64_type = void_type.fn_type(&[i64_type.into()], false);
        self.module
            .add_function("__Eth_print_i64", print_fn_i64_type, None);

        // Declare __Eth_read
        let read_fn_type = i8_ptr_type.fn_type(&[], false);
        self.module
            .add_function("__Eth_read", read_fn_type, Some(Linkage::External));
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

    //  Block & Statements

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
            Stmt::Match { scrutinee, arms } => self.compile_match(scrutinee, arms),
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

    fn compile_for(&mut self, name: &str, iter: &Expr, body: &Block) -> Result<(), String> {
        // for (i in range_expr) { body }
        // range_expr should evaluate to an array or range

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        // Create basic blocks for the loop
        let init_bb = self.context.append_basic_block(parent_fn, "for.init");
        let cond_bb = self.context.append_basic_block(parent_fn, "for.cond");
        let body_bb = self.context.append_basic_block(parent_fn, "for.body");
        let inc_bb = self.context.append_basic_block(parent_fn, "for.inc");
        let end_bb = self.context.append_basic_block(parent_fn, "for.end");

        // Jump to initialization
        self.builder
            .build_unconditional_branch(init_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        // === INITIALIZATION BLOCK ===
        self.builder.position_at_end(init_bb);

        // Extract range bounds directly without compiling the entire expression
        // (compile_expr would reject the range operator)

        // Determine if it's an array (pointer) iteration or a range iteration

        // For now, let's implement array iteration
        // Assume iter_value is a pointer to the start of an array
        // We need to know the length - for this implementation,
        // we'll assume a convention where arrays are passed with length

        // Create loop counter variable
        let counter_type = self.context.i64_type();
        let counter = self
            .builder
            .build_alloca(counter_type, "for.counter")
            .map_err(|e| format!("Failed to allocate counter: {:?}", e))?;

        // Initialize counter to 0
        self.builder
            .build_store(counter, counter_type.const_zero())
            .map_err(|e| format!("Failed to store counter: {:?}", e))?;

        // For array iteration, we need the array length
        // This is a simplified implementation - in a real compiler,
        // you'd track array lengths separately or use a struct { ptr, len }

        // For demonstration, let's assume we're iterating a fixed range
        // We'll check if the iterator is actually a range expression
        // If iter is a function call like "range(0, 10)", we handle it specially

        let (start_val, end_val) = self.extract_range_bounds(iter)?;

        // Store the loop bounds
        let start_var = self
            .builder
            .build_alloca(counter_type, "for.start")
            .map_err(|e| format!("Failed to allocate start: {:?}", e))?;
        self.builder
            .build_store(start_var, start_val)
            .map_err(|e| format!("Failed to store start: {:?}", e))?;

        let end_var = self
            .builder
            .build_alloca(counter_type, "for.end")
            .map_err(|e| format!("Failed to allocate end: {:?}", e))?;
        self.builder
            .build_store(end_var, end_val)
            .map_err(|e| format!("Failed to store end: {:?}", e))?;

        // Set counter to start value
        self.builder
            .build_store(counter, start_val)
            .map_err(|e| format!("Failed to initialize counter: {:?}", e))?;

        // Jump to condition check
        self.builder
            .build_unconditional_branch(cond_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        // === CONDITION BLOCK ===
        self.builder.position_at_end(cond_bb);

        // Load current counter value
        let current_counter = self
            .builder
            .build_load(counter_type, counter, "counter.val")
            .map_err(|e| format!("Failed to load counter: {:?}", e))?
            .into_int_value();

        // Load end value
        let end_value = self
            .builder
            .build_load(counter_type, end_var, "end.val")
            .map_err(|e| format!("Failed to load end: {:?}", e))?
            .into_int_value();

        // Check if counter < end
        let cond = self
            .builder
            .build_int_compare(IntPredicate::SLT, current_counter, end_value, "for.cond")
            .map_err(|e| format!("Failed to build comparison: {:?}", e))?;

        // Branch based on condition
        self.builder
            .build_conditional_branch(cond, body_bb, end_bb)
            .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

        // === BODY BLOCK ===
        self.builder.position_at_end(body_bb);

        // Save current variable state
        let prev_vars = self.variables.clone();
        let prev_types = self.variable_types.clone();

        // Create the loop variable and bind it to the current counter value
        let loop_var_type = counter_type.as_basic_type_enum();
        let loop_var = self
            .builder
            .build_alloca(loop_var_type, name)
            .map_err(|e| format!("Failed to allocate loop variable: {:?}", e))?;

        // Load current counter and store in loop variable
        let counter_val = self
            .builder
            .build_load(counter_type, counter, "counter.load")
            .map_err(|e| format!("Failed to load counter: {:?}", e))?;

        self.builder
            .build_store(loop_var, counter_val)
            .map_err(|e| format!("Failed to store loop variable: {:?}", e))?;

        // Add loop variable to symbol table
        self.variables.insert(name.to_string(), loop_var);
        self.variable_types.insert(name.to_string(), loop_var_type);

        // Compile the loop body
        self.compile_block(body)?;

        // Restore variable state (remove loop variable from scope)
        self.variables = prev_vars;
        self.variable_types = prev_types;

        // Jump to increment if no terminator (return/break)
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            self.builder
                .build_unconditional_branch(inc_bb)
                .map_err(|e| format!("Failed to build branch: {:?}", e))?;
        }

        // === INCREMENT BLOCK ===
        self.builder.position_at_end(inc_bb);

        // Load counter
        let current = self
            .builder
            .build_load(counter_type, counter, "counter.inc.load")
            .map_err(|e| format!("Failed to load counter: {:?}", e))?
            .into_int_value();

        // Increment by 1
        let incremented = self
            .builder
            .build_int_add(current, counter_type.const_int(1, false), "counter.inc")
            .map_err(|e| format!("Failed to increment counter: {:?}", e))?;

        // Store back
        self.builder
            .build_store(counter, incremented)
            .map_err(|e| format!("Failed to store incremented counter: {:?}", e))?;

        // Jump back to condition check
        self.builder
            .build_unconditional_branch(cond_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        // === END BLOCK ===
        self.builder.position_at_end(end_bb);

        Ok(())
    }

    fn compile_match(&mut self, scrutinee: &Expr, arms: &[MatchArm]) -> Result<(), String> {
        if arms.is_empty() {
            return Err("Match statement must have at least one arm".to_string());
        }

        let scrutinee_val = self.compile_expr(scrutinee)?;
        
        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        let merge_bb = self.context.append_basic_block(parent_fn, "match.end");

        // Compile each arm as a chain of if-else conditions
        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;
            
            let arm_body_bb = self
                .context
                .append_basic_block(parent_fn, &format!("match.arm{}", i));
            let next_check_bb = if !is_last {
                self.context
                    .append_basic_block(parent_fn, &format!("match.check{}", i + 1))
            } else {
                merge_bb
            };

            // Check if pattern matches
            let matches = self.compile_pattern_match(scrutinee_val, &arm.pattern)?;

            self.builder
                .build_conditional_branch(matches, arm_body_bb, next_check_bb)
                .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

            // Compile arm body
            self.builder.position_at_end(arm_body_bb);
            self.compile_block(&arm.body)?;
            
            // Jump to merge if no terminator
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

            // Position for next check
            if !is_last {
                self.builder.position_at_end(next_check_bb);
            }
        }

        self.builder.position_at_end(merge_bb);
        Ok(())
    }

    fn compile_pattern_match(
        &mut self,
        scrutinee_val: BasicValueEnum<'ctx>,
        pattern: &Pattern,
    ) -> Result<inkwell::values::IntValue<'ctx>, String> {
        match pattern {
            Pattern::Literal(lit) => {
                let pattern_val = self.compile_literal(lit)?;
                
                // Compare scrutinee with pattern value
                match (scrutinee_val, pattern_val) {
                    (BasicValueEnum::IntValue(scrutinee_int), BasicValueEnum::IntValue(pattern_int)) => {
                        self.builder
                            .build_int_compare(IntPredicate::EQ, scrutinee_int, pattern_int, "match.cmp")
                            .map_err(|e| format!("Failed to compare integers: {:?}", e))
                    }
                    (BasicValueEnum::FloatValue(scrutinee_float), BasicValueEnum::FloatValue(pattern_float)) => {
                        self.builder
                            .build_float_compare(FloatPredicate::OEQ, scrutinee_float, pattern_float, "match.cmp")
                            .map_err(|e| format!("Failed to compare floats: {:?}", e))
                    }
                    _ => Err("Type mismatch in pattern matching".to_string()),
                }
            }
            Pattern::Identifier(_) => {
                // Wildcard pattern - always matches
                Ok(self.context.bool_type().const_int(1, false))
            }
            Pattern::EnumVariant(enum_name, variant_name) => {
                // Get enum info
                let enum_info = self.enums.get(enum_name).ok_or_else(|| {
                    format!("Enum '{}' not found", enum_name)
                })?;

                // Find variant index
                let variant_idx = enum_info
                    .variants
                    .iter()
                    .position(|(name, _)| name == variant_name)
                    .ok_or_else(|| {
                        format!("Variant '{}' not found in enum '{}'", variant_name, enum_name)
                    })?;

                // Extract discriminant from enum value
                // The discriminant is an IntValue but extract_enum_discriminant returns BasicValueEnum
                // for consistency with other extract methods
                let discriminant = self.extract_enum_discriminant(scrutinee_val)?;
                let tag_val = discriminant.into_int_value();

                let expected_tag = self.context.i32_type().const_int(variant_idx as u64, false);
                
                self.builder
                    .build_int_compare(IntPredicate::EQ, tag_val, expected_tag, "match.enum.cmp")
                    .map_err(|e| format!("Failed to compare enum tags: {:?}", e))
            }
        }
    }

    // Helper method to extract range bounds from iterator expression
    fn extract_range_bounds(
        &mut self,
        iter: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, BasicValueEnum<'ctx>), String> {
        // Check if iter is a binary range operator: start..end
        if let Expr::Binary(start_expr, op, end_expr) = iter {
            if op == &BinOp::Range {
                // Compile the bounds directly without going through compile_expr
                // which would reject the Range operator
                let start = match start_expr.as_ref() {
                    Expr::Literal(lit) => self.compile_literal(lit)?,
                    Expr::Identifier(name) => self.compile_identifier(name)?,
                    other => self.compile_expr(other)?,
                };
                let end = match end_expr.as_ref() {
                    Expr::Literal(lit) => self.compile_literal(lit)?,
                    Expr::Identifier(name) => self.compile_identifier(name)?,
                    other => self.compile_expr(other)?,
                };
                return Ok((start, end));
            }
        }

        // // Check if iter is a function call to "range"
        // if let Expr::Call(func_expr, args) = iter {
        //     if let Expr::Identifier(func_name) = func_expr.as_ref() {
        //         if func_name == "range" {
        //             // Handle range(start, end) or range(end)
        //             match args.len() {
        //                 1 => {
        //                     // range(n) means 0..n
        //                     let end = self.compile_expr(&args[0])?;
        //                     let start = self.context.i64_type().const_zero().into();
        //                     return Ok((start, end));
        //                 }
        //                 2 => {
        //                     // range(start, end) means start..end
        //                     let start = self.compile_expr(&args[0])?;
        //                     let end = self.compile_expr(&args[1])?;
        //                     return Ok((start, end));
        //                 }
        //                 _ => return Err("range() takes 1 or 2 arguments".to_string()),
        //             }
        //         }
        //     }
        // }

        // If not a range call or range operator, try to treat as an array
        // For arrays, we'd need length information
        // This is a simplified fallback - assume it's a range 0..10
        Err("For loop iterator must be a range (e.g., 0..10), range() call, or array (array iteration not fully implemented)".to_string())
    }

    //  Expressions

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
            Expr::EnumVariant(enum_name, variant_name) => {
                self.compile_enum_variant(enum_name, variant_name)
            }
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
        // Range operator should only be used in for loops, not as a standalone expression
        if op == &BinOp::Range {
            return Err("Range operator (..) can only be used in for loops".to_string());
        }

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
                    BinOp::Range => unreachable!("Range should be handled above"),
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

        if func_name == "print" || func_name == "__Eth_print" {
            if args.len() != 1 {
                return Err("print expects 1 argument".to_string());
            }

            let print_fn_str = self
                .module
                .get_function("__Eth_print_str")
                .ok_or("__Eth_print variants should be declared")?;

            let print_fn_i64 = self
                .module
                .get_function("__Eth_print_i64")
                .ok_or("__Eth_print variants should be declared")?;

            let arg_val = self.compile_expr(&args[0])?;
            match &args[0] {
                Expr::Literal(lit) => match lit {
                    Literal::String(_s) => {
                        self.builder
                            .build_call(print_fn_str, &[arg_val.into()], "print_call")
                            .map_err(|e| format!("Failed to build print call: {:?}", e))?;

                        let unit_type = self.context.struct_type(&[], false);
                        return Ok(unit_type.const_zero().into());
                    }
                    Literal::Int(_s) => {
                        self.builder
                            .build_call(print_fn_i64, &[arg_val.into()], "print_call")
                            .map_err(|e| format!("Failed to build print call: {:?}", e))?;

                        let unit_type = self.context.struct_type(&[], false);
                        return Ok(unit_type.const_zero().into());
                    }
                    _ => {
                        return Err("Missing implementation for the type!".to_string());
                    }
                },
                Expr::Identifier(identifier_name) => {
                    let _variable_value = self
                        .variables
                        .get(identifier_name)
                        .ok_or(format!("Failed to find correspoding variable"))?;
                    let variable_type = self
                        .variable_types
                        .get(identifier_name)
                        .ok_or(format!("Failed to any value assigned to the variable"))?;
                    match variable_type {
                        BasicTypeEnum::IntType(_) => {
                            self.builder
                                .build_call(print_fn_i64, &[arg_val.into()], "print_call")
                                .map_err(|e| format!("Failed to build print call: {:?}", e))?;

                            let unit_type = self.context.struct_type(&[], false);
                            return Ok(unit_type.const_zero().into());
                        }
                        BasicTypeEnum::PointerType(_) => {
                            self.builder
                                .build_call(print_fn_str, &[arg_val.into()], "print_call")
                                .map_err(|e| format!("Failed to build print call: {:?}", e))?;

                            let unit_type = self.context.struct_type(&[], false);
                            return Ok(unit_type.const_zero().into());
                        }
                        _ => {
                            return Err("Missing implementation for the type!".to_string());
                        }
                    }
                }
                _ => {
                    return Err("Missing implementation for the type!".to_string());
                }
            }
        }

        if func_name == "read" || func_name == "__Eth_read" {
            let read_fn = self
                .module
                .get_function("__Eth_read")
                .ok_or("__Eth_read should be declared")?;

            let call_site = self
                .builder
                .build_call(read_fn, &[], "read_call")
                .map_err(|e| format!("Failed to build print call: {:?}", e))?;

            match call_site.try_as_basic_value() {
                ValueKind::Basic(value) => return Ok(value),
                ValueKind::Instruction(_) => return Err("read should return a value".to_string()),
            }
        }
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

    fn compile_field(&mut self, _obj: &Expr, _field: &str) -> Result<BasicValueEnum<'ctx>, String> {
        Err("Struct field access not yet implemented".to_string())
    }

    fn compile_index(&mut self, _arr: &Expr, _idx: &Expr) -> Result<BasicValueEnum<'ctx>, String> {
        Err("Array indexing not yet implemented".to_string())
    }

    fn compile_enum_variant(
        &mut self,
        enum_name: &str,
        variant_name: &str,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        self.create_enum_value(enum_name, variant_name, None)
    }

    //  Output

    pub fn get_ir(&self) -> String {
        self.module.to_string()
    }

    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }
}
