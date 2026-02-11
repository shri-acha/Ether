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
    BinOp, Block, Declaration, Expr, Function, FunctionHeader, Literal, MatchArm, Pattern,
    Program, Stmt, StructDef, Type, UnOp, VarDecl,
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
    struct_fields: HashMap<String, Vec<String>>,  // ADD THIS LINE: maps struct name -> field names in order
    enums: HashMap<String, EnumInfo<'ctx>>,
    
    // Lambda tracking
    lambda_counter: usize,
    lambda_functions: HashMap<String, String>,
    lambda_signatures: HashMap<String, FunctionHeader>,
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
            struct_fields: HashMap::new(),  // ADD THIS LINE
            enums: HashMap::new(),
            lambda_counter: 0,
            lambda_functions: HashMap::new(),
            lambda_signatures: HashMap::new(),
        }
    }

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
                Declaration::Struct(_) => {}
                Declaration::Enum(_) => {}
            }
        }

        Ok(())
    }

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
                // Function types are represented as pointers to functions
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

        // ADDED: Store field names in order
        let field_names: Vec<String> = struct_def
            .fields
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        self.struct_fields
            .insert(struct_def.name.clone(), field_names);

        Ok(())
    }

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

    let (_, variant_payload_type) = &enum_info.variants[variant_idx];

    // Validate payload against variant requirements
    match (variant_payload_type, &payload_value) {
        (Some(_), None) => {
            return Err(format!(
                "Enum variant {}::{} requires a payload",
                enum_name, variant_name
            ));
        }
        (None, Some(_)) => {
            return Err(format!(
                "Enum variant {}::{} does not accept a payload",
                enum_name, variant_name
            ));
        }
        _ => {} // Both Some or both None - acceptable
    }

    let num_variants = enum_info.variants.len();
    let discriminant_val: BasicValueEnum = if num_variants <= 256 {
        self.context
            .i8_type()
            .const_int(variant_idx as u64, false)
            .into()
    } else if num_variants <= 65516 {
        self.context
            .i16_type()
            .const_int(variant_idx as u64, false)
            .into()
    } else {
        self.context
            .i32_type()
            .const_int(variant_idx as u64, false)
            .into()
    };

    let enum_type = enum_info.enum_type;
    let payload_field_type = enum_type
        .get_field_type_at_index(1)
        .ok_or("Invalid enum structure")?;

    // Handle payload value with proper type conversion if needed
    let payload_val = if let Some(val) = payload_value {
        // If types don't match exactly, try to bitcast/convert
        if val.get_type() != payload_field_type {
            // Check if we need to bitcast (for type-erased storage)
            match (val, payload_field_type) {
                (BasicValueEnum::IntValue(int_val), BasicTypeEnum::IntType(target_int_type)) => {
                    // Convert between different int sizes
                    if int_val.get_type().get_bit_width() < target_int_type.get_bit_width() {
                        self.builder
                            .build_int_z_extend(int_val, target_int_type, "payload_ext")
                            .map_err(|e| format!("Failed to extend int: {:?}", e))?
                            .into()
                    } else if int_val.get_type().get_bit_width() > target_int_type.get_bit_width() {
                        self.builder
                            .build_int_truncate(int_val, target_int_type, "payload_trunc")
                            .map_err(|e| format!("Failed to truncate int: {:?}", e))?
                            .into()
                    } else {
                        val
                    }
                }
                (BasicValueEnum::StructValue(_), BasicTypeEnum::StructType(_)) => {
                    // Structs can be stored directly if sizes match
                    val
                }
                _ => {
                    // For other types, store as-is and let LLVM handle it
                    // This works because we use the largest payload size
                    val
                }
            }
        } else {
            val
        }
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

        // Allocate regular parameters
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

    fn declare_runtime_linked_functions(&mut self) {
        let i8_type = self.context.i8_type();
        let i64_type = self.context.i64_type();
        let i8_ptr_type = i8_type.ptr_type(AddressSpace::default());
        let void_type = self.context.void_type();

        let print_fn_str_type = void_type.fn_type(&[i8_ptr_type.into()], false);
        self.module
            .add_function("__Eth_print_str", print_fn_str_type, None);

        let print_fn_i64_type = void_type.fn_type(&[i64_type.into()], false);
        self.module
            .add_function("__Eth_print_i64", print_fn_i64_type, None);

        let read_fn_type = i8_ptr_type.fn_type(&[], false);
        self.module
            .add_function("__Eth_read", read_fn_type, Some(Linkage::External));
    }

    fn compile_global_var(&mut self, var: &VarDecl) -> Result<(), String> {
        let value = self.compile_expr(&var.value)?;
        let global =
            self.module
                .add_global(value.get_type(), Some(AddressSpace::default()), &var.name);
        global.set_initializer(&value);

        Ok(())
    }

    fn compile_block(&mut self, block: &Block) -> Result<(), String> {
        for stmt in &block.statements {
            self.compile_stmt(stmt)?;

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
        // If the value is a lambda, compile it and track the assignment
        if let Expr::Function(func) = &var.value {
            // Generate unique name for the lambda
            let lambda_name = format!("__lambda_{}", self.lambda_counter);
            self.lambda_counter += 1;

            // Create a function with the generated name
            let mut lambda_func = func.clone();
            lambda_func.header.name = Some(lambda_name.clone());

            // Store the signature
            self.lambda_signatures.insert(lambda_name.clone(), func.header.clone());

            // Declare and compile the function
            self.declare_function(&lambda_func)?;
            self.compile_function(&lambda_func)?;

            // Track that this variable holds this lambda
            self.lambda_functions.insert(var.name.clone(), lambda_name);
            
            // Create a dummy variable to maintain compatibility with rest of system
            // (the actual function is tracked in lambda_functions)
            let ptr_type = self.context.i8_type().ptr_type(AddressSpace::default());
            let dummy_ptr = ptr_type.const_null();
            let alloca = self
                .builder
                .build_alloca(ptr_type.as_basic_type_enum(), &var.name)
                .map_err(|e| format!("Failed to allocate lambda variable: {:?}", e))?;
            self.builder
                .build_store(alloca, dummy_ptr)
                .map_err(|e| format!("Failed to store lambda variable: {:?}", e))?;

            self.variables.insert(var.name.clone(), alloca);
            self.variable_types.insert(var.name.clone(), ptr_type.as_basic_type_enum());
            
            return Ok(());
        }

        // For non-lambda variables, use the normal path
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

        self.builder.position_at_end(cond_bb);
        let cond_val = self.compile_expr(cond)?.into_int_value();
        self.builder
            .build_conditional_branch(cond_val, body_bb, end_bb)
            .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

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
        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        let init_bb = self.context.append_basic_block(parent_fn, "for.init");
        let cond_bb = self.context.append_basic_block(parent_fn, "for.cond");
        let body_bb = self.context.append_basic_block(parent_fn, "for.body");
        let inc_bb = self.context.append_basic_block(parent_fn, "for.inc");
        let end_bb = self.context.append_basic_block(parent_fn, "for.end");

        self.builder
            .build_unconditional_branch(init_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        self.builder.position_at_end(init_bb);

        let counter_type = self.context.i64_type();
        let counter = self
            .builder
            .build_alloca(counter_type, "for.counter")
            .map_err(|e| format!("Failed to allocate counter: {:?}", e))?;

        self.builder
            .build_store(counter, counter_type.const_zero())
            .map_err(|e| format!("Failed to store counter: {:?}", e))?;

        let (start_val, end_val) = self.extract_range_bounds(iter)?;

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

        self.builder
            .build_store(counter, start_val)
            .map_err(|e| format!("Failed to initialize counter: {:?}", e))?;

        self.builder
            .build_unconditional_branch(cond_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        self.builder.position_at_end(cond_bb);

        let current_counter = self
            .builder
            .build_load(counter_type, counter, "counter.val")
            .map_err(|e| format!("Failed to load counter: {:?}", e))?
            .into_int_value();

        let end_value = self
            .builder
            .build_load(counter_type, end_var, "end.val")
            .map_err(|e| format!("Failed to load end: {:?}", e))?
            .into_int_value();

        let cond = self
            .builder
            .build_int_compare(IntPredicate::SLT, current_counter, end_value, "for.cond")
            .map_err(|e| format!("Failed to build comparison: {:?}", e))?;

        self.builder
            .build_conditional_branch(cond, body_bb, end_bb)
            .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

        self.builder.position_at_end(body_bb);

        let prev_vars = self.variables.clone();
        let prev_types = self.variable_types.clone();

        let loop_var_type = counter_type.as_basic_type_enum();
        let loop_var = self
            .builder
            .build_alloca(loop_var_type, name)
            .map_err(|e| format!("Failed to allocate loop variable: {:?}", e))?;

        let counter_val = self
            .builder
            .build_load(counter_type, counter, "counter.load")
            .map_err(|e| format!("Failed to load counter: {:?}", e))?;

        self.builder
            .build_store(loop_var, counter_val)
            .map_err(|e| format!("Failed to store loop variable: {:?}", e))?;

        self.variables.insert(name.to_string(), loop_var);
        self.variable_types.insert(name.to_string(), loop_var_type);

        self.compile_block(body)?;

        self.variables = prev_vars;
        self.variable_types = prev_types;

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

        self.builder.position_at_end(inc_bb);

        let current = self
            .builder
            .build_load(counter_type, counter, "counter.inc.load")
            .map_err(|e| format!("Failed to load counter: {:?}", e))?
            .into_int_value();

        let incremented = self
            .builder
            .build_int_add(current, counter_type.const_int(1, false), "counter.inc")
            .map_err(|e| format!("Failed to increment counter: {:?}", e))?;

        self.builder
            .build_store(counter, incremented)
            .map_err(|e| format!("Failed to store incremented counter: {:?}", e))?;

        self.builder
            .build_unconditional_branch(cond_bb)
            .map_err(|e| format!("Failed to build branch: {:?}", e))?;

        self.builder.position_at_end(end_bb);

        Ok(())
    }

    fn extract_range_bounds(
        &mut self,
        iter: &Expr,
    ) -> Result<(BasicValueEnum<'ctx>, BasicValueEnum<'ctx>), String> {
        if let Expr::Binary(start_expr, op, end_expr) = iter {
            if op == &BinOp::Range {
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

        Err("For loop iterator must be a range (e.g., 0..10)".to_string())
    }

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
            Expr::Function(func) => self.compile_lambda(func),
            Expr::EnumVariant(enum_name, variant_name) => {
                self.compile_enum_variant(enum_name, variant_name)
            }
            Expr::Match { expr, arms } => self.compile_match(expr, arms),
            Expr::StructLiteral(struct_name, fields) => { 
                self.compile_struct_literal(struct_name, fields)
            }
        }
    }
    
    // ========== SIMPLE LAMBDA SUPPORT ==========
    
    fn compile_lambda(&mut self, _function: &Function) -> Result<BasicValueEnum<'ctx>, String> {
        // Lambdas should only appear in variable declarations
        // They are compiled and tracked there
        Err("Lambda expressions must be assigned to variables".to_string())
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
        if let Some(ptr) = self.variables.get(name) {
            let load_type = self.variable_types.get(name)
                .ok_or_else(|| format!("Unknown type for variable: {}", name))?;
            
            return self.builder
                .build_load(*load_type, *ptr, name)
                .map_err(|e| format!("Failed to load variable: {:?}", e));
        }
        
        Err(format!("Undefined variable: {}", name))
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
                    BinOp::Range => unreachable!(),
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

    fn compile_indirect_call(
        &mut self,
        var_name: &str,
        args: &[Expr],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // Check if this variable holds a lambda function
        if let Some(lambda_name) = self.lambda_functions.get(var_name) {
            // This is a lambda call - look up the function and call it directly
            let function = *self
                .functions
                .get(lambda_name)
                .ok_or_else(|| format!("Lambda function not found: {}", lambda_name))?;

            let mut compiled_args = Vec::new();
            for arg in args {
                let val = self.compile_expr(arg)?;
                compiled_args.push(val.into());
            }

            let call_site = self
                .builder
                .build_call(function, &compiled_args, "lambda_call")
                .map_err(|e| format!("Failed to build lambda call: {:?}", e))?;

            match call_site.try_as_basic_value() {
                ValueKind::Basic(value) => return Ok(value),
                ValueKind::Instruction(_) => return Err("Lambda returned void".to_string()),
            }
        }

        // Not a lambda - this would be an indirect call through a function pointer
        // which we don't support in the simplified version
        Err(format!("Variable {} is not a callable lambda", var_name))
    }

fn compile_call(
    &mut self,
    func_expr: &Expr,
    args: &[Expr],
) -> Result<BasicValueEnum<'ctx>, String> {
    // Check if this is an enum variant constructor call: EnumName::Variant(data)
    if let Expr::EnumVariant(enum_name, variant_name) = func_expr {
        let enum_info = self
            .enums
            .get(enum_name)
            .ok_or_else(|| format!("Unknown enum: {}", enum_name))?;
        
        let variant_info = enum_info
            .variants
            .iter()
            .find(|(name, _)| name == variant_name)
            .ok_or_else(|| format!("Unknown variant: {}", variant_name))?;
        
        let (_, expected_type) = variant_info;
        
        match expected_type {
            Some(_) => {
                // Variant expects data
                if args.len() != 1 {
                    return Err(format!(
                        "Enum variant {}::{} expects 1 argument, got {}",
                        enum_name, variant_name, args.len()
                    ));
                }
                let payload_value = self.compile_expr(&args[0])?;
                return self.create_enum_value(enum_name, variant_name, Some(payload_value));
            }
            None => {
                // Variant expects no data
                if !args.is_empty() {
                    return Err(format!(
                        "Enum variant {}::{} expects no arguments, got {}",
                        enum_name, variant_name, args.len()
                    ));
                }
                return self.create_enum_value(enum_name, variant_name, None);
            }
        }
    }
    
    let func_name = if let Expr::Identifier(name) = func_expr {
        name
    } else {
        return Err("Only direct function calls supported".to_string());
    };
    
    // Handle built-in functions (print, read)
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
                    .ok_or(format!("Failed to find corresponding variable"))?;
                let variable_type = self
                    .variable_types
                    .get(identifier_name)
                    .ok_or(format!("Failed to find value assigned to the variable"))?;
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
            .map_err(|e| format!("Failed to build read call: {:?}", e))?;

        match call_site.try_as_basic_value() {
            ValueKind::Basic(value) => return Ok(value),
            ValueKind::Instruction(_) => return Err("read should return a value".to_string()),
        }
    }

    // Regular function call
    let function = *self
        .functions
        .get(func_name)
        .ok_or_else(|| format!("Undefined function: {}", func_name))?;

    let mut compiled_args = Vec::new();
    for arg in args {
        let val = self.compile_expr(arg)?;
        compiled_args.push(val.into());
    }

    let call_site = self
        .builder
        .build_call(function, &compiled_args, "call")
        .map_err(|e| format!("Failed to build call: {:?}", e))?;

    match call_site.try_as_basic_value() {
        ValueKind::Basic(value) => Ok(value),
        ValueKind::Instruction(_) => Err("Function returned void".to_string()),
    }
}

fn compile_field(&mut self, obj: &Expr, field: &str) -> Result<BasicValueEnum<'ctx>, String> {
    let obj_val = self.compile_expr(obj)?;
    
    // Get the struct value
    let struct_val = match obj_val {
        BasicValueEnum::StructValue(sv) => sv,
        _ => return Err(format!("Field access requires a struct value, got {:?}", obj_val.get_type())),
    };
    
    let struct_type = struct_val.get_type();
    
    // Find which struct this is by matching the type
    let mut struct_name: Option<String> = None;
    for (name, ty) in &self.structs {
        if let BasicTypeEnum::StructType(st) = ty {
            if st == &struct_type {
                struct_name = Some(name.clone());
                break;
            }
        }
    }
    
    let struct_name = struct_name
        .ok_or_else(|| "Could not determine struct type".to_string())?;
    
    // Get the field names for this struct
    let field_names = self.struct_fields
        .get(&struct_name)
        .ok_or_else(|| format!("Struct {} has no field information", struct_name))?;
    
    // Find the index of the requested field
    let field_index = field_names
        .iter()
        .position(|f| f == field)
        .ok_or_else(|| format!("Struct {} has no field '{}'", struct_name, field))?;
    
    // Extract the field value using LLVM's extractvalue instruction
    self.builder
        .build_extract_value(struct_val, field_index as u32, field)
        .map_err(|e| format!("Failed to extract field: {:?}", e))
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
fn compile_struct_literal(
    &mut self,
    struct_name: &str,
    fields: &[(String, Expr)],
) -> Result<BasicValueEnum<'ctx>, String> {
    let struct_type_enum = self
        .structs
        .get(struct_name)
        .ok_or_else(|| format!("Unknown struct: {}", struct_name))?
        .clone();

    let struct_type = match struct_type_enum {
        BasicTypeEnum::StructType(st) => st,
        _ => return Err(format!("{} is not a struct", struct_name)),
    };

    let field_names = self
        .struct_fields
        .get(struct_name)
        .ok_or_else(|| format!("Struct {} has no field information", struct_name))?
        .clone();

    // Check that all fields are provided
    if fields.len() != field_names.len() {
        return Err(format!(
            "Struct {} expects {} fields, got {}",
            struct_name,
            field_names.len(),
            fields.len()
        ));
    }

    // Create a map of provided fields for easy lookup
    let mut provided_fields: HashMap<String, BasicValueEnum<'ctx>> = HashMap::new();
    for (field_name, field_expr) in fields {
        let field_value = self.compile_expr(field_expr)?;
        provided_fields.insert(field_name.clone(), field_value);
    }

    // Build struct in field order (important for LLVM struct layout)
    let mut struct_val = struct_type.get_undef();
    for (idx, expected_field_name) in field_names.iter().enumerate() {
        let field_value = provided_fields
            .get(expected_field_name)
            .ok_or_else(|| format!("Missing field '{}' in struct {}", expected_field_name, struct_name))?;

        struct_val = self
            .builder
            .build_insert_value(struct_val, *field_value, idx as u32, "with_field")
            .map_err(|e| format!("Failed to insert field: {:?}", e))?
            .into_struct_value();
    }

    Ok(struct_val.into())
}

    fn compile_match(
        &mut self,
        match_expr: &Expr,
        arms: &[MatchArm],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        if arms.is_empty() {
            return Err("Match expression must have at least one arm".to_string());
        }

        let match_value = self.compile_expr(match_expr)?;

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        let merge_bb = self.context.append_basic_block(parent_fn, "match.end");

        let first_arm_type = self.infer_expr_type(&arms[0].body)?;
        let result_alloca = self
            .builder
            .build_alloca(first_arm_type, "match.result")
            .map_err(|e| format!("Failed to allocate match result: {:?}", e))?;

        let saved_vars = self.variables.clone();
        let saved_types = self.variable_types.clone();

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;

            let arm_check_bb = self
                .context
                .append_basic_block(parent_fn, &format!("match.arm{}.check", i));
            let arm_body_bb = self
                .context
                .append_basic_block(parent_fn, &format!("match.arm{}.body", i));
            let next_bb = if is_last {
                merge_bb
            } else {
                self.context
                    .append_basic_block(parent_fn, &format!("match.arm{}.next", i + 1))
            };

            self.builder
                .build_unconditional_branch(arm_check_bb)
                .map_err(|e| format!("Failed to build branch: {:?}", e))?;

            self.builder.position_at_end(arm_check_bb);

            self.variables = saved_vars.clone();
            self.variable_types = saved_types.clone();

            let matches = self.compile_pattern_check(match_value, &arm.pattern)?;

            self.builder
                .build_conditional_branch(matches.into_int_value(), arm_body_bb, next_bb)
                .map_err(|e| format!("Failed to build conditional branch: {:?}", e))?;

            self.builder.position_at_end(arm_body_bb);

            let arm_result = self.compile_expr(&arm.body)?;

            self.builder
                .build_store(result_alloca, arm_result)
                .map_err(|e| format!("Failed to store match result: {:?}", e))?;

            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| format!("Failed to build branch: {:?}", e))?;

            if !is_last {
                self.builder.position_at_end(next_bb);
            }
        }

        self.variables = saved_vars;
        self.variable_types = saved_types;

        self.builder.position_at_end(merge_bb);

        self.builder
            .build_load(first_arm_type, result_alloca, "match.result.load")
            .map_err(|e| format!("Failed to load match result: {:?}", e))
    }

    fn compile_pattern_check(
        &mut self,
        value: BasicValueEnum<'ctx>,
        pattern: &Pattern,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        match pattern {
            Pattern::Wildcard => {
                Ok(self.context.bool_type().const_int(1, false).into())
            }
            Pattern::Literal(lit) => {
                self.compile_literal_pattern_check(value, lit)
            }
            Pattern::Identifier(name) => {
                let value_type = value.get_type();
                let alloca = self
                    .builder
                    .build_alloca(value_type, name)
                    .map_err(|e| format!("Failed to allocate pattern variable: {:?}", e))?;

                self.builder
                    .build_store(alloca, value)
                    .map_err(|e| format!("Failed to store pattern variable: {:?}", e))?;

                self.variables.insert(name.clone(), alloca);
                self.variable_types.insert(name.clone(), value_type);

                Ok(self.context.bool_type().const_int(1, false).into())
            }
            Pattern::EnumVariant(variant_path, inner_pattern) => {
                self.compile_enum_pattern_check(value, variant_path, inner_pattern.as_deref())
            }
            Pattern::Tuple(patterns) => {
                self.compile_tuple_pattern_check(value, patterns)
            }
        }
    }

    fn compile_literal_pattern_check(
        &mut self,
        value: BasicValueEnum<'ctx>,
        lit: &Literal,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let literal_value = self.compile_literal(lit)?;

        match (value, literal_value) {
            (BasicValueEnum::IntValue(v), BasicValueEnum::IntValue(l)) => {
                Ok(self
                    .builder
                    .build_int_compare(IntPredicate::EQ, v, l, "lit.cmp")
                    .map_err(|e| format!("Failed to compare integers: {:?}", e))?
                    .into())
            }
            (BasicValueEnum::FloatValue(v), BasicValueEnum::FloatValue(l)) => {
                Ok(self
                    .builder
                    .build_float_compare(FloatPredicate::OEQ, v, l, "lit.cmp")
                    .map_err(|e| format!("Failed to compare floats: {:?}", e))?
                    .into())
            }
            _ => Err("Type mismatch in literal pattern".to_string()),
        }
    }

    fn compile_enum_pattern_check(
        &mut self,
        value: BasicValueEnum<'ctx>,
        variant_path: &str,
        inner_pattern: Option<&Pattern>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let parts: Vec<&str> = variant_path.split("::").collect();
        if parts.len() != 2 {
            return Err(format!("Invalid enum variant path: {}", variant_path));
        }

        let enum_name = parts[0];
        let variant_name = parts[1];

        let (num_variants, _enum_type, variants) = {
            let enum_info = self
                .enums
                .get(enum_name)
                .ok_or_else(|| format!("Unknown enum: {}", enum_name))?;

            (
                enum_info.variants.len(),
                enum_info.enum_type,
                enum_info.variants.clone(),
            )
        };

        let variant_idx = variants
            .iter()
            .position(|(name, _)| name == variant_name)
            .ok_or_else(|| format!("Unknown variant: {}", variant_name))?;

        let discriminant = self.extract_enum_discriminant(value)?;

        let expected_discriminant: BasicValueEnum = if num_variants <= 256 {
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
                .i32_type()
                .const_int(variant_idx as u64, false)
                .into()
        };

        let discriminant_matches = self
            .builder
            .build_int_compare(
                IntPredicate::EQ,
                discriminant.into_int_value(),
                expected_discriminant.into_int_value(),
                "enum.disc.cmp",
            )
            .map_err(|e| format!("Failed to compare discriminants: {:?}", e))?;

        if let Some(inner_pat) = inner_pattern {
            let payload = self.extract_enum_payload(value)?;
            let inner_matches = self.compile_pattern_check(payload, inner_pat)?;

            Ok(self
                .builder
                .build_and(
                    discriminant_matches,
                    inner_matches.into_int_value(),
                    "enum.full.match",
                )
                .map_err(|e| format!("Failed to build and: {:?}", e))?
                .into())
        } else {
            Ok(discriminant_matches.into())
        }
    }

    fn compile_tuple_pattern_check(
        &mut self,
        value: BasicValueEnum<'ctx>,
        patterns: &[Pattern],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let struct_value = value.into_struct_value();

        let mut all_match = self.context.bool_type().const_int(1, false);

        for (i, pattern) in patterns.iter().enumerate() {
            let element = self
                .builder
                .build_extract_value(struct_value, i as u32, &format!("tuple.elem{}", i))
                .map_err(|e| format!("Failed to extract tuple element: {:?}", e))?;

            let element_matches = self.compile_pattern_check(element, pattern)?;

            all_match = self
                .builder
                .build_and(all_match, element_matches.into_int_value(), "tuple.and")
                .map_err(|e| format!("Failed to build and: {:?}", e))?;
        }

        Ok(all_match.into())
    }

    fn infer_expr_type(&self, expr: &Expr) -> Result<BasicTypeEnum<'ctx>, String> {
        match expr {
            Expr::Literal(lit) => match lit {
                Literal::Int(_) => Ok(self.context.i64_type().into()),
                Literal::Float(_) => Ok(self.context.f64_type().into()),
                Literal::Bool(_) => Ok(self.context.bool_type().into()),
                Literal::Char(_) => Ok(self.context.i8_type().into()),
                Literal::String(_) => Ok(self
                    .context
                    .i8_type()
                    .ptr_type(AddressSpace::default())
                    .into()),
            },
            Expr::Identifier(name) => self
                .variable_types
                .get(name)
                .cloned()
                .ok_or_else(|| format!("Unknown variable type: {}", name)),
            Expr::EnumVariant(enum_name, _) => {
                let enum_info = self
                    .enums
                    .get(enum_name)
                    .ok_or_else(|| format!("Unknown enum: {}", enum_name))?;
                Ok(enum_info.enum_type.into())
            }
            _ => {
                Ok(self.context.i64_type().into())
            }
        }
    }

    pub fn get_ir(&self) -> String {
        self.module.to_string()
    }

    pub fn get_module(&self) -> &Module<'ctx> {
        &self.module
    }
}
