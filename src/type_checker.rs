use std::collections::HashMap;
use crate::parser::{Expr, BinOp, UnOp, Literal};//, Type};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferredType {
    Int,
    Float,
    Bool,
    Char,
    String,
    Void,
    Array(Box<InferredType>),
    Function(Vec<InferredType>, Box<InferredType>),
    Custom(String),
    Var(TypeVar),  // Unresolved type variable to be inferred
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TypeVar(usize);

pub struct TypeVarGenerator {
    counter: usize,
}

impl TypeVarGenerator {
    pub fn new() -> Self {
        Self { counter: 0 }
    }
    // Generate a fresh type variable
    pub fn fresh(&mut self) -> TypeVar {
        let var = TypeVar(self.counter);
        self.counter += 1;
        var
    }
}

// A mapping from type variables to their inferred types.
pub type Substitution = HashMap<TypeVar, InferredType>;
// Apply a substitution to an inferred type.
pub fn apply_subst(ty: &InferredType, subst: &Substitution) -> InferredType {
    match ty {
      //check for type variable and replace if found in substitution
        InferredType::Var(tv) => {
            if let Some(replacement) = subst.get(tv) {
                apply_subst(replacement, subst)  // Recursive to handle chains
            } else {
                ty.clone()
            }
        }
        // Recursively apply substitution to composite types
        InferredType::Array(inner) => {
            InferredType::Array(Box::new(apply_subst(inner, subst)))
        }
        // Recursively apply substitution to function parameter and return types
        InferredType::Function(params, ret) => {
            InferredType::Function(
                params.iter().map(|p| apply_subst(p, subst)).collect(),
                Box::new(apply_subst(ret, subst)),
            )
        }
        _ => ty.clone(),
    }
}
// Compose two substitutions into one.
pub fn compose_subst(s1: &Substitution, s2: &Substitution) -> Substitution {
    let mut result = s1.clone();
    for (tv, ty) in s2 {
        result.insert(*tv, apply_subst(ty, s1));
    }
    result
}

// Unify two inferred types, returning a substitution if successful.
pub fn unify(t1: &InferredType, t2: &InferredType) -> Result<Substitution, String> {
    match (t1, t2) {
        (InferredType::Int, InferredType::Int) => Ok(HashMap::new()),
        (InferredType::Float, InferredType::Float) => Ok(HashMap::new()),
        (InferredType::Bool, InferredType::Bool) => Ok(HashMap::new()),
        (InferredType::Char, InferredType::Char) => Ok(HashMap::new()),
        (InferredType::String, InferredType::String) => Ok(HashMap::new()),
        (InferredType::Void, InferredType::Void) => Ok(HashMap::new()),
        
        (InferredType::Var(tv), t) | (t, InferredType::Var(tv)) => {
            if occurs_check(*tv, t) {
                Err(format!("Infinite type: {:?} = {:?}", tv, t))
            } else {
                let mut subst = HashMap::new();
                subst.insert(*tv, t.clone());
                Ok(subst)
            }
        }
        // Unify array types
        (InferredType::Array(a1), InferredType::Array(a2)) => {
            unify(a1, a2)
        }
        // Unify function types
        (InferredType::Function(p1, r1), InferredType::Function(p2, r2)) => {
            if p1.len() != p2.len() {
                return Err(format!(
                    "Function arity mismatch: {} vs {}",
                    p1.len(),
                    p2.len()
                ));
            }
            
            let mut subst = HashMap::new();
            for (param1, param2) in p1.iter().zip(p2.iter()) {
                let s = unify(
                    &apply_subst(param1, &subst),
                    &apply_subst(param2, &subst),
                )?;
                subst = compose_subst(&subst, &s);
            }
            
            let ret_subst = unify(
                &apply_subst(r1, &subst),
                &apply_subst(r2, &subst),
            )?;
            Ok(compose_subst(&subst, &ret_subst))
        }
        
        (InferredType::Custom(n1), InferredType::Custom(n2)) if n1 == n2 => {
            Ok(HashMap::new())
        }
        
        _ => Err(format!("Cannot unify {:?} with {:?}", t1, t2)),
    }
}
// Check if a type variable occurs within a type (to prevent infinite types).
fn occurs_check(tv: TypeVar, ty: &InferredType) -> bool {
    match ty {
        InferredType::Var(v) => *v == tv,
        InferredType::Array(inner) => occurs_check(tv, inner),
        InferredType::Function(params, ret) => {
            params.iter().any(|p| occurs_check(tv, p)) || occurs_check(tv, ret)
        }
        _ => false,
    }
}

// Type environment mapping variable names to their inferred types.
pub struct TypeEnv {
    vars: HashMap<String, InferredType>,
    scopes: Vec<HashMap<String, InferredType>>,
}
// Methods for managing the type environment
impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            scopes: Vec::new(),
        }
    }
    
    pub fn lookup(&self, name: &str) -> Option<&InferredType> {
        // Check scopes from innermost to outermost
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        self.vars.get(name)
    }
    
    pub fn insert(&mut self, name: String, ty: InferredType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name, ty);
        } else {
            self.vars.insert(name, ty);
        }
    }
    
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }
}

// Expression AST nodes (simplified for this example)
pub struct TypeChecker {
    pub env: TypeEnv,
    pub var_gen: TypeVarGenerator,
}
// Methods for type checking and inference
impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::new(),
            var_gen: TypeVarGenerator::new(),
        }
    }
    
    pub fn infer_expr(
        &mut self,
        expr: &Expr,
        subst: &Substitution,
    ) -> Result<(InferredType, Substitution), String> {
        match expr {
            Expr::Literal(lit) => Ok((self.infer_literal(lit), subst.clone())),
            
            Expr::Identifier(name) => {
                let ty = self.env.lookup(name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))?;
                Ok((apply_subst(ty, subst), subst.clone()))
            }
            
            Expr::Binary(left, op, right) => {
                let (left_ty, s1) = self.infer_expr(left, subst)?;
                let (right_ty, s2) = self.infer_expr(right, &s1)?;
                
                let left_ty = apply_subst(&left_ty, &s2);
                let right_ty = apply_subst(&right_ty, &s2);
                
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        // Arithmetic: both operands must be same numeric type
                        let s3 = unify(&left_ty, &right_ty)?;
                        let unified = apply_subst(&left_ty, &s3);
                        
                        // Result type same as operands
                        Ok((unified, compose_subst(&s2, &s3)))
                    }
                    
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        // Comparison: operands must match, result is bool
                        let s3 = unify(&left_ty, &right_ty)?;
                        Ok((InferredType::Bool, compose_subst(&s2, &s3)))
                    }
                    
                    BinOp::And | BinOp::Or => {
                        // Logical: both must be bool
                        let s3 = unify(&left_ty, &InferredType::Bool)?;
                        let s4 = unify(&right_ty, &InferredType::Bool)?;
                        Ok((InferredType::Bool, compose_subst(&compose_subst(&s2, &s3), &s4)))
                    }
                }
            }
            
            Expr::Unary(op, expr) => {
                let (ty, s) = self.infer_expr(expr, subst)?;
                match op {
                    UnOp::Neg => {
                        // Negation works on int or float
                        Ok((ty, s))
                    }
                    UnOp::Not => {
                        let s2 = unify(&ty, &InferredType::Bool)?;
                        Ok((InferredType::Bool, compose_subst(&s, &s2)))
                    }
                }
            }
            
            Expr::Call(func, args) => {
                let (func_ty, s1) = self.infer_expr(func, subst)?;
                
                let mut arg_types = Vec::new();
                let mut current_subst = s1;
                
                for arg in args {
                    let (arg_ty, s) = self.infer_expr(arg, &current_subst)?;
                    arg_types.push(apply_subst(&arg_ty, &s));
                    current_subst = s;
                }
                
                let ret_var = InferredType::Var(self.var_gen.fresh());
                let expected_func_ty = InferredType::Function(arg_types, Box::new(ret_var.clone()));
                
                let s2 = unify(&apply_subst(&func_ty, &current_subst), &expected_func_ty)?;
                let final_subst = compose_subst(&current_subst, &s2);
                
                Ok((apply_subst(&ret_var, &final_subst), final_subst))
            }
            
            Expr::Assign(target, value) => {
                let (target_ty, s1) = self.infer_expr(target, subst)?;
                let (value_ty, s2) = self.infer_expr(value, &s1)?;
                
                let s3 = unify(&apply_subst(&target_ty, &s2), &value_ty)?;
                Ok((apply_subst(&value_ty, &s3), compose_subst(&s2, &s3)))
            }
            
            _ => Err("Unsupported expression for type inference".to_string()),
        }
    }
    
    fn infer_literal(&self, lit: &Literal) -> InferredType {
        match lit {
            Literal::Int(_) => InferredType::Int,
            Literal::Float(_) => InferredType::Float,
            Literal::Bool(_) => InferredType::Bool,
            Literal::Char(_) => InferredType::Char,
            Literal::String(_) => InferredType::String,
        }
    }

    /// Converts AST type definitions into internal InferredType representations
    pub fn convert_ast_type(&self, ast_type: &crate::parser::Type) -> Result<InferredType, String> {
        match ast_type {
            crate::parser::Type::Primitive(s) => match s.as_str() {
                "int" => Ok(InferredType::Int),
                "float" => Ok(InferredType::Float),
                "bool" => Ok(InferredType::Bool),
                "char" => Ok(InferredType::Char),
                "string" => Ok(InferredType::String),
                "void" => Ok(InferredType::Void),
                _ => Ok(InferredType::Custom(s.clone())),
            },
            crate::parser::Type::Array(inner) => {
                let inner_ty = self.convert_ast_type(inner)?;
                Ok(InferredType::Array(Box::new(inner_ty)))
            }
            crate::parser::Type::Custom(s) => Ok(InferredType::Custom(s.clone())),
            crate::parser::Type::Function(header) => {
                let mut params = Vec::new();
                for (_, ty) in &header.params {
                    params.push(self.convert_ast_type(ty)?);
                }
                let ret = self.convert_ast_type(&header.return_type)?;
                Ok(InferredType::Function(params, Box::new(ret)))
            }
        }
    }

    // Registers function signature in the environment and returns its type
    pub fn convert_function_header(&mut self, header: &crate::parser::FunctionHeader) -> Result<InferredType, String> {
        let mut param_types = Vec::new();
        for (_name, ty) in &header.params { // Use .params and handle tuple (Option<String>, Type)
            param_types.push(self.convert_ast_type(ty)?);
        }
        let ret_type = self.convert_ast_type(&header.return_type)?;
        let func_ty = InferredType::Function(param_types, Box::new(ret_type));
        
        if let Some(name) = &header.name {
            self.env.insert(name.clone(), func_ty.clone());
        }
        Ok(func_ty)
    }

    // Recursively checks a block of statements within a new scope
    pub fn check_block(
        &mut self, 
        block: &crate::parser::Block, 
        subst: &Substitution, 
        ret_ty: &InferredType
    ) -> Result<Substitution, String> {
        self.env.push_scope();
        let mut current_subst = subst.clone();
        for stmt in &block.statements {
            current_subst = self.check_stmt(stmt, &current_subst, ret_ty)?;
        }
        self.env.pop_scope();
        Ok(current_subst)
    }

    /// Validates individual statements and updates the type substitution map
    pub fn check_stmt(
        &mut self,
        stmt: &crate::parser::Stmt,
        subst: &Substitution,
        ret_ty: &InferredType,
    ) -> Result<Substitution, String> {
        match stmt {
            crate::parser::Stmt::Var(var_decl) => {
                let expected_ty = match &var_decl.ty {
                    Some(t) => self.convert_ast_type(t)?,
                    None => InferredType::Var(self.var_gen.fresh()),
                };
                let mut current_subst = subst.clone();
                
                let (val_ty, s1) = self.infer_expr(&var_decl.value, &current_subst)?;
                let s2 = unify(&expected_ty, &val_ty)?;
                current_subst = compose_subst(&s1, &s2);
                let resolved_ty = apply_subst(&expected_ty, &current_subst);
                self.env.insert(var_decl.name.clone(), resolved_ty);
                
                self.env.insert(var_decl.name.clone(), expected_ty);
                Ok(current_subst)
            }
            crate::parser::Stmt::Return(expr_opt) => {
                if let Some(expr) = expr_opt {
                    let (val_ty, s1) = self.infer_expr(expr, subst)?;
                    let s2 = unify(ret_ty, &val_ty)?;
                    Ok(compose_subst(&s1, &s2))
                } else {
                    unify(ret_ty, &InferredType::Void)
                }
            }
            crate::parser::Stmt::Expr(expr) => {
                let (_, s) = self.infer_expr(expr, subst)?;
                Ok(s)
            }
            crate::parser::Stmt::Block(block) => self.check_block(block, subst, ret_ty),
            
            // FIXED: Use struct pattern matching syntax with { .. }
            crate::parser::Stmt::If { .. } | crate::parser::Stmt::While { .. } | crate::parser::Stmt::For { .. } => Ok(subst.clone()),
        }
    }
}