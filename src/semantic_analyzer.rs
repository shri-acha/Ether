use crate::parser::Program;
use crate::type_checker::{TypeChecker, Substitution, InferredType};
use crate::error::{EtherError, TypeErrorDetail};

pub fn type_check_program(program: &Program) -> Result<(), EtherError> {
    let mut checker = TypeChecker::new();
    let subst = Substitution::new();

    // Pass 1: Register all function headers (and other globals) to allow forward references
    for decl in &program.declarations {
        if let crate::parser::Declaration::Function(func) = decl {
            checker.convert_function_header(&func.header)
                .map_err(|e| TypeErrorDetail{err_string: e, line: 0, column: 0})?;
        }
        // Structs/Enums would also be registered here in a full implementation
    }

    // Pass 2: Type check function bodies and global variable initializers
    for decl in &program.declarations {
        match decl {
            crate::parser::Declaration::Var(var) => {
                checker.check_stmt(&crate::parser::Stmt::Var(var.clone()), &subst, &InferredType::Void)
                    .map_err(|e| TypeErrorDetail{err_string: e.to_string(), line: 0, column: 0})?;
            }
            crate::parser::Declaration::Function(func) => {
                // Retrieve the already registered function type
                let func_name = func.header.name.as_ref().ok_or(
                    TypeErrorDetail{err_string: "Anonymous function at top level".to_string(), line: 0, column: 0}
                )?;
                
                // We clone to avoid borrowing issues while mutating env
                let func_ty = checker.env.lookup(func_name).unwrap().clone();
                
                if let InferredType::Function(param_types, ret_ty) = func_ty {
                    // Create a scope for the function parameters
                    checker.env.push_scope();

                    // Insert parameters into the local scope
                    for (i, (name_opt, _)) in func.header.params.iter().enumerate() {
                        if let Some(name) = name_opt {
                            if let Some(ty) = param_types.get(i) {
                                checker.env.insert(name.clone(), ty.clone());
                            }
                        }
                    }

                    // Check the function body
                    checker.check_block(&func.body, &subst, &ret_ty)
                        .map_err(|e| TypeErrorDetail{err_string: e, line: 0, column: 0})?;

                    // Clean up the parameter scope
                    checker.env.pop_scope();
                }
            }
            _ => {} 
        }
    }
    Ok(())
}