use crate::parser::Program;
use crate::type_checker::{TypeChecker, Substitution, InferredType};
use crate::error::{EtherError, TypeErrorDetail};

pub fn type_check_program(program: &Program) -> Result<(), EtherError> {
    let mut checker = TypeChecker::new();
    let subst = Substitution::new();

    for decl in &program.declarations {
        match decl {
            crate::parser::Declaration::Function(func) => {
                let func_ty = checker.convert_function_header(&func.header)
                    .map_err(|e| TypeErrorDetail{err_string: e, line: 0, column: 0})?;
                
                if let InferredType::Function(_, ret_ty) = func_ty {
                    checker.check_block(&func.body, &subst, &ret_ty)
                        .map_err(|e| TypeErrorDetail{err_string: e, line: 0, column: 0})?;
                }
            }
            _ => {} // Handle other declarations
        }
    }
    Ok(())
}
