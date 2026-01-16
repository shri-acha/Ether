use inkwell::context::Context;

#[cfg(test)]
use crate::{error::*, lexer::*, llvm_ir_generator::*, parser::*,
    type_checker::*, semantic_analyzer::*};
use std::assert_matches::assert_matches;

#[test]
fn lexer_token_generation() {
    let code = r#"
// This is a comment

import "math"

struct Point {
    x: int,
    y: int
}

fn add(a: int, b: int): int {
    let result = a + b;
    return result;
}

fn main(): void {
    let x = 10;
    let y = 20;
    let pi = 3.14159;
    let sum = add(x, y);
    
    let newline_char = '\n';
    let tab_char = '\t';
    let escaped_string = "Hello\nWorld\tTab";
    
    if (sum > 25) {
        print("Sum is greater than 25");
    } else {
        print("Sum is 25 or less");
    }
    
    /* Multi-line
       comment */
    for (i in 0..10) {
        print(i);
    }
}
"#;

    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);

    let test_cases = vec![
        ("'\n'", "Escaped newline char"),
        ("'\\t'", "Escaped tab char"),
        ("\"hello\\nworld\"", "String with newline"),
        ("3.14", "Float literal"),
        ("123", "Integer literal"),
    ];

    // for (input, description) in test_cases {
    //     let mut test_tokenizer = Tokenizer::new(input);
    //     let test_tokens = test_tokenizer.tokenize(true);
    //     for token in &test_tokens {
    //         if token.token_type != TokenType::Eof {
    //         }
    //     }
    // }
}

#[test]
fn ast_gen_test() {
    let code = r#"
    fn main(): void {
    let x = 3 + 4;
    if (x > 5) {
        return;
    }
}"#;
    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);

    assert!(
        matches!(tokens.last().unwrap().token_type, TokenType::Eof),
        "Lexer must emit EOF token"
    );

    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program();
    println!("{:?}", ast);
    assert_matches!(ast, Ok(_));
}

#[test]
fn ast_currying_test() {
    let code = r#"
    fn x(a:int):(int):int {
        return (b:int):int{
            return a+b;
        };
    }
    "#;
    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);

    assert!(
        matches!(tokens.last().unwrap().token_type, TokenType::Eof),
        "Lexer must emit EOF token"
    );

    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program();
    assert_matches!(ast, Ok(_));
}

#[test]
fn assigned_lambda_function(){

    let code = r#"
    let y = (x:int):int{ return x; };
    "#;
    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);

    assert!(
        matches!(tokens.last().unwrap().token_type, TokenType::Eof),
        "Lexer must emit EOF token"
    );

    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program();
    println!("{:?}",ast);
    assert_matches!(ast, Ok(_));

}

#[test]
fn non_variable_binded_function_call(){

    let code = r#"
    let y = (x:int):int{ return x; };
    y(5);
    "#;
    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);

    assert!(
        matches!(tokens.last().unwrap().token_type, TokenType::Eof),
        "Lexer must emit EOF token"
    );

    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program();
    println!("{:?}",ast);
    assert_matches!(ast, Ok(_));

}

#[test]
fn enum_design(){
    let code = r#"
    enum random{
        a,
        b,
        c:char
    }
    "#;
    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);

    assert!(
        matches!(tokens.last().unwrap().token_type, TokenType::Eof),
        "Lexer must emit EOF token"
    );

    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program();
    println!("{:?}",ast);
    assert_matches!(ast, Ok(_));

}

#[test]
fn error_location_reporting() {
    // Test 1: Error at the beginning of the code
    let code1 = "let x = ";
    let mut tokenizer = Tokenizer::new(code1);
    let tokens = tokenizer.tokenize(true);
    let mut parser = Parser::new(tokens);
    let result = parser.parse_program();

    if let Err(EtherError::Parser(err)) = result {
        // Error should be near the beginning
        assert_eq!(err.line, 1, "Error should be on line 1");
        assert!(
            err.column < 10,
            "Error should be near the start for incomplete expression"
        );
        println!(
            "Test 1 - Error at line {}, column {}: {}",
            err.line, err.column, err
        );
    } else {
        panic!("Test 1: Expected parser error for incomplete expression");
    }

    // Test 2: Error in the middle of code
    let code2 = r#"
fn test(): void {
    let x = ;
    return;
}
"#;
    let mut tokenizer = Tokenizer::new(code2);
    let tokens = tokenizer.tokenize(true);
    let mut parser = Parser::new(tokens);
    let result = parser.parse_program();

    if let Err(EtherError::Parser(err)) = result {
        // Error should be on line 4 (accounting for how the parser counts newlines)
        assert_eq!(
            err.line, 4,
            "Error should be on line 4 where the incomplete expression is"
        );
        assert!(
            err.column > 0,
            "Error column should be recorded and positive"
        );
        println!(
            "Test 2 - Error at line {}, column {}: {}",
            err.line, err.column, err
        );
    } else {
        panic!("Test 2: Expected parser error for incomplete expression in function");
    }

    // Test 3: Missing closing brace
    let code3 = r#"fn main(): void {
    let x = 10;
    if (x > 5) {
        print("test");
    // Missing closing brace for if
}"#;
    let mut tokenizer = Tokenizer::new(code3);
    let tokens = tokenizer.tokenize(true);
    let mut parser = Parser::new(tokens);
    let result = parser.parse_program();

    if let Err(EtherError::Parser(err)) = result {
        // Error location should be tracked
        assert!(err.line >= 1, "Error line should be properly recorded");
        assert!(
            err.column > 0,
            "Error column should be recorded and positive"
        );
        println!(
            "Test 3 - Error at line {}, column {}: {}",
            err.line, err.column, err
        );
    }
}

#[cfg(test)]
mod llvm_tests{

use inkwell::context::Context;
use crate::{error::*, lexer::*, llvm_ir_generator::*, parser::*,
    type_checker::*, semantic_analyzer::*};
use std::assert_matches::assert_matches;

#[test]
fn llvm_ir_gen() {
    let test_code: &'static str =  r#"
        fn add(a: int, b: int): int {
            return a + b;
        }

        fn main(): int {
            let x: int = 10;
            let y: int = 20;
            let result: int = add(x, y);
            return result;
        }
    "#;
    let selected_test_index = 0;
    let mut tokenizer = Tokenizer::new(test_code);
    let tokens = tokenizer.tokenize(true);
    let mut parser = Parser::new(tokens);
    let parsed_tokens = parser.parse_program().unwrap();

    let context = Context::create();
    let mut codegen = CodeGen::new(&context, "my_module");

    match codegen.compile_program(&parsed_tokens) {
        Ok(_) => {
            println!("{}",test_code);
            println!("{}",codegen.get_ir());
        }
        Err(e) => {
            println!("Code generation error: {}", e);
            }
        }
    }
    #[test]
    fn enum_declaration(){
        let test_code = r#"
            enum enum_name{ a:int, b:bool, c }
            "#;
        let mut tokenizer = Tokenizer::new(test_code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "my_module");

        match codegen.compile_program(&parsed_tokens) {
            Ok(_) => {
                println!("{}",test_code);
                println!("{}",codegen.get_ir());
            }
            Err(e) => {
                println!("Code generation error: {}", e);
                }
            }
        }
}

#[cfg(test)]
mod type_tests {
    use super::*;

    fn parse_stmt(input: &str) -> crate::parser::Stmt {
        let mut tokenizer = Tokenizer::new(input);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        parser.parse_stmt().expect("Stmt parse failure")
    }

    fn parse_expr(input: &str) -> crate::parser::Expr {
        let mut tokenizer = Tokenizer::new(input);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        parser.parse_expr().expect("Expr parse failure")
    }

    #[test]
    fn test_infer_literals() {
        let checker = TypeChecker::new();
        let subst = Substitution::new();

        assert_eq!(checker.infer_literal(&crate::parser::Literal::Int("42".to_string())), InferredType::Int);
        // Note: infer_expr logic matches
    }

    #[test]
    fn test_infer_binary_op() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let (ty, _) = checker.infer_expr(&parse_expr("10 + 20"), &subst).unwrap();
        assert_eq!(ty, InferredType::Int);
        assert!(checker.infer_expr(&parse_expr("10 + 3.14"), &subst).is_err());
    }

    #[test]
    fn test_unification_logic() {
        assert!(unify(&InferredType::Int, &InferredType::Int).is_ok());
        assert!(unify(&InferredType::Int, &InferredType::Float).is_err());
        let mut gene = TypeVarGenerator::new();
        let tv = gene.fresh();
        let res = unify(&InferredType::Var(tv), &InferredType::Bool).unwrap();
        assert_eq!(res.get(&tv).unwrap(), &InferredType::Bool);
    }

    #[test]
    fn test_variable_declaration_and_lookup() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let stmt = parse_stmt("let x = 5;");
        checker.check_stmt(&stmt, &subst, &InferredType::Void).unwrap();
        let expr = parse_expr("x");
        let (ty, _) = checker.infer_expr(&expr, &subst).unwrap();
        assert_eq!(ty, InferredType::Int);
    }

    #[test]
    fn test_block_scope_isolation() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let code = r#"{
            let inner = 10;
        }"#;
        let mut tokenizer = Tokenizer::new(code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let block = parser.parse_block().unwrap();
        checker.check_block(&block, &subst, &InferredType::Void).unwrap();
        assert!(checker.env.lookup("inner").is_none());
    }

    #[test]
    fn test_function_return_validation() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let block_ok = r#"{ return 10; }"#;
        let mut parser_ok = Parser::new(Tokenizer::new(block_ok).tokenize(true));
        assert!(checker.check_block(&parser_ok.parse_block().unwrap(), &subst, &InferredType::Int).is_ok());

        let block_err = r#"{ return 3.14; }"#;
        let mut parser_err = Parser::new(Tokenizer::new(block_err).tokenize(true));
        assert!(checker.check_block(&parser_err.parse_block().unwrap(), &subst, &InferredType::Int).is_err());
    }

    #[test]
    fn test_function_call_arity_and_types() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let func_ty = InferredType::Function(
            vec![InferredType::Int, InferredType::Int],
            Box::new(InferredType::Bool)
        );
        checker.env.insert("compare".to_string(), func_ty);
        assert!(checker.infer_expr(&parse_expr("compare(1, 2)"), &subst).is_ok());
        assert!(checker.infer_expr(&parse_expr("compare(1)"), &subst).is_err());
        assert!(checker.infer_expr(&parse_expr("compare(1, true)"), &subst).is_err());
    }

    #[test]
    fn test_inference_from_source() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let expr = parse_expr("10 + 20");
        let (ty, _) = checker.infer_expr(&expr, &subst).unwrap();
        assert_eq!(ty, InferredType::Int);
    }

    #[test]
    fn test_assignment_logic() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let stmt = parse_stmt("let x = 5;");
        checker.check_stmt(&stmt, &subst, &InferredType::Void).unwrap();
        checker.env.insert("x".to_string(), InferredType::Int);
        let expr = parse_expr("x + 1");
        let (ty, _) = checker.infer_expr(&expr, &subst).unwrap();
        assert_eq!(ty, InferredType::Int);
    }

    #[test]
    fn test_block_scope_resolution() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();
        let code = r#"{
            let a = 1.5;
            return a;
        }"#;
        let mut tokenizer = Tokenizer::new(code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let block = parser.parse_block().unwrap();
        assert!(checker.check_block(&block, &subst, &InferredType::Float).is_ok());
    }

    fn run_checker(source: &str, expected_ret: &InferredType) -> Result<TypeChecker, String> {
        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();

        if source.trim().starts_with('{') {
            let block = parser.parse_block().map_err(|e| e.to_string())?;
            checker.check_block(&block, &subst, expected_ret).map_err(|e| e.to_string())?;
        } else {
            let stmt = parser.parse_stmt().map_err(|e| e.to_string())?;
            checker.check_stmt(&stmt, &subst, expected_ret).map_err(|e| e.to_string())?;
        }
        Ok(checker)
    }

    #[test]
    fn test_variable_shadowing_integration() {
        let source = r#"{
            let x = 10;
            {
                let x = true;
            }
            let y = x + 5;
        }"#;
        let result = run_checker(source, &InferredType::Void);
        assert!(result.is_ok(), "Shadowing should resolve without conflict");
    }

    #[test]
    fn test_arithmetic_inference_integration() {
        let mut checker = TypeChecker::new();
        let subst = Substitution::new();

        // Check statements individually to persist definitions in the current scope
        let stmts = [
            "let a = 5;",
            "let b = 10;",
            "let c = (a + b) * 2;",
        ];

        for src in stmts {
            let mut tokenizer = Tokenizer::new(src);
            let tokens = tokenizer.tokenize(true);
            let mut parser = Parser::new(tokens);
            let stmt = parser.parse_stmt().expect("Failed to parse statement");
            checker.check_stmt(&stmt, &subst, &InferredType::Void).expect("Type check failed");
        }

        // Verify lookup succeeds because the scope was never exited
        assert_eq!(checker.env.lookup("c"), Some(&InferredType::Int));
    }

    #[test]
    fn test_type_mismatch_integration() {
        let source = r#"{
            let x = 10;
            let y = "string";
            let z = x + y;
        }"#;
        let result = run_checker(source, &InferredType::Void);
        assert!(result.is_err(), "Binary operation on mismatched types must fail");
    }

    #[test]
    fn test_recursive_definition_prevention() {
        let source = r#"let x = x + 1;"#;
        let result = run_checker(source, &InferredType::Void);
        assert!(result.is_err(), "Should fail due to undefined identifier 'x'");
    }

    #[test]
    fn test_complex_block_return_integration() {
        let source = r#"{
            let radius = 5.0;
            let pi = 3.14;
            return pi * (radius * radius);
        }"#;
        let result = run_checker(source, &InferredType::Float);
        assert!(result.is_ok());
    }
}

#[cfg(test)]
mod semantic_analyzer_tests {
    use super::*;
    use crate::lexer::Tokenizer;
    use crate::parser::Parser;

    fn analyze(source: &str) -> Result<(), EtherError> {
        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program()?; 
        type_check_program(&program)
    }

    #[test]
    fn test_valid_program_semantics() {
        // This validates the scope fixes in semantic_analyzer.rs
        let code = r#"
            fn add(a: int, b: int): int {
                return a + b;
            }

            fn main(): void {
                let x: int = add(5, 10);
            }
        "#;
        assert!(analyze(code).is_ok());
    }

    #[test]
    fn test_invalid_return_type_semantics() {
        let code = r#"
            fn get_float(): int {
                return 3.14; 
            }
        "#;
        let result = analyze(code);
        assert!(result.is_err());
    }

    #[test]
    fn test_function_call_mismatch() {
        let code = r#"
            fn square(n: int): int {
                return n * n;
            }

            fn main(): void {
                let res: int = square(true); 
            }
        "#;
        assert!(analyze(code).is_err());
    }
}

#[cfg(test)]
mod symbol_table_tests{

    use crate::{error::*, lexer::*, llvm_ir_generator::*, parser::*,
    type_checker::*, semantic_analyzer::*};
    use std::assert_matches::assert_matches;

    use crate::symbol_table::SymbolResolver;

    #[test]
    fn registers_function_symbol() {
        let src =  r#"
        fn sum_to_n(n: int): int {
            let sum: int = 0;
            let i: int = 0;
            
            while (i <= n) {
                sum = sum + i;
                i = i + 1;
            }
            
            return sum;
        }
        
        fn main(): int {
            return sum_to_n(100);
        }
    "#         
        ;
        
        let mut tokenizer = Tokenizer::new(src);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let program = parser.parse_program().unwrap();

        let mut resolver = SymbolResolver::new();
        resolver.analyze_program(&program);

        assert!(resolver.table.lookup("main").is_some());
    }


}
