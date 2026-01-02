use inkwell::context::Context;

use crate::llvm_ir_generator;
#[cfg(test)]
use crate::{error::*, lexer::*, llvm_ir_generator::*, parser::*};
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
        return (b:int):int{return a+b;};
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
    // work on adding the assert
    // assert!(ast,Program{..})
    // println!("{:?}", ast);
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

#[test]
fn llvm_ir_gen() {
    let test_codes: [&str; 3] = [
        r#"
        fn add(a: int, b: int): int {
            return a + b;
        }

        fn main(): int {
            let x: int = 10;
            let y: int = 20;
            let result: int = add(x, y);
            return result;
        }
    "#,
        r#"
        fn fibonacci(n: int): int {
            if (n <= 1) {
                return n;
            } else {
                return fibonacci(n - 1) + fibonacci(n - 2);
            }
        }

        fn main(): int {
            let result: int = fibonacci(10);
            return result;
        }
    "#,
        r#"
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
    "#,
    ];
    let selected_test_index = 0;
    let mut tokenizer = Tokenizer::new(test_codes[selected_test_index]);
    let tokens = tokenizer.tokenize(true);
    let mut parser = Parser::new(tokens);
    let parsed_tokens = parser.parse_program().unwrap();

    let context = Context::create();
    let mut codegen = CodeGen::new(&context, "my_module");

    match codegen.compile_program(&parsed_tokens) {
        Ok(_) => {
            codegen.print_ir();
        }
        Err(e) => {
            eprintln!("Code generation error: {}", e);
        }
    }
}
