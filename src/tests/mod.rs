#[cfg(test)]
use crate::{error::*, lexer::*, parser::*};
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
