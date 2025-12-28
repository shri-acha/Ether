mod lexer;
mod parser;

use lexer::{Tokenizer,TokenType};
use parser::{Parser};
fn main() {
    let code = r#"
    // This is a comment ->(sagar) the code previously here want in compliance with the grammar either we change the grammar or proceed acc to the production rules
    //no dots(from the import file thing) , no structs or edge cases here for parser test. 

    fn main(): void {
    let x = 3 + 4;
    if (x > 5) {
        return;
    }
}
    
    "#;
    
    let mut tokenizer = Tokenizer::new(code);
    let tokens = tokenizer.tokenize(true);
    
    //printing tokens visual 
    println!("Generated {} tokens:\n", tokens.len());
    for token in &tokens {
        println!("{}", token);
    }
    
    // Test edge cases //sagar commented this out because edge cases are edgy
    // println!("\n--- Testing Edge Cases ---");
    
    // let test_cases = vec![
    //     ("'\n'", "Escaped newline char"),
    //     ("'\\t'", "Escaped tab char"),
    //     ("\"hello\\nworld\"", "String with newline"),
    //     ("3.14", "Float literal"),
    //     ("123", "Integer literal"),
    // ];
    
    // for (input, description) in test_cases {
    //     println!("\nTest: {} (input: {})", description, input);
    //     let mut test_tokenizer = Tokenizer::new(input);
    //     let test_tokens = test_tokenizer.tokenize(true);
    //     for token in &test_tokens {
    //         if token.token_type != TokenType::Eof {
    //             println!("  {}", token);
    //         }
    //     }
    // }

    //me(sagar) put this to check paraser hai 
    assert!(
        matches!(tokens.last().unwrap().token_type, TokenType::Eof),
        "Lexer must emit EOF token"
    );

    let mut parser = Parser::new(tokens);
    let ast = parser.parse_program();

    //ast tree dumpy
    println!("\n\nAst ::\n");
    println!("{:#?}",ast);

}
