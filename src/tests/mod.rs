#[cfg(test)]
use super::*;
#[test]
fn lexer_token_generation(){
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

println!("Generated {} tokens:\n", tokens.len());
for token in &tokens {
    println!("{}", token);
}

// Test edge cases
println!("\n--- Testing Edge Cases ---");

let test_cases = vec![
    ("'\n'", "Escaped newline char"),
    ("'\\t'", "Escaped tab char"),
    ("\"hello\\nworld\"", "String with newline"),
    ("3.14", "Float literal"),
    ("123", "Integer literal"),
];

for (input, description) in test_cases {
    println!("\nTest: {} (input: {})", description, input);
    let mut test_tokenizer = Tokenizer::new(input);
    let test_tokens = test_tokenizer.tokenize(true);
    for token in &test_tokens {
        if token.token_type != TokenType::Eof {
            println!("  {}", token);
        }
    }
}
}
