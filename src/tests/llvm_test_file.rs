#[cfg(test)]
mod llvm_tests {

    use crate::{codegen::*, error::*, lexer::*, parser::*, semantic_analyzer::*, type_checker::*};
    use inkwell::context::Context;
    use std::assert_matches::assert_matches;
    use std::fs;

    #[test]
    fn llvm_ir_gen() {
        let test_code: &'static str = r#"
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
        let mut tokenizer = Tokenizer::new(test_code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "my_module");

        match codegen.compile_program(&parsed_tokens) {
            Ok(_) => {
                println!("{}", test_code);
                println!("{}", codegen.get_ir());
                let ir = codegen.get_ir();
                fs::write("./src/tests/llvm-ir-files/func-test-out.ll", ir)
                    .expect("Failed to write IR");
            }
            Err(e) => {
                println!("Code generation error: {}", e);
            }
        }
    }

    #[test]
    fn llvm_ir_while_loop() {
        let test_code: &'static str = r#"

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
            return sum_to_n(10);
        }
    "#;
        let mut tokenizer = Tokenizer::new(test_code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();
        // println!("{:?}", parsed_tokens);

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "my_module");

        match codegen.compile_program(&parsed_tokens) {
            Ok(_) => {
                println!("{}", test_code);
                println!("{}", codegen.get_ir());
                let ir = codegen.get_ir();
                fs::write("./src/tests/llvm-ir-files/while-loop-test-out.ll", ir)
                    .expect("Failed to write IR");
            }
            Err(e) => {
                println!("Code generation error: {}", e);
            }
        }
    }

    #[test]
    fn llvm_ir_if_else() {
        let test_code: &'static str = r#"
        fn fibonacci(n: int): int {
            if (n <= 1) {
                return n;
            } else {
                return fibonacci(n - 1) + fibonacci(n - 2);
            }
        }
        fn main(): int {
            let result: int = fibonacci(7);
            return result;
        }
    "#;
        let mut tokenizer = Tokenizer::new(test_code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "my_module");

        match codegen.compile_program(&parsed_tokens) {
            Ok(_) => {
                println!("{}", test_code);
                println!("{}", codegen.get_ir());
                let ir = codegen.get_ir();
                fs::write("./src/tests/llvm-ir-files/if-else-test-out.ll", ir)
                    .expect("Failed to write IR");
            }
            Err(e) => {
                println!("Code generation error: {}", e);
            }
        }
    }

    #[test]
    fn enum_declaration() {
        let test_code = r#"
            enum enum_name{ a:int, b:bool, c }
            fn main(): void {
                let x:enum_name = 5;
            }
            "#;
        let mut tokenizer = Tokenizer::new(test_code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "my_module");

        match codegen.compile_program(&parsed_tokens) {
            Ok(_) => {
                println!("{}", test_code);
                println!("{}", codegen.get_ir());
            }
            Err(e) => {
                println!("Code generation error: {}", e);
            }
        }
    }

    #[test]
    fn llvm_ir_for_loop_with_range() {
        let test_code: &'static str = r#"
        fn main(): int {
            let end_value = 6;
            let result:int = 0;
            for (i in 2..end_value) {
                result = result + i;
            }
            return result;
        }
    "#;
        let mut tokenizer = Tokenizer::new(test_code);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, "my_module");

        match codegen.compile_program(&parsed_tokens) {
            Ok(_) => {
                println!("{}", test_code);
                println!("{}", codegen.get_ir());
                let ir = codegen.get_ir();
                fs::write(
                    "./src/tests/llvm-ir-files/for-loop-with-range-test-out.ll",
                    ir,
                )
                .expect("Failed to write IR");
            }
            Err(e) => {
                println!("Code generation error: {}", e);
            }
        }
    }
}
