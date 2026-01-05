#![feature(assert_matches)]

mod error;
mod lexer;
mod llvm_ir_generator;
mod parser;
mod tests;

mod semantic_analyzer;
mod type_checker;

use lexer::{TokenType, Tokenizer};
use parser::Parser;
fn main() {}
