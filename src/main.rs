#![feature(assert_matches)]
mod codegen;
mod error;
mod lexer;
mod parser;
mod semantic_analyzer;
mod symbol_table;
mod type_checker;
mod eth_clap;
use clap::Parser;
use eth_clap::EthArgs;
use crate::eth_clap::handle_eth_args;
#[cfg(test)]
mod tests;


fn main() {
    let args = EthArgs::parse();
    handle_eth_args(&args);
}
