#![feature(assert_matches)]
mod codegen;
mod error;
mod eth_clap;
mod lexer;
mod parser;
mod semantic_analyzer;
mod symbol_table;
mod type_checker;
use crate::eth_clap::handle_eth_args;
use clap::Parser;
use eth_clap::EthArgs;
#[cfg(test)]
mod tests;

fn main() {
    let args = EthArgs::parse();
    handle_eth_args(&args);
}
