use std::fmt;
use std::result;
use thiserror::Error;

pub type EtherResult<T> = result::Result<T, EtherError>;

#[derive(Debug, Clone, Error)]
#[error("{err_string} at {}:{}", line, column)]
pub struct ParserError {
    err_string: String,
    pub line: usize,
    pub column: usize,
}

impl ParserError {
    pub fn new(err_string: String, line: usize, column: usize) -> Self {
        Self { err_string, line, column }
    }
}

#[derive(Debug, Clone, Error)]
#[error("{err_string}")]
pub struct TokenizerError {
    err_string: String,
}

impl TokenizerError {
    pub fn new(err_string: String) -> Self {
        Self { err_string }
    }
}

#[derive(Debug, Error)]
pub enum EtherError {
    Parser(ParserError),
    Tokenizer(TokenizerError),
}

impl fmt::Display for EtherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EtherError::Tokenizer(e) => write!(f, "Error in tokenizing: {}", e),
            EtherError::Parser(e) => write!(f, "Error in parsing: {}", e),
        }
    }
}
