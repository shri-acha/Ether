use std::result;
use thiserror::Error;

pub type EtherResult<T> = result::Result<T, EtherError>;

#[derive(Debug, Clone, Error)]
#[error("{err_string} at {line}:{column}")]
pub struct ParserError {
    pub err_string: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Error)]
#[error("{err_string} at {line}:{column}")]
pub struct TokenizerError {
    pub err_string: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Error)]
#[error("{err_string} at {line}:{column}")]
pub struct TypeErrorDetail {
    pub err_string: String,
    pub line: usize,
    pub column: usize,
}

// removed: manually display error of 'fmt::Display impl' for EtherError
// since 'thiserror' crate automatically implements it based on the #[error(...)] attribute
#[derive(Debug, Error)]
pub enum EtherError {
    #[error("Error in parsing: {0}")]
    Parser(#[from] ParserError),

    #[error("Error in tokenizing: {0}")]
    Tokenizer(#[from] TokenizerError),

    #[error("Type error: {0}")]
    TypeInfer(#[from] TypeErrorDetail),
}
