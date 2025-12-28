use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    // Keywords
    Fn,
    Let,
    If,
    Else,
    While,
    For,
    Return,
    Import,
    In,
    True,
    False,
    Struct,
    
    // Type keywords
    Int,
    Float,
    Bool,
    String,
    Char,
    Void,
    
    // Literals
    Number32(i32),
    FloatLit32(f32),  // Added for floating-point numbers
    StringLit(String),
    CharLit(char),
    Identifier(String),
    
    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Assign,
    
    // Comparison operators
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    
    // Logical operators
    And,
    Or,
    Not,
    
    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Semicolon,
    Comma,
    Colon,
    Dot,
    Arrow,
    
    // Special
    Eof,
    Comment(String),
}

#[derive(Debug, Clone)]
pub struct Token {
    pub token_type: TokenType,
    pub line: usize,
    pub column: usize,
}

impl Token {
    pub fn new(token_type: TokenType, line: usize, column: usize) -> Self {
        Token {
            token_type,
            line,
            column,
        }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Token({:?} at {}:{})", self.token_type, self.line, self.column)
    }
}

pub struct Tokenizer {
    source: Vec<char>,
    pos: usize,
    line: usize,
    column: usize,
}

impl Tokenizer {
    pub fn new(source: &str) -> Self {
        Tokenizer {
            source: source.chars().collect(),
            pos: 0,
            line: 1,
            column: 1,
        }
    }
    
    fn current_char(&self) -> Option<char> {
        if self.pos >= self.source.len() {
            None
        } else {
            Some(self.source[self.pos])
        }
    }
    
    fn peek_char(&self, offset: usize) -> Option<char> {
        let pos = self.pos + offset;
        if pos >= self.source.len() {
            None
        } else {
            Some(self.source[pos])
        }
    }
    
    fn advance(&mut self) -> Option<char> {
        if self.pos >= self.source.len() {
            return None;
        }
        let ch = self.source[self.pos];
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(ch)
    }
    
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn read_single_line_comment(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.column;
        let mut value = String::new();
        
        self.advance(); // /
        self.advance(); // /
        
        while let Some(ch) = self.current_char() {
            if ch == '\n' {
                break;
            }
            value.push(ch);
            self.advance();
        }
        
        if self.current_char() == Some('\n') {
            self.advance();
        }
        
        Token::new(TokenType::Comment(value), start_line, start_col)
    }
    
    fn read_multi_line_comment(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.column;
        let mut value = String::new();
        
        self.advance(); // /
        self.advance(); // *
        
        while let Some(ch) = self.current_char() {
            if ch == '*' && self.peek_char(1) == Some('/') {
                self.advance(); // *
                self.advance(); // /
                break;
            }
            value.push(ch);
            self.advance();
        }
        
        Token::new(TokenType::Comment(value), start_line, start_col)
    }
    
    fn read_number(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.column;
        let mut value = String::new();
        let mut is_float = false;
        
        // Read integer part
        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() {
                value.push(ch);
                self.advance();
            } else if ch == '.' && !is_float && self.peek_char(1).map_or(false, |c| c.is_ascii_digit()) {
                // Only treat as float if there's a digit after the dot
                is_float = true;
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        let token_type = if is_float {
            TokenType::FloatLit(value)
        } else {
            TokenType::Number(value)
        };
        
        Token::new(token_type, start_line, start_col)
    }
    
    fn read_identifier(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.column;
        let mut value = String::new();
        
        while let Some(ch) = self.current_char() {
            if ch.is_alphanumeric() || ch == '_' {
                value.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        
        // Check if it's a keyword
        let token_type = match value.as_str() {
            "fn" => TokenType::Fn,
            "let" => TokenType::Let,
            "if" => TokenType::If,
            "else" => TokenType::Else,
            "while" => TokenType::While,
            "for" => TokenType::For,
            "return" => TokenType::Return,
            "import" => TokenType::Import,
            "in" => TokenType::In,
            "true" => TokenType::True,
            "false" => TokenType::False,
            "struct" => TokenType::Struct,
            "int" => TokenType::Int,
            "float" => TokenType::Float,
            "bool" => TokenType::Bool,
            "string" => TokenType::String,
            "char" => TokenType::Char,
            "void" => TokenType::Void,
            _ => TokenType::Identifier(value),
        };
        
        Token::new(token_type, start_line, start_col)
    }
    
    fn unescape_string(s: &str) -> String {
        let mut result = String::new();
        let mut chars = s.chars();
        
        while let Some(ch) = chars.next() {
            if ch == '\\' {
                match chars.next() {
                    Some('n') => result.push('\n'),
                    Some('t') => result.push('\t'),
                    Some('r') => result.push('\r'),
                    Some('\\') => result.push('\\'),
                    Some('"') => result.push('"'),
                    Some('0') => result.push('\0'),
                    Some(c) => {
                        result.push('\\');
                        result.push(c);
                    }
                    None => result.push('\\'),
                }
            } else {
                result.push(ch);
            }
        }
        
        result
    }
    
    fn read_string(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.column;
        let mut value = String::new();
        
        self.advance(); // opening "
        
        while let Some(ch) = self.current_char() {
            if ch == '"' {
                break;
            }
            if ch == '\\' {
                value.push(ch);
                self.advance();
                if let Some(next_ch) = self.current_char() {
                    value.push(next_ch);
                    self.advance();
                }
            } else {
                value.push(ch);
                self.advance();
            }
        }
        
        if self.current_char() == Some('"') {
            self.advance(); // closing "
        }
        
        // Unescape the string
        let unescaped = Self::unescape_string(&value);
        Token::new(TokenType::StringLit(unescaped), start_line, start_col)
    }
    
    fn read_char(&mut self) -> Token {
        let start_line = self.line;
        let start_col = self.column;
        
        self.advance(); // opening '
        
        let ch = if let Some('\\') = self.current_char() {
            self.advance(); // consume \
            match self.current_char() {
                Some('n') => { self.advance(); '\n' }
                Some('t') => { self.advance(); '\t' }
                Some('r') => { self.advance(); '\r' }
                Some('\'') => { self.advance(); '\'' }
                Some('\\') => { self.advance(); '\\' }
                Some('"') => { self.advance(); '"' }
                Some('0') => { self.advance(); '\0' }
                Some(c) => {
                    // Invalid escape: treat as literal character
                    let result = c;
                    self.advance();
                    result
                }
                None => '\0'
            }
        } else {
            let result = self.current_char().unwrap_or('\0');
            self.advance();
            result
        };
        
        if self.current_char() == Some('\'') {
            self.advance(); // closing '
        }
        
        Token::new(TokenType::CharLit(ch), start_line, start_col)
    }
    
    pub fn tokenize(&mut self, skip_comments: bool) -> Vec<Token> {
        let mut tokens = Vec::new();
        
        while self.pos < self.source.len() {
            self.skip_whitespace();
            
            if self.current_char().is_none() {
                break;
            }
            
            let ch = self.current_char().unwrap();
            let start_line = self.line;
            let start_col = self.column;
            
            // Comments
            if ch == '/' && self.peek_char(1) == Some('/') {
                let token = self.read_single_line_comment();
                if !skip_comments {
                    tokens.push(token);
                }
                continue;
            }
            
            if ch == '/' && self.peek_char(1) == Some('*') {
                let token = self.read_multi_line_comment();
                if !skip_comments {
                    tokens.push(token);
                }
                continue;
            }
            
            // Numbers (including floats)
            if ch.is_ascii_digit() {
                tokens.push(self.read_number());
                continue;
            }
            
            // Identifiers and keywords
            if ch.is_alphabetic() || ch == '_' {
                tokens.push(self.read_identifier());
                continue;
            }
            
            // String literals
            if ch == '"' {
                tokens.push(self.read_string());
                continue;
            }
            
            // Char literals
            if ch == '\'' {
                tokens.push(self.read_char());
                continue;
            }
            
            // Two-character operators
            if ch == '=' && self.peek_char(1) == Some('=') {
                tokens.push(Token::new(TokenType::Eq, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            if ch == '!' && self.peek_char(1) == Some('=') {
                tokens.push(Token::new(TokenType::Ne, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            if ch == '<' && self.peek_char(1) == Some('=') {
                tokens.push(Token::new(TokenType::Le, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            if ch == '>' && self.peek_char(1) == Some('=') {
                tokens.push(Token::new(TokenType::Ge, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            if ch == '&' && self.peek_char(1) == Some('&') {
                tokens.push(Token::new(TokenType::And, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            if ch == '|' && self.peek_char(1) == Some('|') {
                tokens.push(Token::new(TokenType::Or, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            if ch == '-' && self.peek_char(1) == Some('>') {
                tokens.push(Token::new(TokenType::Arrow, start_line, start_col));
                self.advance();
                self.advance();
                continue;
            }
            
            // Single-character tokens
            let token_type = match ch {
                '+' => Some(TokenType::Plus),
                '-' => Some(TokenType::Minus),
                '*' => Some(TokenType::Multiply),
                '/' => Some(TokenType::Divide),
                '=' => Some(TokenType::Assign),
                '<' => Some(TokenType::Lt),
                '>' => Some(TokenType::Gt),
                '!' => Some(TokenType::Not),
                '(' => Some(TokenType::LParen),
                ')' => Some(TokenType::RParen),
                '{' => Some(TokenType::LBrace),
                '}' => Some(TokenType::RBrace),
                '[' => Some(TokenType::LBracket),
                ']' => Some(TokenType::RBracket),
                ';' => Some(TokenType::Semicolon),
                ',' => Some(TokenType::Comma),
                ':' => Some(TokenType::Colon),
                '.' => Some(TokenType::Dot),
                _ => None,
            };
            
            if let Some(tt) = token_type {
                tokens.push(Token::new(tt, start_line, start_col));
                self.advance();
                continue;
            }
            
            // Unknown character - skip with warning (could be logged in production)
            eprintln!("Warning: Skipping unknown character '{}' at {}:{}", ch, start_line, start_col);
            self.advance();
        }
        
        // Add EOF token
        tokens.push(Token::new(TokenType::Eof, self.line, self.column));
        tokens
    }
}
