#![allow(dead_code)]

use crate::error::{EtherError, EtherResult, ParserError};
use crate::lexer::{Token, TokenType};

// ================= AST =================

#[derive(Debug, PartialEq, Eq)]
pub struct Program {
    pub imports: Vec<Import>,
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Import {
    pub module: String,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Declaration {
    Function(Function),
    Struct(StructDef),
    Enum(EnumDef),
    Var(VarDecl),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Type {
    Primitive(String),
    Array(Box<Type>),
    Function(FunctionHeader),
    Custom(String),
}

#[derive(Debug, PartialEq, Eq)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct EnumDef {
    pub name: String,
    pub fields: Vec<(String, Option<Type>)>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionHeader {
    pub name: Option<String>,
    pub params: Vec<(Option<String>, Type)>,
    pub return_type: Box<Type>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Function {
    pub header: FunctionHeader,
    pub body: Block,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct VarDecl {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expr,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Stmt {
    Var(VarDecl),
    Return(Option<Expr>),
    Expr(Expr),
    Block(Block),
    If {
        cond: Expr,
        then_block: Block,
        else_block: Option<Block>,
    },
    While {
        cond: Expr,
        body: Block,
    },
    For {
        name: String,
        iter: Expr,
        body: Block,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Expr {
    Literal(Literal),
    Identifier(String),
    Assign(Box<Expr>, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Field(Box<Expr>, String),
    Function(Function),
    Index(Box<Expr>, Box<Expr>),
    EnumVariant(String, String), // enum_name, variant_name
    Match {
        expr: Box<Expr>,
        arms: Vec<MatchArm>,
    },
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Pattern {
    Wildcard,                          // _
    Literal(Literal),                  // 42, true, "hello"
    Identifier(String),                // x (binding)
    EnumVariant(String, Option<Box<Pattern>>), // Option::Some(x) or Status::Ok
    Tuple(Vec<Pattern>),               // (x, y, z)
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Literal {
    Int(String),
    Float(String),
    Bool(bool),
    String(String),
    Char(char),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Range,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum UnOp {
    Neg,
    Not,
}

// ================= PARSER =================

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ========== Core Token Management ==========

    fn current_token(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn position(&self) -> (usize, usize) {
        self.current_token()
            .or_else(|| self.tokens.last())
            .map(|t| (t.line, t.column))
            .unwrap_or((1, 1))
    }

    fn peek(&self) -> EtherResult<TokenType> {
        self.current_token()
            .map(|t| t.token_type.clone())
            .ok_or_else(|| self.error("Unexpected end of input"))
    }

    fn advance(&mut self) -> EtherResult<TokenType> {
        let token = self.peek()?;
        self.pos += 1;
        Ok(token)
    }

    fn check(&self, expected: &TokenType) -> bool {
        self.peek().ok().as_ref() == Some(expected)
    }

    fn consume(&mut self, expected: TokenType) -> bool {
        if self.check(&expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    // ========== Error Handling ==========

    fn error(&self, message: impl Into<String>) -> EtherError {
        let (line, column) = self.position();
        ParserError {
            err_string: message.into(),
            line,
            column,
        }
        .into()
    }

    fn expect(&mut self, expected: TokenType) -> EtherResult<()> {
        let actual = self.advance()?;
        if actual == expected {
            Ok(())
        } else {
            Err(self.error(format!("Expected {:?}, got {:?}", expected, actual)))
        }
    }

    fn expect_identifier(&mut self) -> EtherResult<String> {
        match self.advance()? {
            TokenType::Identifier(name) => Ok(name),
            token => Err(self.error(format!("Expected identifier, got {:?}", token))),
        }
    }

    // ========== Lookahead Helpers ==========

    /// Check if current position starts a function definition: `(params): ReturnType { ... }`
    fn is_function_definition(&self) -> bool {
        if !self.check(&TokenType::LParen) {
            return false;
        }

        self.find_matching_paren()
            .and_then(|pos| self.tokens.get(pos))
            .map(|token| token.token_type == TokenType::Colon)
            .unwrap_or(false)
    }

    fn find_matching_paren(&self) -> Option<usize> {
        let mut pos = self.pos + 1;
        let mut depth = 1;

        while depth > 0 && pos < self.tokens.len() {
            match &self.tokens[pos].token_type {
                TokenType::LParen => depth += 1,
                TokenType::RParen => depth -= 1,
                _ => {}
            }
            pos += 1;
        }

        (depth == 0).then_some(pos)
    }

    // ========== Top-Level Parsing ==========

    pub fn parse_program(&mut self) -> EtherResult<Program> {
        let imports = self.parse_imports()?;
        let declarations = self.parse_declarations()?;

        Ok(Program {
            imports,
            declarations,
        })
    }

    fn parse_imports(&mut self) -> EtherResult<Vec<Import>> {
        let mut imports = Vec::new();
        while self.check(&TokenType::Import) {
            imports.push(self.parse_import()?);
        }
        Ok(imports)
    }

    fn parse_import(&mut self) -> EtherResult<Import> {
        self.expect(TokenType::Import)?;
        match self.advance()? {
            TokenType::StringLit(module) => Ok(Import { module }),
            token => Err(self.error(format!("Expected string literal, got {:?}", token))),
        }
    }

    fn parse_declarations(&mut self) -> EtherResult<Vec<Declaration>> {
        let mut declarations = Vec::new();
        while !self.check(&TokenType::Eof) {
            declarations.push(self.parse_declaration()?);
        }
        Ok(declarations)
    }

    fn parse_declaration(&mut self) -> EtherResult<Declaration> {
        match self.peek()? {
            TokenType::Fn => {
                self.advance()?;
                let name = self.expect_identifier().ok();
                Ok(Declaration::Function(self.parse_function(name)?))
            }
            TokenType::Struct => {
                self.advance()?;
                Ok(Declaration::Struct(self.parse_struct()?))
            }
            TokenType::Enum => {
                self.advance()?;
                Ok(Declaration::Enum(self.parse_enum()?))
            }
            TokenType::Let => {
                self.advance()?;
                Ok(Declaration::Var(self.parse_var_decl()?))
            }
            token => Err(self.error(format!("Invalid declaration start: {:?}", token))),
        }
    }

    // ========== Type Parsing ==========

    fn parse_type(&mut self) -> EtherResult<Type> {
        match self.advance()? {
            TokenType::Int => Ok(Type::Primitive("int".into())),
            TokenType::Float => Ok(Type::Primitive("float".into())),
            TokenType::Bool => Ok(Type::Primitive("bool".into())),
            TokenType::String => Ok(Type::Primitive("string".into())),
            TokenType::Char => Ok(Type::Primitive("char".into())),
            TokenType::Void => Ok(Type::Primitive("void".into())),
            TokenType::Identifier(name) => Ok(Type::Custom(name)),
            TokenType::LBracket => self.parse_array_type(),
            TokenType::LParen => self.parse_function_type(),
            token => Err(self.error(format!("Invalid type: {:?}", token))),
        }
    }

    fn parse_array_type(&mut self) -> EtherResult<Type> {
        let element_type = self.parse_type()?;
        self.expect(TokenType::RBracket)?;
        Ok(Type::Array(Box::new(element_type)))
    }

    fn parse_function_type(&mut self) -> EtherResult<Type> {
        let mut params = Vec::new();

        if !self.check(&TokenType::RParen) {
            loop {
                let ty = self.parse_type()?;
                params.push((None, ty));

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
        }

        self.expect(TokenType::RParen)?;
        self.expect(TokenType::Colon)?;
        let return_type = self.parse_type()?;

        Ok(Type::Function(FunctionHeader {
            name: None,
            params,
            return_type: Box::new(return_type),
        }))
    }

    // ========== Struct and Enum Parsing ==========

    fn parse_struct(&mut self) -> EtherResult<StructDef> {
        let name = self.expect_identifier()?;
        self.expect(TokenType::LBrace)?;

        let mut fields = Vec::new();
        if !self.check(&TokenType::RBrace) {
            loop {
                let field_name = self.expect_identifier()?;
                self.expect(TokenType::Colon)?;
                let field_type = self.parse_type()?;
                fields.push((field_name, field_type));

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
        }

        self.expect(TokenType::RBrace)?;
        Ok(StructDef { name, fields })
    }

    fn parse_enum(&mut self) -> EtherResult<EnumDef> {
        let name = self.expect_identifier()?;
        self.expect(TokenType::LBrace)?;

        let mut fields = Vec::new();
        if !self.check(&TokenType::RBrace) {
            loop {
                let variant_name = self.expect_identifier()?;
                let variant_type = if self.consume(TokenType::Colon) {
                    self.parse_type().ok()
                } else {
                    None
                };

                fields.push((variant_name, variant_type));

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
        }

        self.expect(TokenType::RBrace)?;
        Ok(EnumDef { name, fields })
    }

    // ========== Function Parsing ==========

    fn parse_function(&mut self, name: Option<String>) -> EtherResult<Function> {
        self.expect(TokenType::LParen)?;

        let mut params = Vec::new();
        if !self.check(&TokenType::RParen) {
            loop {
                let param_name = self.expect_identifier()?;
                self.expect(TokenType::Colon)?;
                let param_type = self.parse_type()?;
                params.push((Some(param_name), param_type));

                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
        }

        self.expect(TokenType::RParen)?;
        self.expect(TokenType::Colon)?;
        let return_type = self.parse_type()?;
        let body = self.parse_block()?;

        Ok(Function {
            header: FunctionHeader {
                name,
                params,
                return_type: Box::new(return_type),
            },
            body,
        })
    }

    // ========== Statement Parsing ==========

    pub fn parse_block(&mut self) -> EtherResult<Block> {
        self.expect(TokenType::LBrace)?;
        let mut statements = Vec::new();

        while !self.check(&TokenType::RBrace) {
            statements.push(self.parse_stmt()?);
        }

        self.expect(TokenType::RBrace)?;
        Ok(Block { statements })
    }

    pub fn parse_stmt(&mut self) -> EtherResult<Stmt> {
        match self.peek()? {
            TokenType::Let => self.parse_let_stmt(),
            TokenType::Return => self.parse_return_stmt(),
            TokenType::If => self.parse_if_stmt(),
            TokenType::While => self.parse_while_stmt(),
            TokenType::For => self.parse_for_stmt(),
            TokenType::LBrace => Ok(Stmt::Block(self.parse_block()?)),
            _ => self.parse_expr_stmt(),
        }
    }

    fn parse_let_stmt(&mut self) -> EtherResult<Stmt> {
        self.advance()?;
        Ok(Stmt::Var(self.parse_var_decl()?))
    }

    fn parse_return_stmt(&mut self) -> EtherResult<Stmt> {
        self.advance()?;
        let expr = if self.check(&TokenType::Semicolon) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        self.expect(TokenType::Semicolon)?;
        Ok(Stmt::Return(expr))
    }

    fn parse_expr_stmt(&mut self) -> EtherResult<Stmt> {
        let expr = self.parse_expr()?;
        self.expect(TokenType::Semicolon)?;
        Ok(Stmt::Expr(expr))
    }

    fn parse_var_decl(&mut self) -> EtherResult<VarDecl> {
        let name = self.expect_identifier()?;
        let ty = if self.consume(TokenType::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenType::Assign)?;
        let value = self.parse_expr()?;
        self.expect(TokenType::Semicolon)?;

        Ok(VarDecl { name, ty, value })
    }

    fn parse_if_stmt(&mut self) -> EtherResult<Stmt> {
        self.expect(TokenType::If)?;
        self.expect(TokenType::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        let then_block = self.parse_block()?;

        let else_block = if self.consume(TokenType::Else) {
            Some(self.parse_block()?)
        } else {
            None
        };

        Ok(Stmt::If {
            cond,
            then_block,
            else_block,
        })
    }

    fn parse_while_stmt(&mut self) -> EtherResult<Stmt> {
        self.expect(TokenType::While)?;
        self.expect(TokenType::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for_stmt(&mut self) -> EtherResult<Stmt> {
        self.expect(TokenType::For)?;
        self.expect(TokenType::LParen)?;
        let name = self.expect_identifier()?;
        self.expect(TokenType::In)?;
        let iter = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::For { name, iter, body })
    }

    // ========== Expression Parsing (Precedence Climbing) ==========

    pub fn parse_expr(&mut self) -> EtherResult<Expr> {
        match self.peek()? {
            TokenType::Match => self.parse_match_expr(),
            _ if self.is_function_definition() => Ok(Expr::Function(self.parse_function(None)?)),
            _ => self.parse_assignment(),
        }
    }

    fn parse_assignment(&mut self) -> EtherResult<Expr> {
        let left = self.parse_or()?;

        if self.consume(TokenType::Assign) {
            let right = self.parse_assignment()?;
            Ok(Expr::Assign(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_or(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_and()?;

        while self.consume(TokenType::Or) {
            let right = self.parse_and()?;
            expr = Expr::Binary(Box::new(expr), BinOp::Or, Box::new(right));
        }

        Ok(expr)
    }

    fn parse_and(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_equality()?;

        while self.consume(TokenType::And) {
            let right = self.parse_equality()?;
            expr = Expr::Binary(Box::new(expr), BinOp::And, Box::new(right));
        }

        Ok(expr)
    }

    fn parse_equality(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_range()?;

        while let Some(op) = self.try_consume_comparison_op() {
            let right = self.parse_range()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }

        Ok(expr)
    }

    fn try_consume_comparison_op(&mut self) -> Option<BinOp> {
        let op = match self.peek().ok()? {
            TokenType::Eq => BinOp::Eq,
            TokenType::Ne => BinOp::Ne,
            TokenType::Lt => BinOp::Lt,
            TokenType::Gt => BinOp::Gt,
            TokenType::Le => BinOp::Le,
            TokenType::Ge => BinOp::Ge,
            _ => return None,
        };

        self.pos += 1;
        Some(op)
    }

    fn parse_range(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_additive()?;

        while self.consume(TokenType::Range) {
            let right = self.parse_additive()?;
            expr = Expr::Binary(Box::new(expr), BinOp::Range, Box::new(right));
        }

        Ok(expr)
    }

    fn parse_additive(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_multiplicative()?;

        while let Some(op) = self.try_consume_additive_op() {
            let right = self.parse_multiplicative()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }

        Ok(expr)
    }

    fn try_consume_additive_op(&mut self) -> Option<BinOp> {
        let op = match self.peek().ok()? {
            TokenType::Plus => BinOp::Add,
            TokenType::Minus => BinOp::Sub,
            _ => return None,
        };

        self.pos += 1;
        Some(op)
    }

    fn parse_multiplicative(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_unary()?;

        while let Some(op) = self.try_consume_multiplicative_op() {
            let right = self.parse_unary()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(right));
        }

        Ok(expr)
    }

    fn try_consume_multiplicative_op(&mut self) -> Option<BinOp> {
        let op = match self.peek().ok()? {
            TokenType::Multiply => BinOp::Mul,
            TokenType::Divide => BinOp::Div,
            _ => return None,
        };

        self.pos += 1;
        Some(op)
    }

    fn parse_unary(&mut self) -> EtherResult<Expr> {
        match self.peek()? {
            TokenType::Minus => {
                self.advance()?;
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)))
            }
            TokenType::Not => {
                self.advance()?;
                Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            expr = match self.peek()? {
                TokenType::DoubleColon => self.parse_enum_variant(expr)?,
                TokenType::LParen => self.parse_call(expr)?,
                TokenType::Dot => self.parse_field_access(expr)?,
                TokenType::LBracket => self.parse_index(expr)?,
                _ => break,
            };
        }

        Ok(expr)
    }

    fn parse_enum_variant(&mut self, expr: Expr) -> EtherResult<Expr> {
        let Expr::Identifier(enum_name) = expr else {
            return Err(self.error(":: operator can only be used after an identifier"));
        };

        self.advance()?;
        let variant_name = self.expect_identifier()?;
        Ok(Expr::EnumVariant(enum_name, variant_name))
    }

    fn parse_call(&mut self, callee: Expr) -> EtherResult<Expr> {
        self.advance()?;
        let mut args = Vec::new();

        if !self.check(&TokenType::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if !self.consume(TokenType::Comma) {
                    break;
                }
            }
        }

        self.expect(TokenType::RParen)?;
        Ok(Expr::Call(Box::new(callee), args))
    }

    fn parse_field_access(&mut self, object: Expr) -> EtherResult<Expr> {
        self.advance()?;
        let field = self.expect_identifier()?;
        Ok(Expr::Field(Box::new(object), field))
    }

    fn parse_index(&mut self, array: Expr) -> EtherResult<Expr> {
        self.advance()?;
        let index = self.parse_expr()?;
        self.expect(TokenType::RBracket)?;
        Ok(Expr::Index(Box::new(array), Box::new(index)))
    }

    fn parse_primary(&mut self) -> EtherResult<Expr> {
        match self.advance()? {
            TokenType::Number(n) => Ok(Expr::Literal(Literal::Int(n.to_string()))),
            TokenType::FloatLit(f) => Ok(Expr::Literal(Literal::Float(f.to_string()))),
            TokenType::StringLit(s) => Ok(Expr::Literal(Literal::String(s.to_string()))),
            TokenType::CharLit(c) => Ok(Expr::Literal(Literal::Char(c))),
            TokenType::True => Ok(Expr::Literal(Literal::Bool(true))),
            TokenType::False => Ok(Expr::Literal(Literal::Bool(false))),
            TokenType::Identifier(name) => Ok(Expr::Identifier(name.clone())),
            TokenType::LParen => self.parse_grouped_expr(),
            token => Err(self.error(format!("Invalid expression start: {:?}", token))),
        }
    }

    fn parse_grouped_expr(&mut self) -> EtherResult<Expr> {
        let expr = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        Ok(expr)
    }

    // ========== Match Expression Parsing ==========

    fn parse_match_expr(&mut self) -> EtherResult<Expr> {
        self.expect(TokenType::Match)?;
        
        // Parse the matched expression
        let expr = Box::new(self.parse_assignment()?);
        
        self.expect(TokenType::LBrace)?;
        
        let mut arms = Vec::new();
        
        // Parse match arms
        while !self.check(&TokenType::RBrace) {
            arms.push(self.parse_match_arm()?);
            
            // Optional trailing comma
            self.consume(TokenType::Comma);
        }
        
        self.expect(TokenType::RBrace)?;
        
        Ok(Expr::Match { expr, arms })
    }

    fn parse_match_arm(&mut self) -> EtherResult<MatchArm> {
        let pattern = self.parse_pattern()?;
        
        self.expect(TokenType::FatArrow)?;
        
        let body = self.parse_assignment()?;
        
        Ok(MatchArm { pattern, body })
    }

    fn parse_pattern(&mut self) -> EtherResult<Pattern> {
        match self.peek()? {
            TokenType::Underscore => {
                self.advance()?;
                Ok(Pattern::Wildcard)
            }
            TokenType::Number(n) => {
                self.advance()?;
                Ok(Pattern::Literal(Literal::Int(n)))
            }
            TokenType::FloatLit(f) => {
                self.advance()?;
                Ok(Pattern::Literal(Literal::Float(f)))
            }
            TokenType::StringLit(s) => {
                self.advance()?;
                Ok(Pattern::Literal(Literal::String(s)))
            }
            TokenType::CharLit(c) => {
                self.advance()?;
                Ok(Pattern::Literal(Literal::Char(c)))
            }
            TokenType::True => {
                self.advance()?;
                Ok(Pattern::Literal(Literal::Bool(true)))
            }
            TokenType::False => {
                self.advance()?;
                Ok(Pattern::Literal(Literal::Bool(false)))
            }
            TokenType::Identifier(name) => {
                let id = name.clone();
                self.advance()?;
                
                // Check for enum variant pattern
                if self.consume(TokenType::DoubleColon) {
                    let variant = self.expect_identifier()?;
                    
                    // Check for variant with data: EnumName::Variant(pattern)
                    let inner_pattern = if self.consume(TokenType::LParen) {
                        let pat = self.parse_pattern()?;
                        self.expect(TokenType::RParen)?;
                        Some(Box::new(pat))
                    } else {
                        None
                    };
                    
                    Ok(Pattern::EnumVariant(
                        format!("{}::{}", id, variant),
                        inner_pattern,
                    ))
                } else {
                    // Just a binding identifier
                    Ok(Pattern::Identifier(id))
                }
            }
            TokenType::LParen => {
                self.advance()?;
                let mut patterns = Vec::new();
                
                if !self.check(&TokenType::RParen) {
                    loop {
                        patterns.push(self.parse_pattern()?);
                        if !self.consume(TokenType::Comma) {
                            break;
                        }
                    }
                }
                
                self.expect(TokenType::RParen)?;
                Ok(Pattern::Tuple(patterns))
            }
            token => Err(self.error(format!("Invalid pattern: {:?}", token))),
        }
    }
}
