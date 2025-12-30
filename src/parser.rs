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
    Var(VarDecl),
}

#[derive(Debug, PartialEq, Eq)]
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
pub struct FunctionHeader {
    pub name: Option<String>,
    pub params: Vec<(Option<String>, Type)>,
    pub return_type: Box<Type>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Function {
    pub header: FunctionHeader,
    pub body: Block,
}

#[derive(Debug, PartialEq, Eq)]
pub struct VarDecl {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expr,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

#[derive(Debug, PartialEq, Eq)]
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

#[derive(Debug, PartialEq, Eq)]
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
}

#[derive(Debug, PartialEq, Eq)]
pub enum Literal {
    Int(String),
    Float(String),
    Bool(bool),
    String(String),
    Char(char),
}

#[derive(Debug, PartialEq, Eq)]
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
}

#[derive(Debug, PartialEq, Eq)]
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

    // ---------- core helpers ----------

    fn peek(&self) -> EtherResult<TokenType> {
        self.tokens
            .get(self.pos)
            .map(|t| t.token_type.clone())
            .ok_or_else(|| EtherError::Parser(ParserError::new("Unexpected end of input".into())))
    }

    fn next(&mut self) -> EtherResult<TokenType> {
        let token = self.peek();
        self.pos += 1;
        token
    }

    fn expect(&mut self, expected: TokenType) -> EtherResult<()> {
        let tok = self.next()?;
        if tok == expected {
            Ok(())
        } else {
            Err(EtherError::Parser(ParserError::new(format!(
                "Expected {:?}, got {:?}",
                expected, tok
            ))))
        }
    }

    fn expect_ident(&mut self) -> EtherResult<String> {
        match self.next()? {
            TokenType::Identifier(s) => Ok(s.clone()),
            t => Err(EtherError::Parser(ParserError::new(format!(
                "Expected identifier, got {:?}",
                t
            )))),
        }
    }

    // ---------- top level ----------

    pub fn parse_program(&mut self) -> EtherResult<Program> {
        let mut imports = Vec::new();
        let mut declarations = Vec::new();

        while self.peek()? == TokenType::Import {
            imports.push(self.parse_import()?);
        }

        while self.peek()? != TokenType::Eof {
            declarations.push(self.parse_declaration()?);
        }

        Ok(Program {
            imports,
            declarations,
        })
    }

    fn parse_import(&mut self) -> EtherResult<Import> {
        self.expect(TokenType::Import)?;
        match self.next()? {
            TokenType::StringLit(s) => Ok(Import { module: s.clone() }),
            t => Err(EtherError::Parser(ParserError::new(format!(
                "Expected string literal, got {:?}",
                t
            )))),
        }
    }

    fn parse_declaration(&mut self) -> EtherResult<Declaration> {
        match self.peek()? {
            TokenType::Fn => {
                self.next();
                let name = self.expect_ident().ok();
                Ok(Declaration::Function(self.parse_function(name)?))
            }
            TokenType::Struct =>{
                self.next();
                Ok(Declaration::Struct(self.parse_struct()?))
            }
            TokenType::Let =>{
                self.next();
                Ok(Declaration::Var(self.parse_var_decl()?))
            }
            t => Err(EtherError::Parser(ParserError::new(format!(
                "Invalid declaration start: {:?}",
                t
            )))),
        }
    }

    // ---------- types ----------

    fn parse_type(&mut self) -> EtherResult<Type> {
        match self.next()? {
            TokenType::Int => Ok(Type::Primitive("int".into())),
            TokenType::Float => Ok(Type::Primitive("float".into())),
            TokenType::Bool => Ok(Type::Primitive("bool".into())),
            TokenType::String => Ok(Type::Primitive("string".into())),
            TokenType::Char => Ok(Type::Primitive("char".into())),
            TokenType::Void => Ok(Type::Primitive("void".into())),
            TokenType::Identifier(name) => Ok(Type::Custom(name.clone())),
            TokenType::LBracket => {
                let t = self.parse_type()?;
                self.expect(TokenType::RBracket)?;
                Ok(Type::Array(Box::new(t)))
            }
            TokenType::LParen => {
                let mut params = Vec::new();
                if self.peek()? != TokenType::RParen {
                    loop {
                        let ptype = self.parse_type()?;
                        params.push((None,ptype));
                        if self.peek()? != TokenType::Comma {
                            break;
                        }
                        self.next()?;
                    }
                }
                self.expect(TokenType::RParen)?;
                self.expect(TokenType::Colon)?;
                let ret = self.parse_type()?;
                Ok(Type::Function(FunctionHeader {
                    name: None,
                    params:params,
                    return_type: Box::new(ret),
                }))
            }
            t => Err(EtherError::Parser(ParserError::new(format!(
                "Invalid type: {:?}",
                t
            )))),
        }
    }

    // ---------- declarations ----------

    fn parse_struct(&mut self) -> EtherResult<StructDef> {
        let name = self.expect_ident()?;
        self.expect(TokenType::LBrace)?;

        let mut fields = Vec::new();
        if self.peek()? != TokenType::RBrace {
            loop {
                let fname = self.expect_ident()?;
                self.expect(TokenType::Colon)?;
                let ftype = self.parse_type()?;
                fields.push((fname, ftype));

                if self.peek()? != TokenType::Comma {
                    break;
                }
                self.next()?;
            }
        }

        self.expect(TokenType::RBrace)?;
        Ok(StructDef { name, fields })
    }

    fn parse_function(&mut self,name:Option<String>) -> EtherResult<Function> {
        self.expect(TokenType::LParen)?;

        let mut params = Vec::new();
        if self.peek()? != TokenType::RParen {
            loop {
                let pname = self.expect_ident()?;
                self.expect(TokenType::Colon)?;
                let ptype = self.parse_type()?;
                params.push((Some(pname), ptype));
                if self.peek()? != TokenType::Comma {
                    break;
                }
                self.next()?;
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

    fn parse_block(&mut self) -> EtherResult<Block> {
        // {  statements }
        self.expect(TokenType::LBrace)?;
        let mut statements = Vec::new();
        while self.peek()? != TokenType::RBrace {
            statements.push(self.parse_stmt()?);
        }
        self.expect(TokenType::RBrace)?;
        Ok(Block { statements })
    }

    // ---------- statements ----------

    fn parse_stmt(&mut self) -> EtherResult<Stmt> {
        // { let|return|if|while|block|  }
        match self.peek()? {
            TokenType::Let =>{
                self.next()?;
                Ok(Stmt::Var(self.parse_var_decl()?)) 
            },
            TokenType::Return => {
                self.next()?;
                let expr = if self.peek()? == TokenType::Semicolon {
                    None
                } else {
                    Some(self.parse_expr()?)
                };
                self.expect(TokenType::Semicolon)?;
                Ok(Stmt::Return(expr))
            }
            TokenType::If => self.parse_if(),
            TokenType::While => self.parse_while(),
            TokenType::For => self.parse_for(),
            TokenType::LBrace => Ok(Stmt::Block(self.parse_block()?)),
            _ => {
                let e = self.parse_expr()?;
                self.expect(TokenType::Semicolon)?;
                Ok(Stmt::Expr(e))
            }
        }
    }

    fn parse_var_decl(&mut self) -> EtherResult<VarDecl> {
        let name = self.expect_ident()?;

        let ty = if self.peek()? == TokenType::Colon {
            self.next()?;
            Some(self.parse_type()?)
        } else {
            None
        };

        self.expect(TokenType::Assign)?;
        let value = self.parse_expr()?;
        self.expect(TokenType::Semicolon)?;

        Ok(VarDecl { name, ty, value })
    }

    fn parse_if(&mut self) -> EtherResult<Stmt> {
        self.expect(TokenType::If)?;
        self.expect(TokenType::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        let then_block = self.parse_block()?;

        let else_block = if self.peek()? == TokenType::Else {
            self.next()?;
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

    fn parse_while(&mut self) -> EtherResult<Stmt> {
        self.expect(TokenType::While)?;
        self.expect(TokenType::LParen)?;
        let cond = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body })
    }

    fn parse_for(&mut self) -> EtherResult<Stmt> {
        self.expect(TokenType::For)?;
        self.expect(TokenType::LParen)?;
        let name = self.expect_ident()?;
        self.expect(TokenType::In)?;
        let iter = self.parse_expr()?;
        self.expect(TokenType::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::For { name, iter, body })
    }

    // ---------- expressions ----------

    fn parse_expr(&mut self) -> EtherResult<Expr> {
        if self.peek()? == TokenType::LParen {
            Ok(Expr::Function(self.parse_function(None)?))
        }else{
            self.parse_assignment()
        }
    }

    fn parse_assignment(&mut self) -> EtherResult<Expr> {
        let left = self.parse_or()?;
        if self.peek()? == TokenType::Assign {
            self.next()?;
            let right = self.parse_assignment()?;
            Ok(Expr::Assign(Box::new(left), Box::new(right)))
        } else {
            Ok(left)
        }
    }

    fn parse_or(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_and()?;
        while self.peek()? == TokenType::Or {
            self.next()?;
            let rhs = self.parse_and()?;
            expr = Expr::Binary(Box::new(expr), BinOp::Or, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_equality()?;
        while self.peek()? == TokenType::And {
            self.next()?;
            let rhs = self.parse_equality()?;
            expr = Expr::Binary(Box::new(expr), BinOp::And, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_equality(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_add()?;
        while let Some(op) = match self.peek()? {
            TokenType::Eq => Some(BinOp::Eq),
            TokenType::Ne => Some(BinOp::Ne),
            TokenType::Lt => Some(BinOp::Lt),
            TokenType::Gt => Some(BinOp::Gt),
            TokenType::Le => Some(BinOp::Le),
            TokenType::Ge => Some(BinOp::Ge),
            _ => None,
        } {
            self.next()?;
            let rhs = self.parse_add()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_add(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_mul()?;
        while let Some(op) = match self.peek()? {
            TokenType::Plus => Some(BinOp::Add),
            TokenType::Minus => Some(BinOp::Sub),
            _ => None,
        } {
            self.next()?;
            let rhs = self.parse_mul()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_mul(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_unary()?;
        while let Some(op) = match self.peek()? {
            TokenType::Multiply => Some(BinOp::Mul),
            TokenType::Divide => Some(BinOp::Div),
            _ => None,
        } {
            self.next()?;
            let rhs = self.parse_unary()?;
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        Ok(expr)
    }

    fn parse_unary(&mut self) -> EtherResult<Expr> {
        match self.peek()? {
            TokenType::Minus => {
                self.next()?;
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()?)))
            }
            TokenType::Not => {
                self.next()?;
                Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_unary()?)))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> EtherResult<Expr> {
        let mut expr = self.parse_primary()?;
        loop {
            match self.peek()? {
                TokenType::LParen => {
                    self.next()?;
                    let mut args = Vec::new();
                    if self.peek()? != TokenType::RParen {
                        args.push(self.parse_expr()?);
                        while self.peek()? == TokenType::Comma {
                            self.next()?;
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(TokenType::RParen)?;
                    expr = Expr::Call(Box::new(expr), args);
                }
                TokenType::Dot => {
                    self.next()?;
                    let field = self.expect_ident()?;
                    expr = Expr::Field(Box::new(expr), field);
                }
                TokenType::LBracket => {
                    self.next()?;
                    let idx = self.parse_expr()?;
                    self.expect(TokenType::RBracket)?;
                    expr = Expr::Index(Box::new(expr), Box::new(idx));
                }
                _ => break,
            }
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> EtherResult<Expr> {
        match self.next()? {
            TokenType::Number(n) => Ok(Expr::Literal(Literal::Int(n.to_string()))),
            TokenType::FloatLit(f) => Ok(Expr::Literal(Literal::Float(f.to_string()))),
            TokenType::StringLit(s) => Ok(Expr::Literal(Literal::String(s.to_string()))),
            TokenType::CharLit(c) => Ok(Expr::Literal(Literal::Char(c))),
            TokenType::True => Ok(Expr::Literal(Literal::Bool(true))),
            TokenType::False => Ok(Expr::Literal(Literal::Bool(false))),
            TokenType::Identifier(name) => Ok(Expr::Identifier(name.clone())),
            TokenType::LParen => {
                let e = self.parse_expr()?;
                self.expect(TokenType::RParen)?;
                Ok(e)
            }
            t => Err(EtherError::Parser(ParserError::new(format!(
                "Invalid expression start: {:?}",
                t
            )))),
        }
    }
}
