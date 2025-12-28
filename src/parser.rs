#![allow(dead_code)] //only for now since ast isnt being usued anywhere yet

use crate::lexer::{Token, TokenType};

#[derive(Debug)]
pub struct Program {
    pub imports: Vec<Import>,
    pub declarations: Vec<Declaration>,
}

#[derive(Debug)]
pub struct Import {
    pub module: String,
}

#[derive(Debug)]
pub enum Declaration {
    Function(Function),
    Struct(StructDef),
    Var(VarDecl),
}

//  types 
#[derive(Debug, Clone)]
pub enum Type {
    Primitive(String),
    Array(Box<Type>),
    Function(Vec<Type>, Box<Type>),
    Custom(String),
}

// ast 
#[derive(Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, Type)>,
}


#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub return_type: Type,
    pub body: Block,
}

#[derive(Debug)]
pub struct VarDecl {
    pub name: String,
    pub ty: Option<Type>,
    pub value: Expr,
}

#[derive(Debug)]
pub struct Block {
    pub statements: Vec<Stmt>,
}

#[derive(Debug)]
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

//expression expr
#[derive(Debug)]
pub enum Expr {
    Literal(Literal),
    Identifier(String),
    Assign(Box<Expr>, Box<Expr>),
    Binary(Box<Expr>, BinOp, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Field(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
}

#[derive(Debug)]
pub enum Literal {
    Int(String),
    Float(String),
    Bool(bool),
    String(String),
    Char(char),
}

#[derive(Debug)]
pub enum BinOp {
    Add, Sub, Mul, Div,
    Eq, Ne, Lt, Gt, Le, Ge,
    And, Or,
}

#[derive(Debug)]
pub enum UnOp {
    Neg,
    Not,
}

//parser ahhh
pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &TokenType {
        &self.tokens[self.pos].token_type
    }

    fn next(&mut self) -> &TokenType {
        self.pos += 1;
        &self.tokens[self.pos - 1].token_type
    }

    fn expect(&mut self, expected: TokenType) {
        let tok = self.next();
        if *tok != expected {
            panic!("Expected {:?}, got {:?}", expected, tok);
        }
    }

    pub fn parse_program(&mut self) -> Program {
        let mut imports = Vec::new();
        let mut declarations = Vec::new();

        while matches!(self.peek(), TokenType::Import) {
            imports.push(self.parse_import());
        }

        while !matches!(self.peek(), TokenType::Eof) {
            declarations.push(self.parse_declaration());
        }

        Program { imports, declarations }
    }

    fn parse_import(&mut self) -> Import {
        self.expect(TokenType::Import);
        match self.next() {
            TokenType::StringLit(s) => Import { module: s.clone() },
            t => panic!("Expected string literal, got {:?}", t),
        }
    }

    fn parse_declaration(&mut self) -> Declaration {
        match self.peek() {
            TokenType::Fn => Declaration::Function(self.parse_function()),
            TokenType::Struct => Declaration::Struct(self.parse_struct()),
            TokenType::Let => Declaration::Var(self.parse_var_decl()),
            t => panic!("Invalid declaration start: {:?}", t),
        }
    }


    fn parse_type(&mut self) -> Type {
        match self.next() {
            TokenType::Int => Type::Primitive("int".into()),
            TokenType::Float => Type::Primitive("float".into()),
            TokenType::Bool => Type::Primitive("bool".into()),
            TokenType::String => Type::Primitive("string".into()),
            TokenType::Char => Type::Primitive("char".into()),
            TokenType::Void => Type::Primitive("void".into()),
            TokenType::Identifier(name) => Type::Custom(name.clone()),
            TokenType::LBracket => {
                let t = self.parse_type();
                self.expect(TokenType::RBracket);
                Type::Array(Box::new(t))
            }
            TokenType::LParen => {
                let mut params = Vec::new();
                if !matches!(self.peek(), TokenType::RParen) {
                    params.push(self.parse_type());
                    while matches!(self.peek(), TokenType::Comma) {
                        self.next();
                        params.push(self.parse_type());
                    }
                }
                self.expect(TokenType::RParen);
                self.expect(TokenType::Colon);
                let ret = self.parse_type();
                Type::Function(params, Box::new(ret))
            }
            t => panic!("Invalid type: {:?}", t),
        }
    }


    fn parse_struct(&mut self) -> StructDef {
        self.expect(TokenType::Struct);
        let name = self.expect_ident();
        self.expect(TokenType::LBrace);

        let mut fields = Vec::new();
        if !matches!(self.peek(), TokenType::RBrace) {
            loop {
                let fname = self.expect_ident();
                self.expect(TokenType::Colon);
                let ftype = self.parse_type();
                fields.push((fname, ftype));

                if !matches!(self.peek(), TokenType::Comma) {
                    break;
                }
                self.next();
            }
        }

        self.expect(TokenType::RBrace);
        StructDef { name, fields }
    }

    fn parse_function(&mut self) -> Function {
        self.expect(TokenType::Fn);
        let name = self.expect_ident();

        self.expect(TokenType::LParen);
        let mut params = Vec::new();
        if !matches!(self.peek(), TokenType::RParen) {
            loop {
                let pname = self.expect_ident();
                self.expect(TokenType::Colon);
                let ptype = self.parse_type();
                params.push((pname, ptype));
                if !matches!(self.peek(), TokenType::Comma) {
                    break;
                }
                self.next();
            }
        }
        self.expect(TokenType::RParen);

        self.expect(TokenType::Colon);
        let return_type = self.parse_type();
        let body = self.parse_block();

        Function { name, params, return_type, body }
    }

    fn parse_block(&mut self) -> Block {
        self.expect(TokenType::LBrace);
        let mut statements = Vec::new();
        while !matches!(self.peek(), TokenType::RBrace) {
            statements.push(self.parse_stmt());
        }
        self.expect(TokenType::RBrace);
        Block { statements }
    }

    fn parse_stmt(&mut self) -> Stmt {
        match self.peek() {
            TokenType::Let => Stmt::Var(self.parse_var_decl()),
            TokenType::Return => {
                self.next();
                let expr = if matches!(self.peek(), TokenType::Semicolon) {
                    None
                } else {
                    Some(self.parse_expr())
                };
                self.expect(TokenType::Semicolon);
                Stmt::Return(expr)
            }
            TokenType::If => self.parse_if(),
            TokenType::While => self.parse_while(),
            TokenType::For => self.parse_for(),
            TokenType::LBrace => Stmt::Block(self.parse_block()),
            _ => {
                let e = self.parse_expr();
                self.expect(TokenType::Semicolon);
                Stmt::Expr(e)
            }
        }
    }

    fn parse_var_decl(&mut self) -> VarDecl {
        self.expect(TokenType::Let);
        let name = self.expect_ident();
        let ty = if matches!(self.peek(), TokenType::Colon) {
            self.next();
            Some(self.parse_type())
        } else {
            None
        };
        self.expect(TokenType::Assign);
        let value = self.parse_expr();
        self.expect(TokenType::Semicolon);
        VarDecl { name, ty, value }
    }

    fn parse_if(&mut self) -> Stmt {
        self.expect(TokenType::If);
        self.expect(TokenType::LParen);
        let cond = self.parse_expr();
        self.expect(TokenType::RParen);
        let then_block = self.parse_block();

        let else_block = if matches!(self.peek(), TokenType::Else) {
            self.next();
            Some(self.parse_block())
        } else {
            None
        };

        Stmt::If { cond, then_block, else_block }
    }

    fn parse_while(&mut self) -> Stmt {
        self.expect(TokenType::While);
        self.expect(TokenType::LParen);
        let cond = self.parse_expr();
        self.expect(TokenType::RParen);
        let body = self.parse_block();
        Stmt::While { cond, body }
    }

    fn parse_for(&mut self) -> Stmt {
        self.expect(TokenType::For);
        self.expect(TokenType::LParen);
        let name = self.expect_ident();
        self.expect(TokenType::In);
        let iter = self.parse_expr();
        self.expect(TokenType::RParen);
        let body = self.parse_block();
        Stmt::For { name, iter, body }
    }

    fn parse_expr(&mut self) -> Expr {
        self.parse_assignment()
    }

    fn parse_assignment(&mut self) -> Expr {
        let left = self.parse_or();
        if matches!(self.peek(), TokenType::Assign) {
            self.next();
            let right = self.parse_assignment();
            Expr::Assign(Box::new(left), Box::new(right))
        } else {
            left
        }
    }

    fn parse_or(&mut self) -> Expr {
        let mut expr = self.parse_and();
        while matches!(self.peek(), TokenType::Or) {
            self.next();
            let rhs = self.parse_and();
            expr = Expr::Binary(Box::new(expr), BinOp::Or, Box::new(rhs));
        }
        expr
    }

    fn parse_and(&mut self) -> Expr {
        let mut expr = self.parse_equality();
        while matches!(self.peek(), TokenType::And) {
            self.next();
            let rhs = self.parse_equality();
            expr = Expr::Binary(Box::new(expr), BinOp::And, Box::new(rhs));
        }
        expr
    }

    fn parse_equality(&mut self) -> Expr {
        let mut expr = self.parse_add();
        while let Some(op) = match self.peek() {
            TokenType::Eq => Some(BinOp::Eq),
            TokenType::Ne => Some(BinOp::Ne),
            TokenType::Lt => Some(BinOp::Lt),
            TokenType::Gt => Some(BinOp::Gt),
            TokenType::Le => Some(BinOp::Le),
            TokenType::Ge => Some(BinOp::Ge),
            _ => None,
        } {
            self.next();
            let rhs = self.parse_add();
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        expr
    }

    fn parse_add(&mut self) -> Expr {
        let mut expr = self.parse_mul();
        while let Some(op) = match self.peek() {
            TokenType::Plus => Some(BinOp::Add),
            TokenType::Minus => Some(BinOp::Sub),
            _ => None,
        } {
            self.next();
            let rhs = self.parse_mul();
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        expr
    }

    fn parse_mul(&mut self) -> Expr {
        let mut expr = self.parse_unary();
        while let Some(op) = match self.peek() {
            TokenType::Multiply => Some(BinOp::Mul),
            TokenType::Divide => Some(BinOp::Div),
            _ => None,
        } {
            self.next();
            let rhs = self.parse_unary();
            expr = Expr::Binary(Box::new(expr), op, Box::new(rhs));
        }
        expr
    }

    fn parse_unary(&mut self) -> Expr {
        match self.peek() {
            TokenType::Minus => {
                self.next();
                Expr::Unary(UnOp::Neg, Box::new(self.parse_unary()))
            }
            TokenType::Not => {
                self.next();
                Expr::Unary(UnOp::Not, Box::new(self.parse_unary()))
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> Expr {
        let mut expr = self.parse_primary();
        loop {
            match self.peek() {
                TokenType::LParen => {
                    self.next();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), TokenType::RParen) {
                        args.push(self.parse_expr());
                        while matches!(self.peek(), TokenType::Comma) {
                            self.next();
                            args.push(self.parse_expr());
                        }
                    }
                    self.expect(TokenType::RParen);
                    expr = Expr::Call(Box::new(expr), args);
                }
                TokenType::Dot => {
                    self.next();
                    let field = self.expect_ident();
                    expr = Expr::Field(Box::new(expr), field);
                }
                TokenType::LBracket => {
                    self.next();
                    let idx = self.parse_expr();
                    self.expect(TokenType::RBracket);
                    expr = Expr::Index(Box::new(expr), Box::new(idx));
                }
                _ => break,
            }
        }
        expr
    }

    fn parse_primary(&mut self) -> Expr {
        match self.next() {
            TokenType::Number(n) => Expr::Literal(Literal::Int(n.clone())),
            TokenType::FloatLit(f) => Expr::Literal(Literal::Float(f.clone())),
            TokenType::StringLit(s) => Expr::Literal(Literal::String(s.clone())),
            TokenType::CharLit(c) => Expr::Literal(Literal::Char(*c)),
            TokenType::True => Expr::Literal(Literal::Bool(true)),
            TokenType::False => Expr::Literal(Literal::Bool(false)),
            TokenType::Identifier(name) => Expr::Identifier(name.clone()),
            TokenType::LParen => {
                let e = self.parse_expr();
                self.expect(TokenType::RParen);
                e
            }
            t => panic!("Invalid expression start: {:?}", t),
        }
    }

    fn expect_ident(&mut self) -> String {
        match self.next() {
            TokenType::Identifier(s) => s.clone(),
            t => panic!("Expected identifier, got {:?}", t),
        }
    }
}
