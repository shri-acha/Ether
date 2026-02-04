use std::collections::HashMap;

use crate::error::{EtherError, TypeErrorDetail};
use crate::parser::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolKind {
    Variable,
    Function,
    Parameter,
    Struct,
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub scope_level: usize,
}

#[derive(Debug)]
pub struct SymbolTable {
    scopes: Vec<HashMap<String, Symbol>>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn current_scope_level(&self) -> usize {
        self.scopes.len() - 1
    }

    pub fn insert(&mut self, name: String, kind: SymbolKind) -> Result<(), EtherError> {
        // FIX: compute scope_level BEFORE mutable borrow
        let scope_level = self.current_scope_level();
        let current_scope = self.scopes.last_mut().unwrap();

        if current_scope.contains_key(&name) {
            return Err(EtherError::TypeInfer(TypeErrorDetail {
                err_string: format!("Duplicate symbol '{}'", name),
                line: 0,
                column: 0,
            }));
        }

        current_scope.insert(
            name.clone(),
            Symbol {
                name,
                kind,
                scope_level,
            },
        );

        Ok(())
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        for scope in self.scopes.iter().rev() {
            if let Some(sym) = scope.get(name) {
                return Some(sym);
            }
        }
        None
    }
}

pub struct SymbolResolver {
    pub table: SymbolTable,
}

impl SymbolResolver {
    pub fn new() -> Self {
        Self {
            table: SymbolTable::new(),
        }
    }

    pub fn analyze_program(&mut self, program: &Program) -> Result<(), EtherError> {
        for decl in &program.declarations {
            self.analyze_declaration(decl)?;
        }
        Ok(())
    }

    fn analyze_declaration(&mut self, decl: &Declaration) -> Result<(), EtherError> {
        match decl {
            Declaration::Function(func) => self.analyze_function(func),

            Declaration::Var(var) => self.table.insert(var.name.clone(), SymbolKind::Variable),

            Declaration::Struct(def) => self.table.insert(def.name.clone(), SymbolKind::Struct),

            // FIX: handle Enum explicitly
            Declaration::Enum(def) => self.table.insert(def.name.clone(), SymbolKind::Struct),
        }
    }

    fn analyze_function(&mut self, func: &Function) -> Result<(), EtherError> {
        // FIX: function name is Option<String>
        if let Some(name) = &func.header.name {
            self.table.insert(name.clone(), SymbolKind::Function)?;
        }

        self.table.enter_scope();

        // FIX: parameters are (Option<String>, Type)
        for (param_name, _) in &func.header.params {
            if let Some(name) = param_name {
                self.table.insert(name.clone(), SymbolKind::Parameter)?;
            }
        }

        self.analyze_block(&func.body)?;

        self.table.exit_scope();
        Ok(())
    }

    fn analyze_block(&mut self, block: &Block) -> Result<(), EtherError> {
        self.table.enter_scope();

        for stmt in &block.statements {
            self.analyze_stmt(stmt)?;
        }

        self.table.exit_scope();
        Ok(())
    }

    fn analyze_stmt(&mut self, stmt: &Stmt) -> Result<(), EtherError> {
        match stmt {
            Stmt::Var(var) => self.table.insert(var.name.clone(), SymbolKind::Variable),

            Stmt::Block(block) => self.analyze_block(block),

            // FIX: struct-style enum variants
            Stmt::If {
                cond: _,
                then_block,
                else_block,
            } => {
                self.analyze_block(then_block)?;
                if let Some(else_blk) = else_block {
                    self.analyze_block(else_blk)?;
                }
                Ok(())
            }

            Stmt::While { cond: _, body } => self.analyze_block(body),

            Stmt::For {
                name,
                iter: _,
                body,
            } => {
                self.table.enter_scope();
                self.table.insert(name.clone(), SymbolKind::Variable)?;
                self.analyze_block(body)?;
                self.table.exit_scope();
                Ok(())
            }

            Stmt::Return(_) | Stmt::Expr(_) => Ok(()),
        }
    }
}
