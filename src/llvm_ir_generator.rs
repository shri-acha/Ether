use inkwell::IntPredicate;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::values::{FunctionValue, IntValue};

// Example AST nodes
enum Expr {
    Number(i64),
    Add(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
}

pub struct Compiler<'ctx> {
    context: &'ctx Context,
    builder: Builder<'ctx>,
    module: Module<'ctx>,
}

impl<'ctx> Compiler<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("my_module");
        let builder = context.create_builder();

        Compiler {
            context,
            builder,
            module,
        }
    }

    pub fn compile_expr(&self, expr: &Expr) -> IntValue<'ctx> {
        match expr {
            Expr::Number(n) => self.context.i64_type().const_int(*n as u64, false),
            Expr::Add(left, right) => {
                let lhs = self.compile_expr(left);
                let rhs = self.compile_expr(right);
                self.builder
                    .build_int_add(lhs, rhs, "addtmp")
                    .expect("Failed to build add")
            }
            Expr::Multiply(left, right) => {
                let lhs = self.compile_expr(left);
                let rhs = self.compile_expr(right);
                self.builder
                    .build_int_mul(lhs, rhs, "multmp")
                    .expect("Failed to build mul")
            }
        }
    }

    pub fn compile(&self) {
        // Create a function: i64 main()
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[], false);
        let function = self.module.add_function("main", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        // Example: compile (2 + 3) * 4
        let ast = Expr::Multiply(
            Box::new(Expr::Add(
                Box::new(Expr::Number(2)),
                Box::new(Expr::Number(3)),
            )),
            Box::new(Expr::Number(4)),
        );

        let result = self.compile_expr(&ast);
        self.builder
            .build_return(Some(&result))
            .expect("Failed to build return");
    }

    pub fn print_ir(&self) {
        self.module.print_to_stderr();
    }
}

fn main() {
    let context = Context::create();
    let compiler = Compiler::new(&context);

    compiler.compile();
    compiler.print_ir();
}
