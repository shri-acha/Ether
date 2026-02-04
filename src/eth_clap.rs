use crate::parser::Parser;
use crate::lexer::Tokenizer;
use inkwell::context::Context; 
use inkwell::targets::{Target, TargetMachine, InitializationConfig, RelocMode, CodeModel, FileType};
use inkwell::OptimizationLevel;
use crate::codegen::CodeGen;
use std::path::Path;
use std::fs;

#[derive(clap::Parser, Debug)]
#[command(version, about, long_about = None)]
pub(crate) struct EthArgs {
    /// Level of optimization 0,1,2,3 
    #[arg(short = 'O', long, default_value_t = 0)]
    level_of_optimization: u8, 
    /// Files that you want to link post compilation 
    #[arg(short, long)]
    link: Vec<String>, 
    /// Relocation stratergy 
    #[arg(short, long,default_value_t =String::from(""))]
    relocation_stratergy: String,
    #[arg(short, long, default_value_t = String::from("e.o"))]
    output: String,
    /// Files that you want to compile 
    #[arg(required = true)]
    files: Vec<String>,
}

pub fn handle_eth_args(args: &EthArgs) {

    Target::initialize_native(&InitializationConfig::default())
        .expect("Failed to initialize native target");

    let opt_level = match args.level_of_optimization {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        3 => OptimizationLevel::Aggressive,
        _ => {
            eprintln!("Warning: Invalid optimization level. Defaulting to None.");
            OptimizationLevel::None
        }
    };

    let reloc_stratergy = match args.relocation_stratergy.as_str() {
        "pic" => RelocMode::PIC,
        "sta" => RelocMode::Static,
        "dyn" => RelocMode::DynamicNoPic,
        _ => RelocMode::Default,
    };



    for input_file in &args.files{

        let contents = fs::read_to_string(input_file).expect("Failed to read file"); // TODO: work
                                                                                     // on
                                                                                     // buffering
                                                                                     // input file
        let mut tokenizer = Tokenizer::new(&contents);
        let tokens = tokenizer.tokenize(true);
        let mut parser = Parser::new(tokens);
        let parsed_tokens = parser.parse_program().unwrap();

        let context = Context::create();
        let mut codegen = CodeGen::new(&context, &args.output);

        codegen.compile_program(&parsed_tokens).expect("Failed to compile LLVM");

        let execution_engine = codegen
            .get_module()
            .create_jit_execution_engine(opt_level)
            .expect("Failed to create JIT execution engine");

        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).expect("Failed to get target from triple");
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                opt_level,
                reloc_stratergy,
                CodeModel::Default,
            )
            .expect("Failed to create target machine");

        let output_file = Path::new(&args.output);
        target_machine
                .write_to_file(codegen.get_module(), FileType::Object, &output_file)
                .expect("Failed to write object file");
    }
}
