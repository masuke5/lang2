#![warn(rust_2018_idioms, unused_import_braces)]
#![deny(trivial_casts, trivial_numeric_casts, elided_lifetimes_in_paths)]

use std::fs;
use std::path::PathBuf;
use std::process::exit;

use lang2::error::ErrorList;
use lang2::id::{reserved_id, IdMap};
use lang2::{ExecuteMode, ExecuteOption, OptimizeOption};

use clap::{App, Arg, ArgMatches};

fn get_option<'a>(matches: &'a ArgMatches<'a>) -> Result<ExecuteOption, String> {
    if let Some(filepath_str) = matches.value_of("file") {
        // Read the file if a file path is specified
        let input = fs::read_to_string(filepath_str).map_err(|err| format!("{}", err))?;

        let filepath_id = IdMap::new_id(&filepath_str);
        let filepath = PathBuf::from(filepath_str);
        let module_name = filepath.file_stem().unwrap();
        let module_name = IdMap::new_id(&format!("::{}", &module_name.to_string_lossy()));

        Ok(ExecuteOption::new(input, filepath_id, module_name).file_path(filepath))
    } else if let Some(input) = matches.value_of("cmd") {
        Ok(ExecuteOption::new(
            input.to_string(),
            *reserved_id::CMD,
            *reserved_id::CMD,
        ))
    } else {
        Err(String::from("Not specified file or cmd"))
    }
}

fn main() {
    let matches = App::new("lang2")
        .version("0.0")
        .author("masuke5 <s.zerogoichi@gmail.com>")
        .about("lang2 interpreter")
        .arg(
            Arg::with_name("file")
                .help("Runs file")
                .index(1)
                .required(false),
        )
        .arg(
            Arg::with_name("cmd")
                .short("c")
                .long("cmd")
                .help("Runs string")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("dump-token")
                .long("dump-token")
                .help("Dumps tokens"),
        )
        .arg(
            Arg::with_name("dump-ast")
                .long("dump-ast")
                .help("Dumps AST"),
        )
        .arg(Arg::with_name("dump-ir").long("dump-ir").help("Dumps IR"))
        .arg(
            Arg::with_name("dump-insts")
                .long("dump-insts")
                .help("Dumps instructions"),
        )
        .arg(
            Arg::with_name("enable-trace")
                .long("trace")
                .help("Traces instructions"),
        )
        .arg(
            Arg::with_name("enable-measure")
                .long("measure")
                .help("Measures the performance"),
        )
        .arg(
            Arg::with_name("enable-optimization")
                .long("opt")
                .help("Enables optimization"),
        )
        .get_matches();

    let mode = if matches.is_present("dump-token") {
        ExecuteMode::DumpToken
    } else if matches.is_present("dump-ast") {
        ExecuteMode::DumpAST
    } else if matches.is_present("dump-insts") {
        ExecuteMode::DumpInstruction
    } else if matches.is_present("dump-ir") {
        ExecuteMode::DumpIR
    } else {
        ExecuteMode::Normal
    };

    let option = match get_option(&matches) {
        Ok(t) => t,
        Err(err) => {
            eprintln!("Unable to load input: {}", err);
            exit(1);
        }
    };

    let optimize_option = if matches.is_present("enable-optimization") {
        OptimizeOption {
            calc_at_compile_time: true,
        }
    } else {
        OptimizeOption {
            calc_at_compile_time: false,
        }
    };

    option
        .enable_trace(matches.is_present("enable-trace"))
        .enable_measure(matches.is_present("enable-measure"))
        .mode(mode)
        .optimize_option(optimize_option)
        .execute();

    if ErrorList::has_error() {
        exit(1);
    }
}
