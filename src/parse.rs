//! Parses a problem from a file to the internal problem representation

use std::{fs::{self, File}, io::Write};

use crate::problem::Problem;

/// Reads a problem from a file, in TOML format
pub fn parse_problem(file_name: &str) -> Problem {
    let file_contents = fs::read_to_string(file_name).unwrap();
    Problem::new(toml::from_str(&file_contents).unwrap())
}

/// Saves a problem in TOML format to a file
pub fn save_problem(file_name: &str, problem: &Problem) {
    let mut file = File::create(file_name).unwrap();
    write!(file, "{}", toml::to_string(&problem.description).unwrap()).unwrap();
}