#![warn(rust_2018_idioms)]

use ndarray::{array, IxDyn};
use parse::{parse_problem, save_problem};
use problem::{Problem, ProblemDescription};

use crate::{evaluate::evaluate, problem::ScheduleType};

mod baseline;
mod evaluate;
mod parse;
mod problem;


/// Tests the `save_problem` function by writing a small example
/// problem to a file.
#[allow(unused)]
fn save_example_problem() {
    let problem_desc = ProblemDescription {
        n: 3,
        track_costs: array![
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0]
        ].into_shape(IxDyn(&[3, 3])).unwrap(),
        track_times: array![
            [0.0, 3.0, 2.0],
            [3.0, 0.0, 4.0],
            [2.0, 4.0, 0.0]
        ].into_shape(IxDyn(&[3, 3])).unwrap(),
        travel_frequencies: array![
            [0.0, 5.0, 1.0],
            [5.0, 0.0, 2.0],
            [1.0, 2.0, 0.0]
        ].into_shape(IxDyn(&[3, 3])).unwrap(),
        train_price: 10.0,
        total_budget: 30.0,
    };
    let problem = Problem::new(problem_desc);
    save_problem("test_problem.toml", &problem);
}


fn main() {
    save_example_problem();
    let problem = parse_problem("test_problem.toml");
    dbg!(&problem);
    let score = evaluate(&problem, &[vec![0, 1, 2]], &[ScheduleType::Circular]);
    println!("{score}");
}