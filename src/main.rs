#![warn(rust_2018_idioms)]

use ndarray::{array, IxDyn};
use parse::{parse_problem, save_problem};
use problem::{Problem, ProblemDescription};

use crate::{baseline::big_loop, problem::ScheduleType};

mod baseline;
mod evaluate;
mod localsearch;
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
        total_budget: 1000.0,
    };
    let problem = Problem::new(problem_desc);
    save_problem("test_problem.toml", &problem);
}


fn main() {
    let problem = parse_problem("test_problem.toml");
    dbg!(&problem);

    let solution = big_loop(&problem, ScheduleType::Bidirectional);
    dbg!(&solution);
    println!("{}", solution.check_feasibility(&problem));

    let solver = localsearch::Solver { problem: &problem, max_iterations: 10000, neighbour_chance: 1.0, tabu_size: 100 };
    let solution2 = solver.solve();
    dbg!(&solution2);
}