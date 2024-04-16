#![warn(rust_2018_idioms)]

use ndarray::{array, ArrayD, IxDyn};
use parse::{parse_problem, save_problem};
use problem::Problem;

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
    let problem = Problem {
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
    save_problem("test_problem.toml", &problem);
}

#[allow(unused)]
fn rand_mat(n: usize) -> ArrayD<f64> {
    ArrayD::from_shape_fn(IxDyn(&[n, n]), |i| if i[0] == i[1] {0.0} else {fastrand::f64()})
}

#[allow(unused)]
fn gen_random_problem(n: usize, train_price: f64, total_budget: f64) -> Problem {
    let track_costs = rand_mat(n);
    let track_times = rand_mat(n);
    let travel_frequencies = rand_mat(n);
    Problem { n, track_costs, track_times, travel_frequencies, train_price, total_budget }
}

fn main() {
    // let problem = parse_problem("test_problem.toml");
    // let problem = gen_random_problem(100, 1.0, 100.0);
    // save_problem("large_random_problem.toml", &problem);
    let problem = parse_problem("medium_random_problem.toml");
    // dbg!(&problem);

    let solution = big_loop(&problem, ScheduleType::Bidirectional);
    dbg!(&solution);
    println!("{}", solution.check_feasibility(&problem));

    let solver = localsearch::Solver { problem: &problem, max_iterations: 10_000, neighbour_chance: 0.7, tabu_initial_timeout: 10_000 };
    let solution2 = solver.solve();
    dbg!(&solution2);
}