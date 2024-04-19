#![warn(rust_2018_idioms)]

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

use itertools::Itertools;
use ndarray::{array, ArrayD, IxDyn};
use parse::{parse_problem, save_problem};
use problem::Problem;

use crate::{baseline::big_loop, localsearch::metaheuristic::{SimAnneal, SimAnnealParams, TabuParams, TabuSearch}, problem::ScheduleType};

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
fn rand_mat_location(n: usize, x: &[f64], y: &[f64], p: f64) -> ArrayD<f64> {
    ArrayD::from_shape_fn(IxDyn(&[n, n]), |i| if i[0] == i[1] {0.0} else {
        ((fastrand::f64()-0.5)*p + ((x[i[0]]-x[i[1]]).powi(2) + (y[i[0]]-y[i[1]]).powi(2)).sqrt()).max(0.0)
    })
}

#[allow(unused)]
fn gen_random_problem(n: usize, train_price: f64, total_budget: f64) -> Problem {
    let track_costs = rand_mat(n);
    let track_times = rand_mat(n);
    let travel_frequencies = rand_mat(n);
    Problem { n, track_costs, track_times, travel_frequencies, train_price, total_budget }
}

#[allow(unused)]
fn gen_random_problem_location(n: usize, train_price: f64, total_budget: f64) -> Problem {
    let x = (0..n).map(|_| fastrand::f64()).collect_vec();
    let y = (0..n).map(|_| fastrand::f64()).collect_vec();
    let track_costs = rand_mat_location(n, &x, &y, 0.05);
    let track_times = rand_mat_location(n, &x, &y, 0.05);
    let travel_frequencies = rand_mat_location(n, &x, &y, 0.4);
    Problem { n, track_costs, track_times, travel_frequencies, train_price, total_budget }
}

fn main() {
    // let problem = parse_problem("test_problem.toml");
    // let problem = gen_random_problem(40, 1.0, 100.0);
    // let problem = gen_random_problem_location(40, 1.0, 100.0);
    // save_problem("semi_large_random_problem_location.toml", &problem);
    let problem = parse_problem("medium_random_problem.toml");

    let solution = big_loop(&problem, ScheduleType::Bidirectional);
    dbg!(&solution);
    println!("{}", solution.check_feasibility(&problem));

    let solver2 = localsearch::Solver::<SimAnneal> {
        problem: &problem, max_iterations: 100_000, neighbour_chance: 1.0,
        mh_params: SimAnnealParams {
            initial_temp: 540.0,
            temp_scale: (1.0/540.0f64).powf(1.0/100_000.0),
        }
    };
    let solver = localsearch::Solver::<TabuSearch> {
        problem: &problem, max_iterations: 1000, neighbour_chance: 0.8,
        mh_params: TabuParams {
            initial_timeout: 1000,
            size_adjust: 10,
        }
    };
    let solution2 = solver.solve();
    let solution3 = solver2.solve();
    dbg!(&solution2); dbg!(&solution3);
    println!("Tabu: {}, SA: {}", solution2.obj_value, solution3.obj_value);
    // dbg!(&solution3);
    // println!("{}", solution3.obj_value);
}