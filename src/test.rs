use std::fs;

use ndarray::{ArrayD, IxDyn};

use crate::{baseline::big_loop, gen_random_problem, parse::{parse_problem, save_problem}, problem::{ScheduleType, Solution, TrainLine}};


/// Tests saving and loading capabilities, ensuring that
/// problem data is consistently (de)serialised.
#[test]
fn test_problem_serde() {
    let problem = gen_random_problem(10, 1.0, 1.0);
    save_problem("__test.toml", &problem);
    let problem2 = parse_problem("__test.toml");
    assert_eq!(problem, problem2, "Ensure problem data (de)serialises consistently");
    fs::remove_file("__test.toml").unwrap();
}

/// Ensures cost is calculated correctly, given a solution's description
#[test]
fn test_solution_cost() {
    let problem = parse_problem("test_problem.toml");
    let solution = Solution {
        built_tracks: ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![
            false, true, true,
            true, false, false,
            true, false, false,
        ]).unwrap(),
        train_lines: vec![TrainLine { route: vec![0, 1, 2], ty: ScheduleType::Bidirectional, n: 3 }],
        obj_value: 0.0, // arbitrary
    };
    let cost = solution.cost(&problem);
    assert_eq!(cost, 3.0 + 3.0*5.0, "Ensure cost is calculated correctly");
    assert!(solution.check_feasibility(&problem));
}


/// Ensures big loop baseline solution is constructed correctly
#[test]
fn test_big_loop() {
    let problem = parse_problem("test_problem.toml");
    let ref_sol1 = Solution {
        built_tracks: ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![
            false, true, false,
            true, false, true,
            false, true, false,
        ]).unwrap(),
        train_lines: vec![TrainLine { route: vec![0, 1, 2], ty: ScheduleType::Bidirectional, n: 1 }],
        obj_value: 25.0,
    };
    let sol1 = big_loop(&problem, ScheduleType::Bidirectional);
    assert_eq!(sol1, ref_sol1, "Ensure big loop is constructed correctly (bidirectionally)");

    let ref_sol2 = Solution {
        built_tracks: ArrayD::from_shape_vec(IxDyn(&[3, 3]), vec![
            false, true, true,
            true, false, true,
            true, true, false,
        ]).unwrap(),
        train_lines: vec![TrainLine { route: vec![0, 1, 2], ty: ScheduleType::Circular, n: 1 }],
        obj_value: 30.0,
    };
    let sol2 = big_loop(&problem, ScheduleType::Circular);
    assert_eq!(sol2, ref_sol2, "Ensure big loop is constructed correctly (circular)");
}