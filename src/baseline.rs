//! Implements a few extremely simple solvers to use as a baseline
//! for comparison to other solvers.
//! Note that due to their simplicty, many of these may violate
//! budget constraints.

use itertools::Itertools;
use ndarray::ArrayD;

use crate::{evaluate::evaluate, problem::{Problem, ScheduleType, Solution}};

/// Generates a single train that visits every station
pub fn big_loop(problem: &Problem, ty: ScheduleType) -> Solution {
    let train_routes = vec![(0..problem.description.n).collect_vec()];
    let mut built_tracks = ArrayD::<bool>::default(problem.description.track_costs.shape());
    for i in 0..problem.description.n-1 {
        built_tracks[[i, i+1]] = true; built_tracks[[i+1, i]] = true;
    }
    built_tracks[[0, problem.description.n-1]] = true; built_tracks[[problem.description.n-1, 0]] = true;
    let train_types = vec![ty];
    let obj_value = evaluate(problem, &train_routes, &train_types);

    Solution {
        n_trains: 1,
        built_tracks,
        train_routes,
        train_types,
        obj_value
    }
}