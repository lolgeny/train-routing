//! This module contains interfaces for the solver: it has the `Problem` struct, which describes a train routing problem,
//! and the `Solution` struct, which is what the solver returns and represents the optimal solution

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// A description of a general train route problem
#[derive(Deserialize, Serialize, Debug, Clone, PartialEq)]
pub struct Problem {
    /// The number of stations
    pub n: usize,
    /// A symmetric matrix representing the cost to build tracks between two stations
    pub track_costs: ArrayD<f64>,
    /// A symmetric matrix representing the time to travel between two stations, if a track is built
    pub track_times: ArrayD<f64>,
    /// A symmetric matrix representing the frequency between which two stations are travelled
    pub travel_frequencies: ArrayD<f64>,
    /// The price per train
    pub train_price: f64,
    /// The total amount of money that can be allocated
    pub total_budget: f64
}

/// Represents which type of line a train follows:
/// 
/// - `Circular` means it goes to the first station after the last one
/// 
/// - `Bidirectional` means it repeats the track, reversed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScheduleType {
    Circular, Bidirectional
}

/// A train line: its schedule, with how many trains it runs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TrainLine {
    /// A list of stations which trains on this line visit
    pub route: Vec<usize>,
    /// How the train follows this route
    pub ty: ScheduleType,
    /// The number of trains that use this line
    pub n: usize
}

/// The solver's optimal solution to the problem
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// A symmetric matrix showing which tracks are built
    pub built_tracks: ArrayD<bool>,
    /// A list of train lines descriptions
    pub train_lines: Vec<TrainLine>,
    /// The objective value, representing how good the solution is,
    /// where lower is better
    pub obj_value: f64
}
impl Solution {
    /// Calculate the cost of a solution
    pub fn cost(&self, problem: &Problem) -> f64 {
        (self.built_tracks.map(|&x| if x {1.0} else {0.0}) * &problem.track_costs).sum() / 2.0
        + self.train_lines.iter().map(|t| t.n).sum::<usize>() as f64 * problem.train_price
    }
    /// Ensures a solution is feasible by checking it is within budget
    pub fn check_feasibility(&self, problem: &Problem) -> bool {
        self.cost(problem) <= problem.total_budget
    }
}