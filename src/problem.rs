//! This module contains interfaces for the solver: it has the `Problem` struct, which describes a train routing problem,
//! and the `Solution` struct, which is what the solver returns and represents the optimal solution

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

/// A description of a general train route problem
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct ProblemDescription {
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

/// A train routing problem, with its description and cached information
#[derive(Debug, Clone)]
pub struct Problem {
    pub description: ProblemDescription
}
impl Problem {
    /// Create a new problem from its descriptionI
    pub fn new(description: ProblemDescription) -> Self {
        Self {
            description
        }
    }
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

/// The solver's optimal solution to the problem
#[derive(Debug, Clone)]
pub struct Solution {
    /// The number of trains/lines to build
    pub n_trains: usize,
    /// A symmetric matrix showing which tracks are built
    pub built_tracks: ArrayD<bool>,
    /// A list of routes for trains to follow
    pub train_routes: Vec<Vec<usize>>,
    /// A list of schedule types that trains follow
    pub train_types: Vec<ScheduleType>,
    /// The objective value, representing how good the solution is,
    /// where lower is better
    pub obj_value: f64
}
impl Solution {
    /// Ensures a solution is feasible by checking it both logically consistent and within budget
    pub fn check_feasibility(&self, problem: &Problem) -> bool {
        let cost = 
            (self.built_tracks.map(|&x| if x {1.0} else {0.0}) * &problem.description.track_costs).sum() / 2.0
            + self.n_trains as f64 * problem.description.train_price;
        
        self.n_trains == self.train_routes.len()
            && self.n_trains == self.train_types.len()
            && cost <= problem.description.total_budget
    }
}