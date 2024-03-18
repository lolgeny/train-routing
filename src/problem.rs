//! This module contains interfaces for the solver: it has the `Problem` struct, which describes a train routing problem,
//! and the `Solution` struct, which is what the solver returns and represents the optimal solution

use ndarray::{Array, ArrayD};

/// A description of a general train route problem
struct Problem {
    /// The number of stations
    pub n: usize,
    /// A symmetric matrix representing the cost to build tracks between two stations
    pub track_costs: ArrayD<f64>,
    /// A symmetric matrix representing the time to travel between two stations, if a track is built
    pub track_times: ArrayD<f64>,
    /// The price per train
    pub T: f64,
    /// The total budget
    pub B: f64
}

/// Represents which type of line a train follows:
/// 
/// - `Circular` means it goes to the first station after the last one
/// 
/// - `Bidirectional` means it repeats the track, reversed
enum ScheduleType {
    Circular, Bidirectional
}

/// The solver's optimal solution to the problem
struct Solution {
    /// The number of trains/lines to build
    pub n_trains: usize,
    /// A symmetric matrix showing which tracks are built
    pub built_tracks: ArrayD<bool>,
    /// A list of routes for trains to follow
    pub train_routes: Vec<Vec<usize>>,
    /// The objective value, representing how good the solution is,
    /// where lower is better
    pub obj_value: f64
}