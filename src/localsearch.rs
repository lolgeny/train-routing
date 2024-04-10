//! Implements a local search based algorithm for optimising a train routine.
//! It uses the ant colony optimisation meaheuristic

use std::{collections::VecDeque, vec};

use itertools::Itertools;
use ndarray::ArrayD;

use crate::{evaluate::evaluate, problem::{Problem, ScheduleType, Solution}};

/// A possible partial solution that is currently being considered
/// This is the bare bones description used for the tabu
#[derive(Debug, Clone, PartialEq)]
struct WorkingSolutionDescription {
    train_routes: Vec<Vec<usize>>,
    train_types: Vec<ScheduleType>
}
impl Default for WorkingSolutionDescription {
    /// An empty, basic feasible solution
    fn default() -> Self {
        Self { train_routes: vec![], train_types: vec![] }
    }
}

/// A possible partial solution that is currently being considered
#[derive(Debug, Clone, PartialEq)]
struct WorkingSolution {
    description: WorkingSolutionDescription,
    cost: f64,
    built_tracks: ArrayD<bool>,
}
impl WorkingSolution {
    /// An empty, basic feasible solution
    fn new(problem: &Problem) -> Self {
        Self { description: Default::default(), cost: 0.0, built_tracks: ArrayD::from_elem(problem.description.track_costs.shape(), false) }
    }
}   
impl WorkingSolution {
    /// Helper function to evaluate cost
    fn evaluate(&self, solver: &Solver<'_>) -> f64 {
        evaluate(solver.problem, &self.description.train_routes, &self.description.train_types)
    }
    /// Explore neighbours to this solution, by possible allowed moves
    fn generate_neighbours(&self, solver: &Solver<'_>) -> Vec<WorkingSolution> {
        let mut neighbours = vec![];
        
        // Clone a train
        for i in 0..self.description.train_routes.len() {
            if fastrand::f64() > solver.neighbour_chance {continue};
            let mut cloned_trains = self.description.train_routes.clone();
            cloned_trains.push(self.description.train_routes[i].clone());
            let mut cloned_train_types = self.description.train_types.clone();
            cloned_train_types.push(self.description.train_types[i]);
            neighbours.push(Self {
                description: WorkingSolutionDescription {train_routes: cloned_trains, train_types: cloned_train_types},
                cost: self.cost, built_tracks: self.built_tracks.clone()
            });
        }

        // Remove a train
        for i in 0..self.description.train_routes.len() {
            if fastrand::f64() > solver.neighbour_chance {continue};
            let mut cloned_trains = self.description.train_routes.clone();
            cloned_trains.swap_remove(i);
            let mut cloned_train_types = self.description.train_types.clone();
            cloned_train_types.swap_remove(i);
            neighbours.push(Self {
                description: WorkingSolutionDescription {train_routes: cloned_trains, train_types: cloned_train_types},
                cost: self.cost, built_tracks: self.built_tracks.clone()
            });
        }

        // Add a stop to a train
        for i in 0..self.description.train_routes.len() {
            let available_stations = (0..solver.problem.description.n).filter(|x| !self.description.train_routes[i].contains(x)).collect_vec();
            for s in available_stations {
                if fastrand::f64() > solver.neighbour_chance {continue};
                let mut cloned_trains = self.description.train_routes.clone();
                let index = fastrand::usize(0..=cloned_trains[i].len()); // the place to add the stop
                cloned_trains[i].insert(index, s);
                let mut cloned_built_tracks = self.built_tracks.clone();
                if index > 0 && !cloned_built_tracks[[cloned_trains[i-1], cloned_trains[i]]] {

                }
                neighbours.push(Self {
                    description: WorkingSolutionDescription {train_routes: cloned_trains, train_types: self.description.train_types.clone()},
                    cost: self.cost, built_tracks: self.built_tracks.clone()
                });
            }
        }

        // Remove a stop from a train
        for i in 0..self.train_routes.len() {
            for j in 0..self.train_routes[i].len() {
                if fastrand::f64() > solver.neighbour_chance {continue};
                let mut cloned_trains = self.train_routes.clone();
                cloned_trains[i].remove(j);
                neighbours.push(Self {train_routes: cloned_trains, train_types: self.train_types.clone()});
            }
        }
        
        neighbours
    }
}

/// Parameters for the solver.
/// Varying these will change the quality and speed
/// of the solution.
#[derive(Debug, Clone)]
pub struct Solver<'a> {
    /// The actual train problem to solve
    pub problem: &'a Problem,
    /// The max iterations to run the algorithm for,
    /// provided it does not converge beforehand
    pub max_iterations: usize,
    /// The probability a neighbour is constructed
    pub neighbour_chance: f64,
    /// The size of the tabu - larger may lead to higher quality solutions
    /// but use up more memory
    pub tabu_size: usize
}
impl<'a> Solver<'a> {
    /// Solve the problem
    pub fn solve(&self) -> Solution {
        // Construct a basic feasible solution
        let mut solution = WorkingSolution::default();
        let mut best_solution = solution.clone();
        let mut best_score = solution.evaluate(self);
        let mut tabu = VecDeque::with_capacity(self.tabu_size);
        for _ in 0..self.max_iterations {
            // Consider possible neighbours to this solution
            let neighbours = solution.generate_neighbours(self);
            let allowed_neighbours = neighbours.into_iter().filter(|n| !tabu.contains(n)).collect_vec();
            if allowed_neighbours.len() == 0 {continue}; // neighbour_chance is likely too low, or tabu too full
            // UNWRAP: above statement ensures this never panics
            let (i, neighbour, score) = allowed_neighbours.into_iter().enumerate().map(|(i, n)| (i, n, n.evaluate(self)))
                .min_by(|(_, _, score1), (_, _, score2)| score1.total_cmp(score2)).unwrap();
            // Update current solution + tabu
            solution = neighbour;
            if score < best_score {
                best_solution = solution.clone();
            }
            if tabu.len() >= self.tabu_size {tabu.pop_front();}
            tabu.push_back(solution.clone());
        }
        Solution {
            n_trains: best_solution.train_routes.len(), built_tracks: (), train_routes: (), train_types: (), obj_value: () 
        }
    }
}