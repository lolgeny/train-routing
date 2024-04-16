//! Implements a local search based algorithm for optimising a train routine.
//! It uses the ant colony optimisation meaheuristic

use std::{collections::HashMap, vec};

use itertools::Itertools;
use ndarray::ArrayD;

use crate::{baseline, evaluate::evaluate, problem::{Problem, ScheduleType, Solution, TrainLine}};

/// Helper iterator to visit all tracks on a single train line
struct TrainTrackIterator<'a> {
    train_line: &'a TrainLine,
    i: usize
}
impl<'a> TrainTrackIterator<'a> {
    /// Create a new track iterator for a specific line
    pub fn new(train_line: &'a TrainLine) -> Self {
        Self { train_line, i: 0 }
    }
}
impl<'a> Iterator for TrainTrackIterator<'a> {
    type Item = (usize, usize); // represents two stations

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.train_line.route.len() - 1 + match self.train_line.ty {
            ScheduleType::Bidirectional => 0,
            ScheduleType::Circular => 1
        } {
            return None;
        }
        self.i += 1;
        if self.train_line.ty == ScheduleType::Circular && self.i == self.train_line.route.len() {
            // UNWRAP: a train line will always have a station
            return Some((self.train_line.route[0], *self.train_line.route.last().unwrap()));
        }
        Some((self.train_line.route[self.i-1], self.train_line.route[self.i]))
    }
}

/// A possible partial solution that is currently being considered
#[derive(Debug, Clone, PartialEq)]
struct WorkingSolution {
    train_lines: Vec<TrainLine>,
    cost: f64,
    built_tracks: ArrayD<bool>,
}
impl WorkingSolution {
    /// An empty, basic feasible solution
    fn new(problem: &Problem) -> Self {
        // Self { train_lines: vec![TrainLine::default()], cost: 0.0, built_tracks: ArrayD::from_elem(problem.track_costs.shape(), false) }
        let base = baseline::big_loop(problem, ScheduleType::Bidirectional);
        let cost = base.cost(problem);
        Self {
            train_lines: base.train_lines,
            cost,
            built_tracks: base.built_tracks
        }
    }
}   
impl WorkingSolution {
    /// Helper function to evaluate objective
    fn evaluate(&self, solver: &Solver<'_>) -> f64 {
        evaluate(solver.problem, &self.train_lines)
    }
    /// Helper funcction to check cost
    fn calc_cost(&self, solver: &Solver<'_>) -> f64 {
        let mut cost = self.train_lines.iter().map(|l| l.n as f64).sum::<f64>() * solver.problem.train_price;
        for i in 0..solver.problem.n {
            'tracks: for j in 0..solver.problem.n {
                for l in &self.train_lines {
                    for (a, b) in TrainTrackIterator::new(l) {
                        if (i == a && j == b) || (i == b && j == a) {
                            cost += solver.problem.track_costs[[i, j]];
                            continue 'tracks;
                        }
                    }
                }
            }
        }
        cost
    }
    /// Explore neighbours to this solution, by possible allowed moves
    fn generate_neighbours(&self, solver: &Solver<'_>) -> Vec<WorkingSolution> {
        let mut neighbours = vec![];
        
        // Clone a line
        for i in 0..self.train_lines.len() {
            if fastrand::f64() > solver.neighbour_chance {continue};
            let mut cloned_lines = self.train_lines.clone();
            cloned_lines.push(self.train_lines[i].clone());
            neighbours.push(Self {
                // The only new cost is building additional trains, since tracks are already built
                cost: self.cost + self.train_lines[i].n as f64 * solver.problem.train_price,
                built_tracks: self.built_tracks.clone(),
                train_lines: cloned_lines,
            });
        }

        // Remove a line
        if self.train_lines.len() > 1 {
            for i in 0..self.train_lines.len() {
                if fastrand::f64() > solver.neighbour_chance {continue};
                let mut cloned_lines = self.train_lines.clone();
                let removed_line = cloned_lines.swap_remove(i);
                let mut cloned_build_tracks = self.built_tracks.clone();
                let mut cost_saved = removed_line.n as f64 * solver.problem.train_price;
                // Iterate through all tracks and find if any are unnecessary now
                for i in 0..solver.problem.n {
                    'tracks: for j in 0..solver.problem.n {
                        if !cloned_build_tracks[[i, j]] {continue};
                        for l in &cloned_lines {
                            for (a, b) in TrainTrackIterator::new(l) {
                                if (a == i && b == j) || (a == j && b == i) {continue 'tracks;} // another line is using this track
                            }
                        }
                        // If the code reaches here, the track is no longer necessary
                        cost_saved += solver.problem.track_costs[[i, j]];
                        cloned_build_tracks[[i, j]] = false;
                        cloned_build_tracks[[j, i]] = false;
                    }
                }
                neighbours.push(Self {
                    // Remove the cost of the line and tracks which are no longer needed
                    cost: self.cost - cost_saved,
                    built_tracks: self.built_tracks.clone(),
                    train_lines: cloned_lines,
                });
            }
        }

        // Add a stop to a line
        for i in 0..self.train_lines.len() {
            let available_stations = (0..solver.problem.n).filter(|x| !self.train_lines[i].route.contains(x)).collect_vec();
            for s in available_stations {
                if fastrand::f64() > solver.neighbour_chance {continue};
                let mut cloned_lines = self.train_lines.clone();
                let index = fastrand::usize(0..=cloned_lines[i].route.len()); // the place to add the stop
                cloned_lines[i].route.insert(index, s);
                let mut cloned_built_tracks = self.built_tracks.clone();
                let mut additional_cost = 0.0;
                // Check if we need to build new tracks
                let mut connections = Vec::with_capacity(2);
                if index > 0 && !cloned_built_tracks[[index, index-1]] {
                    connections.push((index, index-1));
                }
                if index < cloned_lines[i].route.len()-1 && !cloned_built_tracks[[index, index+1]] {
                    connections.push((index, index+1))
                }
                if (index == 0 || index == cloned_lines[i].route.len()-1) && cloned_lines[i].ty == ScheduleType::Circular {
                    connections.push((0, cloned_lines[i].route.len()-1));
                }
                for (ai, bi) in connections {
                    let a = cloned_lines[i].route[ai];
                    let b = cloned_lines[i].route[bi];
                    if cloned_built_tracks[[a, b]] {continue};
                    cloned_built_tracks[[a, b]] = true;
                    cloned_built_tracks[[b, a]] = true;
                    additional_cost += solver.problem.track_costs[[a, b]];
                }
                neighbours.push(Self {
                    cost: additional_cost, built_tracks: cloned_built_tracks,
                    train_lines: cloned_lines,
                });
            }
        }

        // Remove a stop from a line
        // A line must cover at least two stations
        for i in 0..self.train_lines.len() {
            if self.train_lines[i].route.len() < 3 {continue};
            for index in 0..self.train_lines[i].route.len() {
                if fastrand::f64() > solver.neighbour_chance {continue};
                let mut cloned_lines = self.train_lines.clone();
                let mut cloned_built_tracks = self.built_tracks.clone();
                // Check if any tracks are now unnecessary
                let mut connections = Vec::with_capacity(2);
                if index > 0 && !cloned_built_tracks[[index, index-1]] {
                    connections.push((index, index-1));
                }
                if index < cloned_lines[i].route.len()-1 && !cloned_built_tracks[[index, index+1]] {
                    connections.push((index, index+1))
                }
                if (index == 0 || index == cloned_lines[i].route.len()-1) && cloned_lines[i].ty == ScheduleType::Circular {
                    connections.push((0, cloned_lines[i].route.len()-1));
                }
                cloned_lines[i].route.remove(index);
                let mut cost_saved = 0.0;
                'connections: for (ai, bi) in connections {
                    let a = self.train_lines[i].route[ai];
                    let b = self.train_lines[i].route[bi];
                    for l in &cloned_lines {
                        for (c, d) in TrainTrackIterator::new(l) {
                            if (c == a && d == b) || (c == b && d == a) {continue 'connections}; // this track is needed
                        }
                    }
                    cloned_built_tracks[[a, b]] = false;
                    cloned_built_tracks[[b, a]] = false;
                    cost_saved += solver.problem.track_costs[[a, b]];
                }
                neighbours.push(Self {train_lines: cloned_lines, cost: self.cost-cost_saved, built_tracks: cloned_built_tracks});
            }
        }
        
        // Increase/decrease number of trains on a line
        for i in 0..self.train_lines.len() {
            if fastrand::f64() > solver.neighbour_chance {continue};
            let mut cloned_lines1 = self.train_lines.clone();
            cloned_lines1[i].n += 1;
            neighbours.push(Self { train_lines: cloned_lines1, cost: self.cost + solver.problem.train_price, built_tracks: self.built_tracks.clone() });
            if self.train_lines[i].n > 1 { // only subtract if the line is still running - don't leave a ghost line
                let mut cloned_lines2 = self.train_lines.clone();
                cloned_lines2[i].n -= 1;
                neighbours.push(Self { train_lines: cloned_lines2, cost: self.cost - solver.problem.train_price, built_tracks: self.built_tracks.clone() });
            }
        }

        // Change the type of a line
        for i in 0..self.train_lines.len() {
            if fastrand::f64() > solver.neighbour_chance {continue};
            let mut cloned_lines = self.train_lines.clone();
            let mut cloned_built_tracks = self.built_tracks.clone();
            let mut cost_change = 0.0;
            let a = cloned_lines[i].route[0];
            let b = *cloned_lines[i].route.last().unwrap();
            match cloned_lines[i].ty {
                ScheduleType::Bidirectional => {
                    cloned_lines[i].ty = ScheduleType::Circular;
                    if !cloned_built_tracks[[a, b]] {
                        cloned_built_tracks[[a, b]] = true;
                        cloned_built_tracks[[b, a]] = true;
                        cost_change = solver.problem.track_costs[[a, b]];
                    }
                }
                ScheduleType::Circular => {
                    cloned_lines[i].ty = ScheduleType::Bidirectional;
                    // Find if this track is still necessary
                    let mut found = false;
                    'search: for l in &cloned_lines {
                        for (c, d) in TrainTrackIterator::new(l) {
                            if (c == a && d == b) || (c == b && d == a) {found = true; break 'search;}
                        }
                    }
                    if !found {
                        cloned_built_tracks[[a, b]] = false;
                        cloned_built_tracks[[b, a]] = false;
                        cost_change = -solver.problem.track_costs[[a, b]];
                    }
                }
            }
            neighbours.push(Self { train_lines: cloned_lines, cost: self.cost + cost_change, built_tracks: cloned_built_tracks });
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
    /// The time before tabu times out
    pub tabu_initial_timeout: usize
}
impl<'a> Solver<'a> {
    /// Solve the problem
    pub fn solve(&self) -> Solution {
        // Construct a basic feasible solution
        let mut solution = WorkingSolution::new(self.problem);
        let mut best_solution = solution.clone();
        let mut best_score = solution.evaluate(self);
        let mut tabu = HashMap::new();
        let mut tabu_timeout = self.tabu_initial_timeout;
        let mut current_score = best_score;
        let mut time = 0;
        let mut stale_time = 0;
        let mut good_solutions: Vec<WorkingSolution> = vec![];
        for _ in 0..self.max_iterations {
            // Consider possible neighbours to this solution
            let neighbours = solution.generate_neighbours(self);
            let allowed_neighbours = neighbours.into_iter().filter(
                |n| !tabu.contains_key(&n.train_lines) && n.calc_cost(&self) <= self.problem.total_budget
            ).collect_vec();
            if allowed_neighbours.is_empty() {continue}; // neighbour_chance is likely too low, or tabu too full
            // UNWRAP: above statement ensures this never panics
            let (_, neighbour, score) = allowed_neighbours.into_iter().enumerate().map(|(i, n)| {
                let score = n.evaluate(self);
                (i, n, score)
            })
                .min_by(|(_, _, score1), (_, _, score2)| score1.total_cmp(score2)).unwrap();
            // Update current solution + tabu
            solution = neighbour;
            if score < best_score {
                best_solution = solution.clone();
                best_score = score;
            }
            if current_score <= score { // decrease tabu: selected neighbour is worse
                if tabu_timeout > 10 && current_score < score {tabu_timeout -= 10;}
                stale_time += 1;
            } else { // increase tabu: getting better
                tabu_timeout += 10;
                stale_time = 0;
            }
            current_score = score;
            if stale_time > 20 && !good_solutions.is_empty() { // intensification
                solution = fastrand::choice(&good_solutions).unwrap().clone(); // UNWRAP: never unwraps since we've checked good solutions
                current_score = solution.evaluate(&self);
                stale_time = 0;
            }
            if time % 100 == 0 && !good_solutions.contains(&best_solution) {
                good_solutions.push(best_solution.clone());
            }
            tabu.retain(|_, v| *v + tabu_timeout >= time);
            tabu.insert(solution.train_lines.clone(), time);
            time += 1;
        }
        Solution { built_tracks: best_solution.built_tracks, train_lines: best_solution.train_lines, obj_value: best_score }
    }
}