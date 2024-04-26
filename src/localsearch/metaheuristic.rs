//! Defines tabu search and simulated annealing metaheuristics

use std::collections::HashMap;

use crate::problem::TrainLine;

use super::{Metaheuristic, Solver, WorkingSolution};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct TabuParams {
    /// The time before tabu times out
    pub initial_timeout: usize,
    /// The amonut to adjust the tabu size by every loop
    pub size_adjust: usize
}

#[derive(Debug, Clone)]
pub struct TabuSearch {
    /// The tabu list
    tabu: HashMap<Vec<TrainLine>, usize>,
    /// The configuration of the tabu search
    params: TabuParams,
    /// The current timeout for expiry of tabu
    tabu_timeout: usize
}
impl Metaheuristic for TabuSearch {
    type Params = TabuParams;

    fn new(params: Self::Params) -> Self {
        Self {
            tabu: HashMap::new(),
            params,
            tabu_timeout: params.initial_timeout,
        }
    }

    fn choose_update(&mut self, candidates: Vec<WorkingSolution>, solver: &Solver<'_, Self>, prev_score: f64, time: usize) -> Option<(WorkingSolution, f64)> {
        self.tabu.retain(|_, v| *v + self.tabu_timeout >= time);
        if let Some((solution, score)) = candidates.into_iter().filter(|c| !self.tabu.contains_key(&c.train_lines)).map(|n| {
            let score = n.evaluate(solver);
            (n, score)
        })
            .min_by(|(_, score1), (_, score2)| score1.total_cmp(score2)) {
                if prev_score < score && self.tabu_timeout > self.params.size_adjust { // decrease tabu: selected neighbour is worse
                    self.tabu_timeout -= self.params.size_adjust;
                } else { // increase tabu: getting better
                    self.tabu_timeout += self.params.size_adjust;
                }
                self.tabu.insert(solution.train_lines.clone(), time);
        
                Some((solution, score))
        } else {
            self.tabu_timeout -= self.params.size_adjust;
            None
        }
    }
   
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimAnnealParams {
    pub initial_temp: f64,
    pub temp_scale: f64
}
#[derive(Debug, Clone)]
pub struct SimAnneal {
    temp: f64,
    params: SimAnnealParams
}
impl Metaheuristic for SimAnneal {
    type Params = SimAnnealParams;
    fn new(params: Self::Params) -> Self {
        Self { temp: params.initial_temp, params }
    }
    fn choose_update(&mut self, mut candidates: Vec<WorkingSolution>, solver: &Solver<'_, Self>, prev_score: f64, _time: usize) -> Option<(WorkingSolution, f64)> {
        self.temp *= self.params.temp_scale;
        while !candidates.is_empty() {
            let n = candidates.remove(fastrand::usize(0..candidates.len()));
            let score = n.evaluate(solver);
            if score < prev_score || fastrand::f64() < ((prev_score - score) / self.temp).exp() {return Some((n, score))};
        }
        None
    }
}