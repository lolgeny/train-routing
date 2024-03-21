//! Evaluates a solution by simulating flow on it

use std::collections::BinaryHeap;

use itertools::Itertools;
use ndarray::ArrayD;

use crate::problem::{Problem, ProblemDescription, ScheduleType};
use ScheduleType::*;


/// The direction a simulated train is currently travelling in
#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
enum TravelDirection {
    Forward, Backward
}
use TravelDirection::*;

/// A node used in the priority queue for Dijkstra's algorithm.
/// It has special comparison operations defined for the `BinaryHeap`
/// to use correctly.
#[derive(Debug)]
struct QueueNode {
    /// The station that this node is at
    pub station: usize,
    /// The train line that this node is riding
    pub train: usize,
    /// The current distance travelled by the train
    pub score: f64,
    /// The direction the train this node is on is moving in
    pub direction: TravelDirection,
    /// The position of the current station in the train's schedule
    // Used for efficiency
    pub train_schedule_progress: usize,
    /// Tracks if the node has just switched,
    /// to avoid an infinte loop of switching tracks
    pub has_switched: bool
}
impl PartialEq for QueueNode {
    fn eq(&self, other: &Self) -> bool {
        // Only the score matters when sorting the heap
        self.score == other.score
    }
}
impl Eq for QueueNode {}
impl PartialOrd for QueueNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for QueueNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Uses `total_cmp`: the score will never be infinity or NaN
        // so the well ordering is valid
        other.score.total_cmp(&other.score)
    }
}

/// Evaluates a solution by simulating flow on it
/// 
/// For every station, paths to every other one required are computed via BFS.
pub fn evaluate(
    problem: &Problem,
    train_routes: &[Vec<usize>],
    train_types: &[ScheduleType]
) -> f64 {
    // Create a list of E(X_i) where X_i is the time it takes to wait for train i to reach a commuter
    // This is half the total distance of a cycle
    let train_delays = train_routes.iter().enumerate().map(|(t, route)| {
        let mut total_time: f64 = (0..route.len()-1).map(|i| problem.description.track_times[[route[i], route[i+1]]]).sum();
        if train_types[t] == Circular { // Must also travel to beginning
            total_time += problem.description.track_times[[route[0], route[route.len()-1]]];
        }
        total_time / 2.0
    }).collect_vec();

    let mut station_travel_times = ArrayD::<f64>::ones(problem.description.travel_frequencies.shape()) * 1e10; // TODO: something more robust

    // Iterate over every starting position
    for station in 0..problem.description.n {
        let mut queue = BinaryHeap::new();

        // An ordered list for efficient binary search
        // We only need to visit stations above this one,
        // since the time from previous stations to this one
        // has already been calculated
        let mut stations_unvisited = (station..problem.description.n).collect_vec();
        // Storing previous states
        let mut prev_states = vec![];

        // Start on any train line that goes through this station
        for (train, _) in train_routes.iter().enumerate().filter(|(_, r)| r.contains(&station)) {
            // UNWRAP: this will never panic: the current station, by use of `filter` above,
            // will always be in this train's route.
            let pos = train_routes[train].iter().position(|x| *x == station).unwrap();
            queue.push(QueueNode {station, train, score: 0.0, direction: Forward, train_schedule_progress: pos, has_switched: false});
            if train_types[train] == Bidirectional { // could be riding a bidirectional train backwards
                queue.push(QueueNode {station, train, score: 0.0, direction: Backward, train_schedule_progress: pos, has_switched: false});
            }
        }

        // Algorithm loop, processing the current shortest node
        while let Some(n) = queue.pop() {
            if stations_unvisited.is_empty() {break};
            if let Ok(i) = stations_unvisited.binary_search(&n.station) {
                station_travel_times[[station, n.station]] = n.score;
                station_travel_times[[n.station, station]] = n.score;
                stations_unvisited.remove(i);
            }

            match prev_states.binary_search(&(n.station, n.train, n.direction)) {
                Ok(_) => continue,
                Err(i) => prev_states.insert(i, (n.station, n.train, n.direction))
            }

            // A commuter could stay on the same train
            let next_station_pos = match n.direction {
                Forward => if n.train_schedule_progress + 1 < train_routes[n.train].len() {n.train_schedule_progress + 1} else {0},
                Backward => if n.train_schedule_progress > 0 {n.train_schedule_progress - 1} else {train_routes[n.train].len()-1}
            };
            let next_station = train_routes[n.train][next_station_pos];
            // only push this node if this station has not yet been visited
            if stations_unvisited.binary_search(&next_station).is_ok() {
                queue.push(QueueNode {
                    station: next_station,
                    train: n.train,
                    score: n.score + problem.description.track_times[[n.station, next_station]],
                    direction: n.direction,
                    train_schedule_progress: next_station_pos,
                    has_switched: false
                });
            }

            // A commuter could also switch trains
            if n.has_switched {continue};
            let adjacent_trains = train_routes.iter().enumerate()
                .filter(
                    |(i, r)| *i != n.train && r.contains(&n.station) // ensure the train is different to this + visits this station
                );
            for (a_train, _) in adjacent_trains {
                // UNWRAP: again, by the filter above, this will never panic since `position` will always find this station.
                let pos = train_routes[a_train].iter().position(|x| *x == n.station).unwrap();
                queue.push(QueueNode {
                    station: n.station,
                    train: a_train,
                    score: n.score + train_delays[a_train],
                    direction: Forward,
                    train_schedule_progress: pos,
                    has_switched: true
                });
                if train_types[a_train] == Bidirectional { // riding backwards on a bidirectional train
                    queue.push(QueueNode {
                        station: n.station,
                        train: a_train,
                        score: n.score + train_delays[a_train],
                        direction: Backward,
                        train_schedule_progress: pos,
                        has_switched: true
                    });
                }
            }
        }
    }
    // Elementwise multiply with frequencies to get an overall score
    (dbg!(station_travel_times) * &problem.description.travel_frequencies).sum()
}