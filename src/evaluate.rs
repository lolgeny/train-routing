//! Evaluates a solution by simulating flow on it

use itertools::Itertools;
use ndarray::ArrayD;
use ordered_float::NotNan;
use radix_heap::RadixHeapMap;

use crate::problem::{Problem, ScheduleType, TrainLine};
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
#[derive(Debug, Clone, Copy)]
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
    pub has_switched: bool,
    /// The total lines travelled so far - max 5
    pub total_lines: usize
}
impl PartialEq for QueueNode {
    fn eq(&self, other: &Self) -> bool {
        // Only the score matters when sorting the heap
        self.score == other.score
    }
}
impl Eq for QueueNode {}
impl PartialOrd for QueueNode {
    fn partial_cmp(&self, other: &Self)
        -> Option<std::cmp::Ordering> { Some(self.cmp(other)) }
}
impl Ord for QueueNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Uses `total_cmp`: the score will never be infinity or NaN
        // so the well ordering is valid
        other.score.total_cmp(&other.score)
    }
}

// Large constant penalty for disconnect between stations
const DEFAULT_TRAVEL_TIME: f64 = 1e10;

/// Evaluates a solution by simulating flow on it
/// 
/// For every station, paths to every other one required are computed via BFS.
pub fn evaluate(
    problem: &Problem,
    train_lines: &[TrainLine]
) -> f64 {
    // Create a list of E(X_i) where X_i is the time it takes to wait for train i to reach a commuter
    // This is half the total distance of a cycle over the number of trains on the line
    let train_delays = train_lines.iter().map(|line| {
        let mut total_time: f64 = (0..line.route.len()-1).map(|i| problem.track_times[[line.route[i], line.route[i+1]]]).sum();
        if line.ty == Circular { // Must also travel to beginning
            total_time += problem.track_times[[line.route[0], line.route[line.route.len()-1]]];
        }
        total_time / (2.0 * line.n as f64)
    }).collect_vec();

    let mut station_travel_times = ArrayD::<f64>::ones(problem.travel_frequencies.shape()) * DEFAULT_TRAVEL_TIME; // TODO: something more robust

    // Iterate over every starting position
    let mut queue = RadixHeapMap::new();
    for station in 0..problem.n {
        queue.clear();
        // An ordered list for efficient binary search
        // We only need to visit stations above this one,
        // since the time from previous stations to this one
        // has already been calculated
        let mut stations_unvisited = (station..problem.n).collect_vec();
        // Storing previous states
        let mut prev_states = vec![];

        // Start on any train line that goes through this station
        for (train, line) in train_lines.iter().enumerate().filter(|(_, l)| l.route.contains(&station)) {
            // UNWRAP: this will never panic: the current station, by use of `filter` above,
            // will always be in this train's route.
            let pos = line.route.iter().position(|x| *x == station).unwrap();
            // UNWRAPS: 0 is not nan
            queue.push(NotNan::new(0.0).unwrap(), QueueNode {station, train, score: 0.0, direction: Forward, train_schedule_progress: pos, has_switched: false, total_lines: 1});
            if line.ty == Bidirectional { // could be riding a bidirectional train backwards
                queue.push(NotNan::new(0.0).unwrap(), QueueNode {station, train, score: 0.0, direction: Backward, train_schedule_progress: pos, has_switched: false, total_lines: 1});
            }
        }

        // Algorithm loop, processing the current shortest node
        while let Some((_, n)) = queue.pop() {
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

            if n.total_lines >= 3 {break};

            // Check if we have already got from this station to targets - if so, visit it!
            stations_unvisited = stations_unvisited.into_iter().filter(|u| {
                if station_travel_times[[n.station, *u]] < DEFAULT_TRAVEL_TIME {
                    station_travel_times[[station, *u]] = n.score + station_travel_times[[n.station, *u]];
                    station_travel_times[[*u, station]] = n.score + station_travel_times[[n.station, *u]];
                    return false;
                }
                true
            }).collect();

            // A commuter could stay on the same train
            let next_station_pos = match n.direction {
                Forward => if n.train_schedule_progress + 1 < train_lines[n.train].route.len() {n.train_schedule_progress + 1} else {0},
                Backward => if n.train_schedule_progress > 0 {n.train_schedule_progress - 1} else {train_lines[n.train].route.len()-1}
            };
            let next_station = train_lines[n.train].route[next_station_pos];
            // only push this node if this station has not yet been visited
            if stations_unvisited.binary_search(&next_station).is_ok() {
                let score = n.score + problem.track_times[[n.station, next_station]];
                if let Ok(nnan) = NotNan::new(-score) {
                    queue.push(nnan, QueueNode {
                        station: next_station,
                        train: n.train,
                        score,
                        direction: n.direction,
                        train_schedule_progress: next_station_pos,
                        has_switched: false,
                        total_lines: n.total_lines
                    });
                }
            }

            // A commuter could also switch trains
            if n.has_switched {continue};
            let adjacent_trains = train_lines.iter().enumerate()
                .filter(
                    |(i, l)| *i != n.train && l.route.contains(&n.station) // ensure the train is different to this + visits this station
                );
            for (a_train, _) in adjacent_trains {
                // UNWRAP: again, by the filter above, this will never panic since `position` will always find this station.
                let pos = match train_lines[a_train].route.iter().position(|x| *x == n.station) {
                    Some(x) => x,
                    None => break // this will never happen
                };
                let score = n.score + train_delays[a_train];
                if let Ok(nnan) = NotNan::new(-score) {
                    queue.push(nnan, QueueNode {
                        station: n.station,
                        train: a_train,
                        score,
                        direction: Forward,
                        train_schedule_progress: pos,
                        has_switched: true,
                        total_lines: n.total_lines + 1
                    });
                }
                if train_lines[a_train].ty == Bidirectional { // riding backwards on a bidirectional train
                    let score = n.score + train_delays[a_train];
                    if let Ok(nnan) = NotNan::new(-score) {
                        queue.push(nnan, QueueNode {
                            station: n.station,
                            train: a_train,
                            score,
                            direction: Backward,
                            train_schedule_progress: pos,
                            has_switched: true,
                            total_lines: n.total_lines + 1
                        });
                    }
                }
            }
        }
    }
    // Elementwise multiply with frequencies to get an overall score
    (station_travel_times * &problem.travel_frequencies).sum() / 2.0
}