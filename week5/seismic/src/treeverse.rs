//! Treeverse checkpoint schedule.
//!
//! Ported from `src/treeverse.jl` of GiggleLiu/TreeverseAlgorithm.jl (MIT),
//! commit 72f22fe5fa19a7a103d8140a68634101983d2088, following the endpoint rule
//! the course reference uses: the split is `ceil((d*s + t*p)/(t+d))`, clamped
//! to `max(s+1, p-1)` when it reaches the end, and each call finishes by
//! reversing its first step from that step's saved input state.

use std::collections::HashMap;

/// Replay hooks the scheduler needs from the simulation.
pub trait Replay {
    type State: Clone;
    /// Advance the working state from `s_k` to `s_{k+1}`.
    fn advance(&mut self, step: usize, state: &Self::State) -> Self::State;
    /// One reverse step at `step`, reading `s_step`.
    fn reverse(&mut self, step: usize, state: &Self::State);
}

#[derive(Clone, Copy, Debug)]
pub struct Action {
    pub action: &'static str,
    pub step: usize,
    pub saved_states: usize,
}

/// Binomial capacity `C(delta + tau, delta)` as an exact integer.
pub fn capacity(delta: usize, tau: usize) -> u128 {
    let mut result: u128 = 1;
    for i in 1..=delta {
        result = result * (tau + i) as u128 / i as u128;
    }
    result
}

/// Smallest positive repetition number whose capacity covers `steps`.
pub fn repetition_number(steps: usize, delta: usize) -> usize {
    let mut tau = 1;
    while capacity(delta, tau) < steps as u128 {
        tau += 1;
    }
    tau
}

fn middle(d: i64, t: i64, start: usize, end: usize) -> usize {
    let numerator = d * start as i64 + t * end as i64 + t + d - 1;
    let denominator = t + d;
    let mut k = numerator.div_euclid(denominator) as usize;
    if k >= end && d > 0 {
        k = std::cmp::max(start + 1, end.saturating_sub(1));
    }
    k
}

/// Run the schedule. `initial` is `s_0`; returns the logged actions.
pub fn schedule<R: Replay>(
    steps: usize,
    delta: usize,
    initial: R::State,
    replay: &mut R,
) -> Vec<Action> {
    assert!(delta >= 1, "treeverse needs at least one checkpoint slot");
    let tau = repetition_number(steps, delta) as i64;
    let mut saved: HashMap<usize, R::State> = HashMap::new();
    saved.insert(0, initial);
    let mut actions = Vec::new();
    if steps > 0 {
        visit(
            delta as i64,
            tau,
            0,
            0,
            steps,
            &mut saved,
            &mut actions,
            replay,
        );
    }
    debug_assert_eq!(saved.len(), 1);
    debug_assert!(saved.contains_key(&0));
    actions
}

#[allow(clippy::too_many_arguments)]
fn visit<R: Replay>(
    mut d: i64,
    mut t: i64,
    base: usize,
    start: usize,
    mut end: usize,
    saved: &mut HashMap<usize, R::State>,
    actions: &mut Vec<Action>,
    replay: &mut R,
) {
    if start > base {
        d -= 1;
        let mut state = saved[&base].clone();
        log(actions, saved, "restore", base);
        for step in base..start {
            state = replay.advance(step, &state);
            log(actions, saved, "call", step);
        }
        saved.insert(start, state);
        log(actions, saved, "store", start);
    }
    let mut split = middle(d, t, start, end);
    while t > 0 && split < end {
        visit(d, t, start, split, end, saved, actions, replay);
        t -= 1;
        end = split;
        split = middle(d, t, start, end);
    }
    replay.reverse(start, &saved[&start]);
    log(actions, saved, "grad", start);
    if start > base {
        saved.remove(&start);
        log(actions, saved, "fetch", start);
    }
}

fn log(actions: &mut Vec<Action>, saved: &HashMap<usize, impl Sized>, action: &'static str, step: usize) {
    actions.push(Action {
        action,
        step,
        saved_states: saved.len(),
    });
}
