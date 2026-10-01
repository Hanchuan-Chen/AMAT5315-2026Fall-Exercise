//! Treeverse schedule checks, including the sheet's worked example and the
//! reference forward-step counts for the reflector run.

use seismic::treeverse::{capacity, repetition_number, schedule, Action, Replay};

struct Dummy {
    calls: Vec<usize>,
    grads: Vec<usize>,
}

impl Replay for Dummy {
    type State = usize;
    fn advance(&mut self, step: usize, _state: &usize) -> usize {
        self.calls.push(step);
        step + 1
    }
    fn reverse(&mut self, step: usize, _state: &usize) {
        self.grads.push(step);
    }
}

fn run(steps: usize, delta: usize) -> (Vec<Action>, Dummy) {
    let mut dummy = Dummy {
        calls: Vec::new(),
        grads: Vec::new(),
    };
    let actions = schedule(steps, delta, 0, &mut dummy);
    (actions, dummy)
}

/// The checker's `validate_actions`, reduced to its replay invariants.
fn audit(actions: &[Action], steps: usize, delta: usize) -> (usize, usize) {
    let mut available = vec![0usize];
    let mut working = None;
    let mut reverse = Vec::new();
    let mut peak = 1usize;
    let mut calls = 0usize;
    for event in actions {
        match event.action {
            "restore" => {
                assert!(available.contains(&event.step));
                working = Some(event.step);
            }
            "call" => {
                assert_eq!(working, Some(event.step));
                working = Some(event.step + 1);
                calls += 1;
            }
            "store" => {
                assert_eq!(working, Some(event.step));
                assert!(!available.contains(&event.step));
                available.push(event.step);
                peak = peak.max(available.len());
                assert!(available.len() <= delta + 1);
            }
            "grad" => {
                assert!(available.contains(&event.step));
                reverse.push(event.step);
            }
            "fetch" => {
                assert!(available.contains(&event.step) && event.step != 0);
                available.retain(|s| *s != event.step);
            }
            other => panic!("unknown action {other}"),
        }
        assert_eq!(event.saved_states, available.len(), "stored-state counter");
    }
    assert_eq!(reverse, (0..steps).rev().collect::<Vec<_>>());
    assert_eq!(available, vec![0]);
    (calls, peak)
}

#[test]
fn worked_example_six_steps_two_slots() {
    assert_eq!(capacity(2, 2), 6);
    assert_eq!(repetition_number(6, 2), 2);
    let (actions, dummy) = run(6, 2);
    assert_eq!(dummy.calls.len(), 8, "eight forward advances");
    assert_eq!(dummy.grads, vec![5, 4, 3, 2, 1, 0]);
    assert_eq!(actions.iter().map(|a| a.saved_states).max(), Some(3));
    let (calls, peak) = audit(&actions, 6, 2);
    assert_eq!(calls, 8);
    assert_eq!(peak, 3);
}

#[test]
fn reflector_reference_work_and_peaks() {
    for (delta, expected) in [(1usize, 28680usize), (3, 1695), (5, 990), (10, 642)] {
        let (actions, dummy) = run(240, delta);
        let (calls, peak) = audit(&actions, 240, delta);
        assert_eq!(calls, expected, "delta {delta}");
        assert_eq!(dummy.calls.len(), expected);
        assert_eq!(peak, delta + 1, "delta {delta}");
        assert_eq!(dummy.grads, (0..240).rev().collect::<Vec<_>>());
    }
}
