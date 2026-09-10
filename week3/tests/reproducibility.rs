use ising::{RelaxConfig, relax};

fn config(seed: u64) -> RelaxConfig {
    RelaxConfig {
        l: 16,
        temperature: 2.3,
        equilibration_sweeps: 50,
        measurement_sweeps: 50,
        seed,
    }
}

#[test]
fn same_seed_reproduces_the_complete_rendered_run() {
    let first = relax(config(2026)).render();
    let second = relax(config(2026)).render();
    assert_eq!(first, second);
}

#[test]
fn different_seed_changes_the_run() {
    let first = relax(config(2026)).render();
    let second = relax(config(2027)).render();
    assert_ne!(first, second);
}
