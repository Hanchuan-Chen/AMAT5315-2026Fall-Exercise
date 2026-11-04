use infer::{bp, eval, exact::ExactResult, plan, uai::AppliedModel, uai::Model};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

fn asia() -> Model {
    Model::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("asia.uai")).unwrap()
}

fn exact_inputs(model: &Model, name: &str) -> (AppliedModel, plan::Plan) {
    let observations = model.asia_observations(name).unwrap();
    let applied = AppliedModel::with_selectors(model, &observations).unwrap();
    let labels: Vec<Vec<usize>> = applied.factors.iter().map(|f| f.vars.clone()).collect();
    let plan = plan::greedy(labels, Vec::new(), &applied.dims, name).unwrap();
    (applied, plan)
}

#[test]
fn asia_evidence_masses_and_marginals_match_the_sheet() {
    let model = asia();
    let expected = [("prior", 1.0), ("AD", 0.004501375), ("ADX", 0.00098822675)];
    for (name, mass) in expected {
        let (applied, plan) = exact_inputs(&model, name);
        let result =
            eval::exact(&plan, &applied, &(0..8).collect::<Vec<_>>(), Some((1, 3))).unwrap();
        assert!((result.evidence_mass - mass).abs() / mass < 1e-10);
        let enumeration = ExactResult::asia_enumeration(&applied);
        assert!(infer::exact::max_abs_error(&result.marginals, &enumeration.marginals) < 1e-10);
        assert!(
            result
                .marginals
                .values()
                .all(|v| (v.iter().sum::<f64>() - 1.0).abs() < 1e-12)
        );
        assert!((result.joints["T,L"].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    let (applied, plan) = exact_inputs(&model, "ADX");
    let result = eval::exact(&plan, &applied, &(0..8).collect::<Vec<_>>(), Some((1, 3))).unwrap();
    assert!((result.joints["T,L"][0] - 0.022214).abs() < 5e-6);
    let product = result.marginals[&1][0] * result.marginals[&3][0];
    assert!((product - 0.174026).abs() < 5e-6);
}

#[test]
fn random_and_greedy_plans_replay_and_recount_identically() {
    let model = asia();
    let observations = model.asia_observations("AD").unwrap();
    let applied = AppliedModel::with_selectors(&model, &observations).unwrap();
    let labels: Vec<Vec<usize>> = applied.factors.iter().map(|f| f.vars.clone()).collect();
    let input_labels = labels.clone();
    let random = plan::random(labels, Vec::new(), "AD", 42).unwrap();
    let first = eval::exact(&random, &applied, &[1], None).unwrap();
    let encoded = serde_json::to_string(&random).unwrap();
    let replay: plan::Plan = serde_json::from_str(&encoded).unwrap();
    replay.validate(&input_labels, &[]).unwrap();
    let second = eval::exact(&replay, &applied, &[1], None).unwrap();
    assert!((first.evidence_mass - second.evidence_mass).abs() < 1e-15);
    let a = random.complexity(&applied.dims).unwrap();
    let b = replay.complexity(&applied.dims).unwrap();
    assert_eq!(a, b);

    let greedy = plan::greedy(input_labels, Vec::new(), &applied.dims, "AD").unwrap();
    let work = greedy.complexity(&applied.dims).unwrap().work;
    let greedy_result = eval::exact(&greedy, &applied, &[1], None).unwrap();
    assert!((greedy_result.evidence_mass - 0.004501375).abs() / 0.004501375 < 1e-10);
    assert!(work >= 50);
}

#[test]
fn adx_loopy_bp_converges_but_stays_inaccurate() {
    let model = asia();
    let observations = model.asia_observations("ADX").unwrap();
    let exact_model = AppliedModel::with_selectors(&model, &observations).unwrap();
    let bp_model = AppliedModel::apply(&model, &observations).unwrap();
    let labels = exact_model.factors.iter().map(|f| f.vars.clone()).collect();
    let plan = plan::greedy(labels, Vec::new(), &exact_model.dims, "ADX").unwrap();
    let exact = eval::exact(&plan, &exact_model, &(0..8).collect::<Vec<_>>(), None).unwrap();
    let result = bp::run(&bp_model, 0.5, 1e-10, 1000, &exact.marginals).unwrap();
    assert!(result.converged);
    assert!(result.returned_undamped_residual < 1e-10);
    assert!((result.max_marginal_error - 0.01061).abs() < 2e-4);
    assert_eq!(
        result
            .sweeps
            .iter()
            .map(|record| record.sweep)
            .collect::<Vec<_>>(),
        (1..=result.sweeps_run).collect::<Vec<_>>()
    );
    assert_eq!(
        result.sweeps.last().unwrap().cumulative_update_work,
        result.cumulative_update_work
    );
}

#[test]
fn matrix_and_three_spin_checks_are_within_tolerance() {
    infer::checks::matrix_and_ising().unwrap();
}

#[test]
fn promedus_has_the_pinned_shape_and_evidence() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("promedus");
    let model = Model::load(&root.join("Promedus_14.uai")).unwrap();
    assert_eq!(model.vars, 414);
    assert_eq!(model.factors.len(), 414);
    let evidence = model
        .load_evidence(&root.join("Promedus_14.uai.evid"))
        .unwrap();
    assert_eq!(
        evidence,
        BTreeMap::from([
            (41, 1),
            (47, 1),
            (26, 1),
            (29, 1),
            (28, 1),
            (14, 1),
            (123, 1),
            (4, 1),
            (147, 1)
        ])
    );
}

#[test]
fn seeded_treesa_is_exactly_repeatable() {
    let model = asia();
    let observations = model.asia_observations("AD").unwrap();
    let applied = AppliedModel::with_selectors(&model, &observations).unwrap();
    let labels: Vec<Vec<usize>> = applied.factors.iter().map(|f| f.vars.clone()).collect();
    let first = plan::treesa(labels.clone(), Vec::new(), &applied.dims, "AD", 2026).unwrap();
    let second = plan::treesa(labels, Vec::new(), &applied.dims, "AD", 2026).unwrap();
    assert_eq!(first.tree, second.tree);
    assert_eq!(
        first.complexity(&applied.dims).unwrap(),
        second.complexity(&applied.dims).unwrap()
    );
}

#[test]
fn cli_rejects_seed_outside_random_or_treesa_and_exact_bp_flags() {
    let binary = env!("CARGO_BIN_EXE_infer");
    for args in [
        vec![
            "--model",
            "asia.uai",
            "--evidence",
            "AD",
            "--order",
            "greedy",
            "--seed",
            "1",
            "--out",
            "/tmp/unused",
        ],
        vec![
            "--model",
            "asia.uai",
            "--evidence",
            "AD",
            "--analyze",
            "/tmp/unused.json",
            "--seed",
            "1",
            "--out",
            "/tmp/unused",
        ],
        vec![
            "--model",
            "asia.uai",
            "--evidence",
            "AD",
            "--damping",
            "0.5",
            "--out",
            "/tmp/unused",
        ],
    ] {
        let output = Command::new(binary).args(args).output().unwrap();
        assert!(!output.status.success());
    }
}
