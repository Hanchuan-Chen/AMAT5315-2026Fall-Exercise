//! The Rust loop that chains the Enzyme timestep derivatives through time.

use crate::deriv::{self, Grid};
use crate::experiment::Experiment;
use crate::npy;
use crate::treeverse::{self, Action, Replay};
use serde_json::{json, Value};
use std::path::Path;

#[derive(Clone)]
pub struct State {
    pub prev: Vec<f64>,
    pub u: Vec<f64>,
}

impl State {
    fn zeros(n: usize) -> State {
        State {
            prev: vec![0.0; n],
            u: vec![0.0; n],
        }
    }
}

pub struct Forward {
    /// `[step][receiver]`, flattened by step.
    pub traces: Vec<f64>,
    /// Born tangent `[step][receiver]`, flattened by step.
    pub tangent: Vec<f64>,
    /// Complete states `s_0 .. s_steps` when `store` is set.
    pub states: Vec<State>,
    /// Recording frames `(step, field)`, first shot only.
    pub frames: Vec<(usize, Vec<f64>)>,
}

#[derive(Clone)]
pub struct ShotStats {
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub actions_file: Option<String>,
}

pub struct Stats {
    pub storage: String,
    pub checkpoints: Option<usize>,
    pub reverse_calls: usize,
    pub scheduler_forward_calls: usize,
    pub peak_saved_states: usize,
    pub peak_saved_bytes: usize,
    pub per_shot: Vec<ShotStats>,
}

impl Stats {
    pub fn to_json(&self) -> Value {
        let per_shot: Vec<Value> = self
            .per_shot
            .iter()
            .map(|s| {
                let mut object = serde_json::Map::new();
                object.insert("reverse_calls".into(), json!(s.reverse_calls));
                object.insert(
                    "scheduler_forward_calls".into(),
                    json!(s.scheduler_forward_calls),
                );
                object.insert("peak_saved_states".into(), json!(s.peak_saved_states));
                if let Some(name) = &s.actions_file {
                    object.insert("actions_file".into(), json!(name));
                }
                Value::Object(object)
            })
            .collect();
        json!({
            "storage": self.storage,
            "checkpoints": self.checkpoints,
            "reverse_calls": self.reverse_calls,
            "scheduler_forward_calls": self.scheduler_forward_calls,
            "peak_saved_states": self.peak_saved_states,
            "peak_saved_bytes": self.peak_saved_bytes,
            "per_shot": per_shot,
        })
    }
}

pub struct Migrate {
    pub image: Vec<f64>,
    pub norms: Vec<f64>,
    pub stats: Stats,
    pub frames: Vec<(usize, Vec<f32>)>,
    pub actions: Vec<Vec<Action>>,
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub struct Model {
    pub exp: Experiment,
    pub sigma: Vec<f64>,
    pub den: Vec<f64>,
    pub grid: Grid,
    pub footprints: Vec<Vec<f64>>,
    pub receiver_idx: Vec<usize>,
}

impl Model {
    pub fn new(exp: Experiment) -> Model {
        let n = exp.nx * exp.nz;
        let mut sigma = vec![0.0; n];
        let mut c_max: f64 = 0.0;
        for z in 0..exp.nz {
            for x in 0..exp.nx {
                let distance = [x, z, exp.nx - 1 - x, exp.nz - 1 - z]
                    .into_iter()
                    .min()
                    .unwrap() as f64;
                let ratio = ((exp.sponge_width - distance) / exp.sponge_width).max(0.0);
                sigma[z * exp.nx + x] = exp.sponge_strength * ratio * ratio;
                c_max = c_max.max(exp.background[z * exp.nx + x].abs());
            }
        }
        assert!(
            exp.dt * c_max / exp.dx * std::f64::consts::SQRT_2 < 1.0,
            "acoustic CFL bound violated"
        );
        let den: Vec<f64> = sigma.iter().map(|s| 1.0 + exp.dt * s).collect();
        let mut footprints = Vec::with_capacity(exp.shots.len());
        for shot in &exp.shots {
            let mut footprint = vec![0.0; n];
            for z in 0..exp.nz {
                for x in 0..exp.nx {
                    let dx = x as f64 - shot[0];
                    let dz = z as f64 - shot[1];
                    footprint[z * exp.nx + x] = (-(dx * dx + dz * dz) / 2.0).exp();
                }
            }
            footprints.push(footprint);
        }
        let receiver_idx: Vec<usize> = exp
            .receivers
            .iter()
            .map(|r| {
                assert!(r[0] >= 0 && r[0] < exp.nx as i64 && r[1] >= 0 && r[1] < exp.nz as i64);
                r[1] as usize * exp.nx + r[0] as usize
            })
            .collect();
        let invdx2 = 1.0 / (exp.dx * exp.dx);
        let grid = Grid {
            nx: exp.nx,
            nz: exp.nz,
            invdx2,
            dt: exp.dt,
        };
        Model {
            exp,
            sigma,
            den,
            grid,
            footprints,
            receiver_idx,
        }
    }

    pub fn n(&self) -> usize {
        self.grid.n()
    }

    pub fn state_bytes(&self) -> usize {
        2 * self.n() * 8
    }

    fn factor(&self, step: usize) -> f64 {
        self.exp.source_amplitude * self.exp.ricker(step)
    }

    fn advance(&self, shot: usize, step: usize, c: &[f64], state: &State) -> State {
        let mut out = State::zeros(self.n());
        deriv::primal(
            &self.grid,
            &state.prev,
            &state.u,
            c,
            &self.footprints[shot],
            &self.sigma,
            &self.den,
            self.factor(step),
            &mut out.u,
        );
        out.prev.copy_from_slice(&state.u);
        out
    }

    /// Forward run. With `direction` set, each step also carries one timestep
    /// JVP and records the receiver samples of its tangent (the Born data).
    pub fn forward(
        &self,
        shot: usize,
        c: &[f64],
        direction: Option<&[f64]>,
        store: bool,
        every: Option<usize>,
    ) -> Forward {
        let steps = self.exp.steps;
        let n = self.n();
        let nrec = self.receiver_idx.len();
        let mut state = State::zeros(n);
        let mut dstate = State::zeros(n);
        let mut out = State::zeros(n);
        let mut dout = vec![0.0; n];
        let mut traces = vec![0.0; steps * nrec];
        let mut tangent = vec![0.0; steps * nrec];
        let mut states = if store { vec![state.clone()] } else { Vec::new() };
        let mut frames = Vec::new();
        for step in 0..steps {
            let factor = self.factor(step);
            match direction {
                Some(direction) => {
                    deriv::forward(
                        &self.grid,
                        &state.prev,
                        &dstate.prev,
                        &state.u,
                        &dstate.u,
                        c,
                        direction,
                        &self.footprints[shot],
                        &self.sigma,
                        &self.den,
                        factor,
                        &mut out.u,
                        &mut dout,
                    );
                }
                None => {
                    deriv::primal(
                        &self.grid,
                        &state.prev,
                        &state.u,
                        c,
                        &self.footprints[shot],
                        &self.sigma,
                        &self.den,
                        factor,
                        &mut out.u,
                    );
                }
            }
            out.prev.copy_from_slice(&state.u);
            for (r, &idx) in self.receiver_idx.iter().enumerate() {
                traces[step * nrec + r] = out.u[idx];
                if direction.is_some() {
                    tangent[step * nrec + r] = dout[idx];
                }
            }
            if direction.is_some() {
                dstate.prev.copy_from_slice(&dstate.u);
                dstate.u.copy_from_slice(&dout);
            }
            std::mem::swap(&mut state, &mut out);
            if store {
                states.push(state.clone());
            }
            if let Some(every) = every {
                if (step + 1) % every == 0 {
                    frames.push((step + 1, state.u.clone()));
                }
            }
        }
        Forward {
            traces,
            tangent,
            states,
            frames,
        }
    }

    pub fn born(&self) -> Vec<Vec<f64>> {
        self.exp
            .shots
            .iter()
            .enumerate()
            .map(|(shot, _)| {
                self.forward(shot, &self.exp.background, Some(&self.exp.perturbation), false, None)
                    .tangent
            })
            .collect()
    }

    /// Reverse one timestep. `seed` is the adjoint of the new pressure `u^{k+1}`
    /// (receivers already injected) and is consumed; `state` is `s_k`.
    #[allow(clippy::too_many_arguments)]
    fn reverse_one(
        &self,
        shot: usize,
        step: usize,
        c: &[f64],
        state: &State,
        seed: &mut [f64],
        dprev: &mut [f64],
        du: &mut [f64],
        dc: &mut [f64],
        out: &mut [f64],
        ap: &[f64],
        au: &mut [f64],
        image: &mut [f64],
    ) {
        dprev.fill(0.0);
        du.fill(0.0);
        dc.fill(0.0);
        deriv::reverse(
            &self.grid,
            &state.prev,
            dprev,
            &state.u,
            du,
            c,
            dc,
            &self.footprints[shot],
            &self.sigma,
            &self.den,
            self.factor(step),
            out,
            seed,
        );
        for i in 0..self.n() {
            image[i] += dc[i];
            au[i] = ap[i] + du[i];
        }
    }

    /// Full-history reverse pass: keep every complete state.
    pub fn migrate_full(&self, weights: &[Vec<f64>], every: Option<usize>) -> Migrate {
        let steps = self.exp.steps;
        let n = self.n();
        let nrec = self.receiver_idx.len();
        let mut image = vec![0.0; n];
        let mut norms = Vec::with_capacity(self.exp.shots.len());
        let mut frames = Vec::new();
        let mut per_shot = Vec::new();
        for shot in 0..self.exp.shots.len() {
            let fwd = self.forward(shot, &self.exp.background, None, true, None);
            assert_eq!(fwd.states.len(), steps + 1);
            let mut ap = vec![0.0; n];
            let mut au = vec![0.0; n];
            let mut seed = vec![0.0; n];
            let mut dprev = vec![0.0; n];
            let mut du = vec![0.0; n];
            let mut dc = vec![0.0; n];
            let mut out = vec![0.0; n];
            let mut shot_image = vec![0.0; n];
            for step in (0..steps).rev() {
                let w = &weights[shot];
                for (r, &idx) in self.receiver_idx.iter().enumerate() {
                    au[idx] += w[step * nrec + r];
                }
                seed.copy_from_slice(&au);
                self.reverse_one(
                    shot,
                    step,
                    &self.exp.background,
                    &fwd.states[step],
                    &mut seed,
                    &mut dprev,
                    &mut du,
                    &mut dc,
                    &mut out,
                    &ap,
                    &mut au,
                    &mut shot_image,
                );
                ap.copy_from_slice(&dprev);
                if let Some(every) = every {
                    if shot == 0 && step % every == 0 {
                        frames.push((step, au.iter().map(|v| *v as f32).collect()));
                    }
                }
            }
            norms.push(l2(&shot_image));
            for (dst, value) in image.iter_mut().zip(&shot_image) {
                *dst += value;
            }
            per_shot.push(ShotStats {
                reverse_calls: steps,
                scheduler_forward_calls: steps,
                peak_saved_states: steps + 1,
                actions_file: None,
            });
        }
        let peak = steps + 1;
        let stats = Stats {
            storage: "full".into(),
            checkpoints: None,
            reverse_calls: steps * self.exp.shots.len(),
            scheduler_forward_calls: steps * self.exp.shots.len(),
            peak_saved_states: peak,
            peak_saved_bytes: peak * self.state_bytes(),
            per_shot,
        };
        Migrate {
            image,
            norms,
            stats,
            frames,
            actions: Vec::new(),
        }
    }

    /// Treeverse reverse pass: save, restore and replay within a checkpoint budget.
    pub fn migrate_treeverse(
        &self,
        weights: &[Vec<f64>],
        delta: usize,
        every: Option<usize>,
    ) -> Migrate {
        let steps = self.exp.steps;
        assert!(delta >= 1, "--checkpoints must be a positive integer");
        let mut image = vec![0.0; self.n()];
        let mut norms = Vec::with_capacity(self.exp.shots.len());
        let mut frames = Vec::new();
        let mut per_shot = Vec::new();
        let mut all_actions = Vec::new();
        for shot in 0..self.exp.shots.len() {
            let mut replay = TreeReplay {
                model: self,
                shot,
                weights: &weights[shot],
                image: vec![0.0; self.n()],
                ap: vec![0.0; self.n()],
                au: vec![0.0; self.n()],
                frames: Vec::new(),
                every: every.filter(|_| shot == 0),
                seed: vec![0.0; self.n()],
                dprev: vec![0.0; self.n()],
                du: vec![0.0; self.n()],
                dc: vec![0.0; self.n()],
                out: vec![0.0; self.n()],
            };
            let actions = treeverse::schedule(steps, delta, State::zeros(self.n()), &mut replay);
            norms.push(l2(&replay.image));
            for (i, value) in replay.image.iter().enumerate() {
                image[i] += value;
            }
            frames.extend(replay.frames.drain(..));
            let calls = actions.iter().filter(|a| a.action == "call").count();
            let peak = actions.iter().map(|a| a.saved_states).max().unwrap_or(1);
            per_shot.push(ShotStats {
                reverse_calls: steps,
                scheduler_forward_calls: calls,
                peak_saved_states: peak,
                actions_file: Some(format!("actions-{shot}.json")),
            });
            all_actions.push(actions);
        }
        let peak = per_shot.iter().map(|s| s.peak_saved_states).max().unwrap_or(1);
        let stats = Stats {
            storage: "treeverse".into(),
            checkpoints: Some(delta),
            reverse_calls: steps * self.exp.shots.len(),
            scheduler_forward_calls: per_shot.iter().map(|s| s.scheduler_forward_calls).sum(),
            peak_saved_states: peak,
            peak_saved_bytes: peak * self.state_bytes(),
            per_shot,
        };
        Migrate {
            image,
            norms,
            stats,
            frames,
            actions: all_actions,
        }
    }

    pub fn write_actions(&self, out_dir: &Path, actions: &[Vec<Action>]) {
        for (shot, log) in actions.iter().enumerate() {
            let entries: Vec<Value> = log
                .iter()
                .map(|a| json!({"action": a.action, "step": a.step, "saved_states": a.saved_states}))
                .collect();
            let path = out_dir.join(format!("actions-{shot}.json"));
            std::fs::write(&path, serde_json::to_string(&entries).unwrap() + "\n")
                .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
        }
    }

    pub fn write_wavefield(&self, path: &Path, frames: &[(usize, Vec<f32>)]) -> Vec<usize> {
        let n = self.n();
        let mut flat = Vec::with_capacity(frames.len() * n);
        for (_, field) in frames {
            flat.extend_from_slice(field);
        }
        npy::write_f32(path, &[frames.len(), self.exp.nz, self.exp.nx], &flat).unwrap();
        frames.iter().map(|(step, _)| *step).collect()
    }
}

struct TreeReplay<'a> {
    model: &'a Model,
    shot: usize,
    weights: &'a [f64],
    image: Vec<f64>,
    ap: Vec<f64>,
    au: Vec<f64>,
    frames: Vec<(usize, Vec<f32>)>,
    every: Option<usize>,
    seed: Vec<f64>,
    dprev: Vec<f64>,
    du: Vec<f64>,
    dc: Vec<f64>,
    out: Vec<f64>,
}

impl Replay for TreeReplay<'_> {
    type State = State;

    fn advance(&mut self, step: usize, state: &State) -> State {
        self.model.advance(self.shot, step, &self.model.exp.background, state)
    }

    fn reverse(&mut self, step: usize, state: &State) {
        let nrec = self.model.receiver_idx.len();
        for (r, &idx) in self.model.receiver_idx.iter().enumerate() {
            self.au[idx] += self.weights[step * nrec + r];
        }
        self.seed.copy_from_slice(&self.au);
        self.dprev.fill(0.0);
        self.du.fill(0.0);
        self.dc.fill(0.0);
        deriv::reverse(
            &self.model.grid,
            &state.prev,
            &mut self.dprev,
            &state.u,
            &mut self.du,
            &self.model.exp.background,
            &mut self.dc,
            &self.model.footprints[self.shot],
            &self.model.sigma,
            &self.model.den,
            self.model.factor(step),
            &mut self.out,
            &mut self.seed,
        );
        for i in 0..self.model.n() {
            self.image[i] += self.dc[i];
            self.au[i] = self.ap[i] + self.du[i];
        }
        self.ap.copy_from_slice(&self.dprev);
        if let Some(every) = self.every {
            if step % every == 0 {
                self.frames
                    .push((step, self.au.iter().map(|v| *v as f32).collect()));
            }
        }
    }
}
