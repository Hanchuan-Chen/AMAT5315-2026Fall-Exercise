use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Factor {
    pub vars: Vec<usize>,
    pub dims: Vec<usize>,
    /// UAI/table order: the last scope variable varies fastest.
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Model {
    pub vars: usize,
    pub dims: Vec<usize>,
    pub factors: Vec<Factor>,
}

fn tokens(text: &str) -> Vec<&str> {
    text.lines()
        .map(|line| line.split('#').next().unwrap_or(""))
        .flat_map(str::split_whitespace)
        .collect()
}

impl Model {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let tok = tokens(&text);
        let mut at = 0;
        let next = |at: &mut usize| -> Result<&str, String> {
            let value = tok
                .get(*at)
                .ok_or_else(|| "unexpected end of UAI file".to_string())?;
            *at += 1;
            Ok(value)
        };
        if next(&mut at)? != "MARKOV" {
            return Err("only MARKOV UAI files are supported".into());
        }
        let vars: usize = next(&mut at)?.parse().map_err(|_| "bad variable count")?;
        let mut dims = Vec::with_capacity(vars);
        for _ in 0..vars {
            dims.push(
                next(&mut at)?
                    .parse()
                    .map_err(|_| "bad variable dimension")?,
            );
        }
        let count: usize = next(&mut at)?.parse().map_err(|_| "bad factor count")?;
        let mut scopes = Vec::with_capacity(count);
        for _ in 0..count {
            let rank: usize = next(&mut at)?.parse().map_err(|_| "bad factor rank")?;
            let mut scope = Vec::with_capacity(rank);
            for _ in 0..rank {
                scope.push(next(&mut at)?.parse().map_err(|_| "bad factor variable")?);
            }
            scopes.push(scope);
        }
        let mut factors = Vec::with_capacity(count);
        for scope in scopes {
            let entries: usize = next(&mut at)?.parse().map_err(|_| "bad table size")?;
            let factor_dims: Vec<usize> = scope.iter().map(|v| dims[*v]).collect();
            let expected: usize = factor_dims.iter().product();
            if entries != expected {
                return Err(format!(
                    "factor {scope:?} says {entries} entries, expected {expected}"
                ));
            }
            let mut values = Vec::with_capacity(entries);
            for _ in 0..entries {
                values.push(
                    next(&mut at)?
                        .parse()
                        .map_err(|_| "bad factor table entry")?,
                );
            }
            factors.push(Factor {
                vars: scope,
                dims: factor_dims,
                values,
            });
        }
        if at != tok.len() {
            return Err(format!("{} trailing UAI tokens", tok.len() - at));
        }
        Ok(Self {
            vars,
            dims,
            factors,
        })
    }

    pub fn variable_names(&self) -> Vec<String> {
        if self.vars == 8 {
            "A T S L B E X D"
                .split_whitespace()
                .map(str::to_string)
                .collect()
        } else {
            (0..self.vars).map(|v| v.to_string()).collect()
        }
    }

    pub fn asia_observations(&self, name: &str) -> Result<BTreeMap<usize, usize>, String> {
        if self.vars != 8 {
            return match name.to_ascii_lowercase().as_str() {
                "prior" | "none" => Ok(BTreeMap::new()),
                _ => Err("use --evidence-file for a non-ASIA model".into()),
            };
        }
        let mut obs = BTreeMap::new();
        match name.to_ascii_uppercase().as_str() {
            "PRIOR" => {}
            "AD" => {
                obs.insert(0, 0);
                obs.insert(7, 0);
            }
            "ADX" => {
                obs.insert(0, 0);
                obs.insert(7, 0);
                obs.insert(6, 0);
            }
            other => return Err(format!("unknown ASIA evidence case {other}")),
        }
        Ok(obs)
    }

    pub fn load_evidence(&self, path: &Path) -> Result<BTreeMap<usize, usize>, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let tok = tokens(&text);
        let count: usize = tok
            .first()
            .ok_or("empty evidence file")?
            .parse()
            .map_err(|_| "bad evidence count")?;
        if tok.len() != 1 + 2 * count {
            return Err("evidence file token count does not match its header".into());
        }
        let mut obs = BTreeMap::new();
        for pair in tok[1..].chunks_exact(2) {
            let var: usize = pair[0].parse().map_err(|_| "bad evidence variable")?;
            let state: usize = pair[1].parse().map_err(|_| "bad evidence state")?;
            if var >= self.vars || state >= self.dims[var] {
                return Err(format!("invalid evidence {var}={state}"));
            }
            obs.insert(var, state);
        }
        Ok(obs)
    }
}

/// A factor after observations have selected a column of its original table.
/// Axes keep their original UAI scope order.
#[derive(Clone, Debug)]
pub struct AppliedFactor {
    pub vars: Vec<usize>,
    pub dims: Vec<usize>,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct AppliedModel {
    pub vars: usize,
    pub dims: Vec<usize>,
    pub observations: BTreeMap<usize, usize>,
    pub factors: Vec<AppliedFactor>,
    pub scalar: f64,
}

fn strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

impl AppliedModel {
    /// Exact inference keeps every original table unchanged and represents each
    /// observation as a one-hot selector factor.
    pub fn with_selectors(
        model: &Model,
        observations: &BTreeMap<usize, usize>,
    ) -> Result<Self, String> {
        for (&var, &state) in observations {
            if var >= model.vars || state >= model.dims[var] {
                return Err(format!("invalid observation {var}={state}"));
            }
        }
        let mut factors: Vec<AppliedFactor> = model
            .factors
            .iter()
            .map(|factor| AppliedFactor {
                vars: factor.vars.clone(),
                dims: factor.dims.clone(),
                values: factor.values.clone(),
            })
            .collect();
        for (&var, &state) in observations {
            let mut values = vec![0.0; model.dims[var]];
            values[state] = 1.0;
            factors.push(AppliedFactor {
                vars: vec![var],
                dims: vec![model.dims[var]],
                values,
            });
        }
        Ok(Self {
            vars: model.vars,
            dims: model.dims.clone(),
            observations: observations.clone(),
            factors,
            scalar: 1.0,
        })
    }

    pub fn apply(model: &Model, observations: &BTreeMap<usize, usize>) -> Result<Self, String> {
        for (&var, &state) in observations {
            if var >= model.vars || state >= model.dims[var] {
                return Err(format!("invalid observation {var}={state}"));
            }
        }
        let mut scalar = 1.0;
        let mut factors = Vec::new();
        for factor in &model.factors {
            let original_strides = strides(&factor.dims);
            let keep: Vec<usize> = (0..factor.vars.len())
                .filter(|&axis| !observations.contains_key(&factor.vars[axis]))
                .collect();
            let vars: Vec<usize> = keep.iter().map(|&a| factor.vars[a]).collect();
            let dims: Vec<usize> = keep.iter().map(|&a| factor.dims[a]).collect();
            let out_strides = strides(&dims);
            let mut values = vec![0.0; dims.iter().product()];
            for (index, &value) in factor.values.iter().enumerate() {
                let mut rest = index;
                let mut retained = 0usize;
                let mut allowed = true;
                for (axis, &stride) in original_strides.iter().enumerate() {
                    let state = rest / stride;
                    rest %= stride;
                    if let Some(&observed) = observations.get(&factor.vars[axis]) {
                        allowed &= state == observed;
                    } else {
                        let out_axis = keep.iter().position(|&a| a == axis).unwrap();
                        retained += state * out_strides[out_axis];
                    }
                }
                if allowed {
                    values[retained] = value;
                }
            }
            if vars.is_empty() {
                scalar *= values[0];
            } else {
                factors.push(AppliedFactor { vars, dims, values });
            }
        }
        Ok(Self {
            vars: model.vars,
            dims: model.dims.clone(),
            observations: observations.clone(),
            factors,
            scalar,
        })
    }

    pub fn factor_value(&self, factor: &AppliedFactor, assignment: &[usize]) -> f64 {
        let st = strides(&factor.dims);
        let index: usize = factor
            .vars
            .iter()
            .zip(&st)
            .map(|(&var, &stride)| assignment[var] * stride)
            .sum();
        factor.values[index]
    }
}
