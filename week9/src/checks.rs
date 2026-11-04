use serde::Serialize;
use tenferro_cpu::CpuEngine;
use tenferro_einsum::TensorEinsumExt;
use tenferro_tensor::Tensor;

#[derive(Clone, Debug, Serialize)]
pub struct TensorChecks {
    pub a: [[f64; 2]; 2],
    pub b: [[f64; 2]; 2],
    pub tensor_ab: [[f64; 2]; 2],
    pub tensor_trace: f64,
    pub tensor_ising_z: f64,
    pub rust_direct_ab: [[f64; 2]; 2],
    pub rust_direct_trace: f64,
    pub rust_enumeration_ising_z: f64,
    pub max_abs_error: f64,
}

pub fn matrix_and_ising() -> Result<TensorChecks, String> {
    let a = Tensor::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])
        .map_err(|e| e.to_string())?;
    let b = Tensor::from_vec_col_major(vec![2, 2], vec![2.0, 1.0, 0.0, 2.0])
        .map_err(|e| e.to_string())?;
    let mut engine = CpuEngine::serial();
    let ab = [&a, &b]
        .einsum("ij,jk->ik", &mut engine)
        .map_err(|e| e.to_string())?;
    let trace = [&ab]
        .einsum("ii->", &mut engine)
        .map_err(|e| e.to_string())?;
    let abv = ab.as_slice::<f64>().map_err(|e| e.to_string())?;
    let tv = trace.as_slice::<f64>().map_err(|e| e.to_string())?;

    let direct_ab = [
        [1.0 * 2.0 + 2.0 * 1.0, 1.0 * 0.0 + 2.0 * 2.0],
        [3.0 * 2.0 + 4.0 * 1.0, 3.0 * 0.0 + 4.0 * 2.0],
    ];
    let direct_trace = direct_ab[0][0] + direct_ab[1][1];
    let direct_col = [
        direct_ab[0][0],
        direct_ab[1][0],
        direct_ab[0][1],
        direct_ab[1][1],
    ];
    let ab_error = abv
        .iter()
        .zip(direct_col)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);

    let beta: f64 = 0.5;
    let j: f64 = 1.0;
    let bond = Tensor::from_vec_col_major(
        vec![2, 2],
        [1.0, -1.0, -1.0, 1.0]
            .into_iter()
            .map(|s| (beta * j * s).exp())
            .collect(),
    )
    .map_err(|e| e.to_string())?;
    let first = [&bond, &bond]
        .einsum("ij,jk->ik", &mut engine)
        .map_err(|e| e.to_string())?;
    let ring = [&first, &bond]
        .einsum("ik,ki->", &mut engine)
        .map_err(|e| e.to_string())?;
    let zv = ring.as_slice::<f64>().map_err(|e| e.to_string())?;
    let first_v = first.as_slice::<f64>().map_err(|e| e.to_string())?;
    let bond_v = bond.as_slice::<f64>().map_err(|e| e.to_string())?;
    let tensor_trace: f64 = (0..2)
        .flat_map(|i| (0..2).map(move |k| (i, k)))
        .map(|(i, k)| first_v[i + 2 * k] * bond_v[k + 2 * i])
        .sum();
    let direct_z = (0..8)
        .map(|bits| {
            let spins = [
                if bits & 1 == 0 { -1.0 } else { 1.0 },
                if bits & 2 == 0 { -1.0 } else { 1.0 },
                if bits & 4 == 0 { -1.0 } else { 1.0 },
            ];
            (beta * j * (spins[0] * spins[1] + spins[1] * spins[2] + spins[2] * spins[0])).exp()
        })
        .sum::<f64>();
    let z_error = (zv[0] - direct_z).abs();
    if ab_error >= 1e-12 || (tv[0] - direct_trace).abs() >= 1e-12 || z_error >= 1e-12 {
        eprintln!(
            "check failure AB={abv:?} trace={} direct_trace={} Z={} tensor_trace={} direct_Z={} errors=({ab_error:e}, {:e}, {z_error:e})",
            tv[0],
            direct_trace,
            zv[0],
            tensor_trace,
            direct_z,
            (tv[0] - direct_trace).abs()
        );
        return Err("matrix/Ising tensor checks exceed 1e-12".into());
    }
    let tensor_ab = [[abv[0], abv[2]], [abv[1], abv[3]]];
    let result = TensorChecks {
        a: [[1.0, 2.0], [3.0, 4.0]],
        b: [[2.0, 0.0], [1.0, 2.0]],
        tensor_ab,
        tensor_trace: tv[0],
        tensor_ising_z: zv[0],
        rust_direct_ab: direct_ab,
        rust_direct_trace: direct_trace,
        rust_enumeration_ising_z: direct_z,
        max_abs_error: ab_error.max((tv[0] - direct_trace).abs()).max(z_error),
    };
    println!(
        "checks AB=[[{:.1},{:.1}],[{:.1},{:.1}]] tr(AB)={:.1} Z={:.15} max_abs_error={:.3e}",
        abv[0],
        abv[2],
        abv[1],
        abv[3],
        tv[0],
        zv[0],
        ab_error.max((tv[0] - direct_trace).abs()).max(z_error)
    );
    Ok(result)
}
