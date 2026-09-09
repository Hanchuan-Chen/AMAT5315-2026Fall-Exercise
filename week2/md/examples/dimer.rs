use std::env;
use std::error::Error;

use md::simulation::{Euler, VelocityVerlet, run_dimer};
use plotters::coord::types::RangedCoordf64;
use plotters::prelude::*;

type Chart<'a> = ChartContext<'a, BitMapBackend<'a>, Cartesian2d<RangedCoordf64, RangedCoordf64>>;

fn draw_curve(
    chart: &mut Chart<'_>,
    times: &[f64],
    errors: &[f64],
    scale: f64,
    color: RGBColor,
    label: &str,
) -> Result<(), Box<dyn Error>> {
    chart
        .draw_series(LineSeries::new(
            times
                .iter()
                .copied()
                .zip(errors.iter().map(|error| scale * error)),
            color.stroke_width(2),
        ))?
        .label(label)
        .legend(move |(x, y)| PathElement::new([(x, y), (x + 28, y)], color.stroke_width(2)));
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = env::args()
        .nth(1)
        .unwrap_or_else(|| "../dimer.png".to_owned());
    let euler = run_dimer(&Euler, 500, 0.01);
    let verlet = run_dimer(&VelocityVerlet, 500, 0.01);
    let verlet_long = run_dimer(&VelocityVerlet, 5000, 0.01);

    let root = BitMapBackend::new(&output, (1200, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    let mut left = ChartBuilder::on(&panels[0])
        .caption("First 500 steps", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..5.0, -0.1..2.2)?;
    left.configure_mesh()
        .x_desc("time t")
        .y_desc("(E(t) - E0) / |E0|")
        .draw()?;
    draw_curve(
        &mut left,
        euler.times(),
        euler.relative_errors(),
        1.0,
        RGBColor(190, 45, 45),
        "forward Euler",
    )?;
    draw_curve(
        &mut left,
        verlet.times(),
        verlet.relative_errors(),
        1.0,
        RGBColor(35, 90, 175),
        "velocity-Verlet",
    )?;
    left.configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    let long_scaled: Vec<f64> = verlet_long
        .relative_errors()
        .iter()
        .map(|error| 1000.0 * error)
        .collect();
    let limit = long_scaled
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(0.5)
        * 1.1;
    let mut right = ChartBuilder::on(&panels[1])
        .caption("Velocity-Verlet: 5000 steps", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(0.0..50.0, -limit..limit)?;
    right
        .configure_mesh()
        .x_desc("time t")
        .y_desc("relative energy error x 1000")
        .draw()?;
    draw_curve(
        &mut right,
        verlet_long.times(),
        verlet_long.relative_errors(),
        1000.0,
        RGBColor(35, 90, 175),
        "velocity-Verlet",
    )?;
    right
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    println!("wrote {output}");
    Ok(())
}
