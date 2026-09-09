use std::env;
use std::error::Error;

use md::physics::{energy, force};
use plotters::prelude::*;

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 800;
const LIMIT: f64 = 3.0;

fn energy_color(value: f64) -> RGBColor {
    let value = value.clamp(-1.0, 1.0);
    if value < 0.0 {
        interpolate(RGBColor(59, 76, 192), WHITE, value + 1.0)
    } else {
        interpolate(WHITE, RGBColor(180, 4, 38), value)
    }
}

fn interpolate(start: RGBColor, end: RGBColor, fraction: f64) -> RGBColor {
    // Work with signed differences so decreasing channels do not underflow.
    let channel =
        |a: u8, b: u8| (f64::from(a) + fraction * (f64::from(b) - f64::from(a))).round() as u8;
    RGBColor(
        channel(start.0, end.0),
        channel(start.1, end.1),
        channel(start.2, end.2),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let output = env::args()
        .nth(1)
        .unwrap_or_else(|| "../field.png".to_owned());
    let root = BitMapBackend::new(&output, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;
    let (plot_area, legend_area) = root.split_horizontally(850);

    let mut chart = ChartBuilder::on(&plot_area)
        .caption(
            "Lennard-Jones pair energy and radial force",
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(-LIMIT..LIMIT, -LIMIT..LIMIT)?;

    let cells = 180;
    let spacing = 2.0 * LIMIT / f64::from(cells);
    chart.draw_series((0..cells).flat_map(|ix| {
        (0..cells).map(move |iy| {
            let x0 = -LIMIT + f64::from(ix) * spacing;
            let y0 = -LIMIT + f64::from(iy) * spacing;
            let r = ((x0 + spacing / 2.0).powi(2) + (y0 + spacing / 2.0).powi(2)).sqrt();
            let value = if r < 0.65 { 1.0 } else { energy(r).min(1.0) };
            Rectangle::new(
                [(x0, y0), (x0 + spacing, y0 + spacing)],
                energy_color(value).filled(),
            )
        })
    }))?;

    chart
        .configure_mesh()
        .x_desc("x separation")
        .y_desc("y separation")
        .axis_style(BLACK.mix(0.75))
        .light_line_style(WHITE.mix(0.35))
        .draw()?;

    let r0 = 2.0_f64.powf(1.0 / 6.0);
    let dashed_circle = (0..72).filter(|index| index % 2 == 0).map(|index| {
        let a0 = f64::from(index) * std::f64::consts::TAU / 72.0;
        let a1 = f64::from(index + 1) * std::f64::consts::TAU / 72.0;
        PathElement::new(
            vec![
                (r0 * a0.cos(), r0 * a0.sin()),
                (r0 * a1.cos(), r0 * a1.sin()),
            ],
            BLACK.mix(0.8).stroke_width(2),
        )
    });
    chart.draw_series(dashed_circle)?;

    for ix in -6_i32..=6 {
        for iy in -6_i32..=6 {
            let x = f64::from(ix) * 0.42;
            let y = f64::from(iy) * 0.42;
            let r = x.hypot(y);
            if !(0.58..=2.9).contains(&r) {
                continue;
            }

            let radial_force = force(r);
            let length = 0.12 + 0.18 * radial_force.abs().tanh();
            let direction = radial_force.signum();
            let ux = direction * x / r;
            let uy = direction * y / r;
            let end = (x + length * ux, y + length * uy);
            let head = 0.055;
            let back = (end.0 - head * ux, end.1 - head * uy);
            let perp = (-uy * head * 0.7, ux * head * 0.7);
            let style = BLACK.mix(0.72).stroke_width(1);

            chart.draw_series([
                PathElement::new(vec![(x, y), end], style),
                PathElement::new(vec![end, (back.0 + perp.0, back.1 + perp.1)], style),
                PathElement::new(vec![end, (back.0 - perp.0, back.1 - perp.1)], style),
            ])?;
        }
    }

    chart.draw_series(std::iter::once(Circle::new((0.0, 0.0), 7, BLACK.filled())))?;
    chart.draw_series(std::iter::once(Text::new(
        "r0",
        (r0 + 0.08, 0.12),
        ("sans-serif", 18).into_font().color(&BLACK),
    )))?;

    legend_area.fill(&WHITE)?;
    legend_area.draw(&Text::new(
        "U(r)",
        (36, 100),
        ("sans-serif", 22).into_font().color(&BLACK),
    ))?;
    for index in 0..200 {
        let value = 1.0 - 2.0 * f64::from(index) / 199.0;
        let y0 = 140 + index * 2;
        legend_area.draw(&Rectangle::new(
            [(35, y0), (75, y0 + 2)],
            energy_color(value).filled(),
        ))?;
    }
    for (label, y) in [("+1", 140), ("0", 340), ("-1", 540)] {
        legend_area.draw(&Text::new(
            label,
            (85, y + 7),
            ("sans-serif", 18).into_font().color(&BLACK),
        ))?;
    }
    legend_area.draw(&Text::new(
        "capped",
        (28, 570),
        ("sans-serif", 16).into_font().color(&BLACK),
    ))?;
    legend_area.draw(&Text::new(
        "arrows: force",
        (15, 640),
        ("sans-serif", 16).into_font().color(&BLACK),
    ))?;
    legend_area.draw(&Text::new(
        "dashed: r0",
        (15, 670),
        ("sans-serif", 16).into_font().color(&BLACK),
    ))?;

    root.present()?;
    println!("wrote {output}");
    Ok(())
}
