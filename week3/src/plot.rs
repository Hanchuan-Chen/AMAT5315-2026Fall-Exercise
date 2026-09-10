use std::error::Error;
use std::path::Path;

use plotters::prelude::*;

use crate::AnalysisSummary;
use crate::integrated_autocorrelation_time;

const TC: f64 = 2.26919;

fn onsager(temperature: f64) -> f64 {
    if temperature >= TC {
        0.0
    } else {
        (1.0 - (2.0 / temperature).sinh().powi(-4)).powf(0.125)
    }
}

pub fn write_tau_plot(summary: &AnalysisSummary, directory: &Path) -> Result<(), Box<dyn Error>> {
    let path = directory.join("tau.png");
    let root = BitMapBackend::new(&path, (960, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let curves: Vec<_> = summary
        .by_size
        .iter()
        .map(|(&l, points)| {
            (
                l,
                points
                    .iter()
                    .map(|point| {
                        (
                            point.temperature,
                            integrated_autocorrelation_time(
                                &point
                                    .magnetizations
                                    .iter()
                                    .map(|value| value.abs())
                                    .collect::<Vec<_>>(),
                            ),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let maximum = curves
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(_, tau)| *tau))
        .fold(1.0_f64, f64::max);
    let t_min = curves
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(temperature, _)| *temperature))
        .fold(f64::INFINITY, f64::min);
    let t_max = curves
        .iter()
        .flat_map(|(_, points)| points.iter().map(|(temperature, _)| *temperature))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Integrated autocorrelation time", ("sans-serif", 32))
        .margin(24)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(
            (t_min - 0.05)..(t_max + 0.05),
            (0.4..maximum * 1.2).log_scale(),
        )?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("tau_int (sweeps)")
        .draw()?;
    for (index, (l, points)) in curves.iter().enumerate() {
        let color = Palette99::pick(index);
        chart
            .draw_series(LineSeries::new(points.iter().copied(), &color))?
            .label(format!("L={l}"))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], Palette99::pick(index)));
    }
    chart.draw_series([PathElement::new(
        [(TC, 0.4), (TC, maximum * 1.2)],
        BLACK.mix(0.5),
    )])?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

pub fn write_tau_comparison(
    metropolis: &AnalysisSummary,
    wolff: &AnalysisSummary,
    output: &Path,
) -> Result<usize, Box<dyn Error>> {
    let l = metropolis
        .by_size
        .keys()
        .rev()
        .find(|size| wolff.by_size.contains_key(size))
        .copied()
        .ok_or("the runs share no lattice size")?;
    let make_curve = |summary: &AnalysisSummary| {
        summary.by_size[&l]
            .iter()
            .map(|point| {
                let absolute: Vec<_> = point
                    .magnetizations
                    .iter()
                    .map(|value| value.abs())
                    .collect();
                (
                    point.temperature,
                    integrated_autocorrelation_time(&absolute),
                )
            })
            .collect::<Vec<_>>()
    };
    let metropolis_curve = make_curve(metropolis);
    let wolff_temperatures: std::collections::BTreeSet<_> = wolff.by_size[&l]
        .iter()
        .map(|point| (point.temperature * 1_000_000.0).round() as i64)
        .collect();
    let metropolis_curve: Vec<_> = metropolis_curve
        .into_iter()
        .filter(|(temperature, _)| {
            let key = (*temperature * 1_000_000.0).round() as i64;
            wolff_temperatures.contains(&key)
        })
        .collect();
    let wolff_curve = make_curve(wolff);
    let maximum = metropolis_curve
        .iter()
        .chain(&wolff_curve)
        .map(|(_, tau)| *tau)
        .fold(1.0_f64, f64::max);
    let t_min = wolff_curve.first().ok_or("Wolff run has no points")?.0;
    let t_max = wolff_curve.last().ok_or("Wolff run has no points")?.0;
    let root = BitMapBackend::new(output, (960, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Update-rule autocorrelation at L={l}"),
            ("sans-serif", 32),
        )
        .margin(24)
        .x_label_area_size(45)
        .y_label_area_size(65)
        .build_cartesian_2d(
            (t_min - 0.02)..(t_max + 0.02),
            (0.4..maximum * 1.2).log_scale(),
        )?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("tau_int (sweeps)")
        .draw()?;
    chart
        .draw_series(LineSeries::new(metropolis_curve, &RED))?
        .label("single-flip Metropolis")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED));
    chart
        .draw_series(LineSeries::new(wolff_curve, &BLUE))?
        .label("Wolff clusters")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], BLUE));
    chart.draw_series([PathElement::new(
        [(TC, 0.4), (TC, maximum * 1.2)],
        BLACK.mix(0.5),
    )])?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(l)
}

pub fn write_thermodynamic_plots(
    summary: &AnalysisSummary,
    directory: &Path,
) -> Result<(), Box<dyn Error>> {
    let magnetization_path = directory.join("magnetization.png");
    let root = BitMapBackend::new(&magnetization_path, (960, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Mean absolute magnetization", ("sans-serif", 32))
        .margin(24)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(1.45..3.55, 0.0..1.02)?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("<|m|>")
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            (150..=350)
                .map(|value| value as f64 / 100.0)
                .map(|t| (t, onsager(t))),
            &RED,
        ))?
        .label("Onsager")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED));
    for (index, (&l, points)) in summary.by_size.iter().enumerate() {
        let color = Palette99::pick(index);
        chart
            .draw_series(LineSeries::new(
                points
                    .iter()
                    .map(|point| (point.temperature, point.mean_abs_m)),
                &color,
            ))?
            .label(format!("L={l}"))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], Palette99::pick(index)));
    }
    chart.draw_series([PathElement::new([(TC, 0.0), (TC, 1.0)], BLACK.mix(0.5))])?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;

    let susceptibility_path = directory.join("susceptibility.png");
    let root = BitMapBackend::new(&susceptibility_path, (960, 640)).into_drawing_area();
    root.fill(&WHITE)?;
    let maximum = summary
        .by_size
        .values()
        .flatten()
        .map(|point| point.chi)
        .fold(1.0_f64, f64::max);
    let mut chart = ChartBuilder::on(&root)
        .caption("Magnetic susceptibility", ("sans-serif", 32))
        .margin(24)
        .x_label_area_size(45)
        .y_label_area_size(55)
        .build_cartesian_2d(1.9..2.8, 0.0..maximum * 1.1)?;
    chart
        .configure_mesh()
        .x_desc("temperature T")
        .y_desc("chi(T)")
        .draw()?;
    for (index, (&l, points)) in summary.by_size.iter().enumerate() {
        let color = Palette99::pick(index);
        chart
            .draw_series(LineSeries::new(
                points.iter().map(|point| (point.temperature, point.chi)),
                &color,
            ))?
            .label(format!("L={l}"))
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], Palette99::pick(index)));
    }
    chart.draw_series([PathElement::new([(TC, 0.0), (TC, maximum)], BLACK.mix(0.5))])?;
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}
