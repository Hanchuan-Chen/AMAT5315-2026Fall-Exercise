use std::error::Error;
use std::path::Path;

use plotters::prelude::*;

use crate::AnalysisSummary;

const TC: f64 = 2.26919;

fn onsager(temperature: f64) -> f64 {
    if temperature >= TC {
        0.0
    } else {
        (1.0 - (2.0 / temperature).sinh().powi(-4)).powf(0.125)
    }
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
    chart.configure_mesh().x_desc("temperature T").y_desc("<|m|>").draw()?;
    chart
        .draw_series(LineSeries::new(
            (150..=350).map(|value| value as f64 / 100.0).map(|t| (t, onsager(t))),
            &RED,
        ))?
        .label("Onsager")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], RED));
    for (index, (&l, points)) in summary.by_size.iter().enumerate() {
        let color = Palette99::pick(index);
        chart
            .draw_series(LineSeries::new(
                points.iter().map(|point| (point.temperature, point.mean_abs_m)),
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
    chart.configure_mesh().x_desc("temperature T").y_desc("chi(T)").draw()?;
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
