use std::error::Error;
use std::path::PathBuf;

use plotters::prelude::*;

const NAIVE: [(f64, f64); 3] = [
    (100.0, 0.023_572_083 / 500.0),
    (400.0, 0.048_330_625 / 500.0),
    (1600.0, 0.441_588_833 / 500.0),
];
const CELLS: [(f64, f64); 3] = [
    (100.0, 0.025_951_500 / 500.0),
    (400.0, 0.050_668_541 / 500.0),
    (1600.0, 0.166_970_834 / 500.0),
];

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("scaling.png"));
    let root = BitMapBackend::new(&output, (1000, 680)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Force-search scaling", ("sans-serif", 36).into_font())
        .margin(30)
        .x_label_area_size(65)
        .y_label_area_size(90)
        .build_cartesian_2d(
            (80.0_f64..2000.0_f64).log_scale(),
            (3.0e-5_f64..2.0e-3_f64).log_scale(),
        )?;
    chart
        .configure_mesh()
        .x_desc("particles N (log scale)")
        .y_desc("wall seconds / production step (log scale)")
        .x_labels(6)
        .y_labels(8)
        .label_style(("sans-serif", 18))
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    for (data, color, label) in [
        (&NAIVE[..], RED, "naive all pairs"),
        (&CELLS[..], BLUE, "cell list"),
    ] {
        chart
            .draw_series(LineSeries::new(data.iter().copied(), color.stroke_width(3)))?
            .label(label)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 30, y)], color.stroke_width(3)));
        chart.draw_series(
            data.iter()
                .map(|&(x, y)| Circle::new((x, y), 6, color.filled())),
        )?;
    }
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .background_style(WHITE.mix(0.9))
        .border_style(BLACK)
        .label_font(("sans-serif", 19))
        .draw()?;
    root.present()?;
    println!("wrote {}", output.display());
    Ok(())
}
