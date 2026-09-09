use std::fs;
use std::path::Path;
use std::process::Command;

use plotters::prelude::*;
use tempfile::NamedTempFile;

use crate::analysis::{CheckError, radial_distribution};
use crate::artifacts::{RunArtifacts, validate_artifacts};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 480;
const RADIAL_BINS: usize = 64;
const RADIAL_WINDOW: usize = 20;
const MAX_BYTES: u64 = 2_000_000;

#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("video I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Analysis(#[from] CheckError),
    #[error("contract: {0}")]
    Contract(String),
    #[error("render frame {frame}: {message}")]
    Render { frame: usize, message: String },
    #[error("FFmpeg failed: {0}")]
    Ffmpeg(String),
    #[error("encoded video is {0} bytes; required < 2000000")]
    TooLarge(u64),
}

fn render_error(frame: usize, error: impl std::fmt::Display) -> VideoError {
    VideoError::Render {
        frame,
        message: error.to_string(),
    }
}

fn render_frame(
    artifacts: &RunArtifacts,
    frame_index: usize,
    output: &Path,
) -> Result<(), VideoError> {
    let frame = &artifacts.frames[frame_index];
    let start = (frame_index + 1).saturating_sub(RADIAL_WINDOW);
    let radial = radial_distribution(
        &artifacts.frames[start..=frame_index],
        &artifacts.run,
        RADIAL_BINS,
    )?;
    let root = BitMapBackend::new(output, (WIDTH, HEIGHT)).into_drawing_area();
    root.fill(&WHITE)
        .map_err(|error| render_error(frame_index, error))?;
    let panels = root.split_evenly((1, 2));

    let [box_x, box_y] = artifacts.run.box_size;
    let extent = box_x.max(box_y);
    let x_padding = 0.5 * (extent - box_x);
    let y_padding = 0.5 * (extent - box_y);
    let mut particles = ChartBuilder::on(&panels[0])
        .caption(
            format!("Periodic fluid: step {}, t={:.2}", frame.step, frame.t),
            ("sans-serif", 20),
        )
        .margin(12)
        .x_label_area_size(34)
        .y_label_area_size(42)
        .build_cartesian_2d(-x_padding..box_x + x_padding, -y_padding..box_y + y_padding)
        .map_err(|error| render_error(frame_index, error))?;
    particles
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .light_line_style(RGBColor(225, 225, 225))
        .draw()
        .map_err(|error| render_error(frame_index, error))?;
    particles
        .draw_series(std::iter::once(Rectangle::new(
            [(0.0, 0.0), (box_x, box_y)],
            BLACK.stroke_width(2),
        )))
        .map_err(|error| render_error(frame_index, error))?;
    particles
        .draw_series(
            frame
                .pos
                .iter()
                .map(|position| Circle::new((position[0], position[1]), 3, BLUE.filled())),
        )
        .map_err(|error| render_error(frame_index, error))?;

    let radial_max = radial.r.last().copied().unwrap_or(1.0);
    let observed_max = radial.g.iter().copied().fold(0.0, f64::max);
    let vertical_max = (1.1 * observed_max).max(2.5);
    let mut structure = ChartBuilder::on(&panels[1])
        .caption(
            format!("g(r), trailing {} frames", frame_index + 1 - start),
            ("sans-serif", 20),
        )
        .margin(12)
        .x_label_area_size(38)
        .y_label_area_size(45)
        .build_cartesian_2d(0.0..radial_max, 0.0..vertical_max)
        .map_err(|error| render_error(frame_index, error))?;
    structure
        .configure_mesh()
        .x_desc("r")
        .y_desc("g(r)")
        .light_line_style(RGBColor(225, 225, 225))
        .draw()
        .map_err(|error| render_error(frame_index, error))?;
    structure
        .draw_series(LineSeries::new(
            [(0.0, 1.0), (radial_max, 1.0)],
            BLACK.mix(0.35).stroke_width(1),
        ))
        .map_err(|error| render_error(frame_index, error))?;
    structure
        .draw_series(LineSeries::new(
            radial.r.into_iter().zip(radial.g),
            RED.stroke_width(2),
        ))
        .map_err(|error| render_error(frame_index, error))?;

    root.present()
        .map_err(|error| render_error(frame_index, error))
}

fn encode_frames(input_pattern: &Path, output: &Path, crf: u8) -> Result<(), VideoError> {
    let encoded = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-framerate")
        .arg("20")
        .arg("-start_number")
        .arg("0")
        .arg("-i")
        .arg(input_pattern)
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("medium")
        .arg("-crf")
        .arg(crf.to_string())
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-movflags")
        .arg("+faststart")
        .arg("-f")
        .arg("mp4")
        .arg(output)
        .output()
        .map_err(|error| VideoError::Ffmpeg(format!("could not start executable: {error}")))?;
    if !encoded.status.success() {
        return Err(VideoError::Ffmpeg(
            String::from_utf8_lossy(&encoded.stderr).trim().to_owned(),
        ));
    }
    Ok(())
}

pub fn render_video(artifacts: &RunArtifacts, output: &Path) -> Result<(), VideoError> {
    validate_artifacts(artifacts).map_err(|error| VideoError::Contract(error.to_string()))?;
    let frames_directory = tempfile::tempdir()?;
    for frame_index in 0..artifacts.frames.len() {
        let frame_path = frames_directory
            .path()
            .join(format!("frame_{frame_index:06}.png"));
        render_frame(artifacts, frame_index, &frame_path)?;
    }

    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary_video = NamedTempFile::new_in(parent)?;
    let temporary_path = temporary_video.path().to_path_buf();
    let input_pattern = frames_directory.path().join("frame_%06d.png");
    encode_frames(&input_pattern, &temporary_path, 30)?;
    let mut size = fs::metadata(&temporary_path)?.len();
    if size >= MAX_BYTES {
        encode_frames(&input_pattern, &temporary_path, 34)?;
        size = fs::metadata(&temporary_path)?.len();
    }
    if size >= MAX_BYTES {
        return Err(VideoError::TooLarge(size));
    }
    temporary_video
        .persist(output)
        .map_err(|error| VideoError::Io(error.error))?;
    Ok(())
}
