use md::fluid::{RunConfig, simulate};
use md::video::render_video;

#[test]
#[ignore = "requires the external FFmpeg executable"]
fn short_video_is_nonempty_and_below_size_limit() {
    let artifacts = simulate(&RunConfig {
        n: 16,
        rho: 0.2,
        eq_steps: 50,
        steps: 20,
        sample_every: 5,
        ..RunConfig::default()
    })
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("short.mp4");

    render_video(&artifacts, &output).unwrap();

    let size = output.metadata().unwrap().len();
    assert!(size > 0);
    assert!(size < 2_000_000);
}
