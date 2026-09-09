# Week 2: Agentic coding with Rust

This directory contains a two-dimensional molecular-dynamics program built in
stages from tested Lennard-Jones physics.

## Pair field

From `week2/`, reproduce `field.png` with:

```bash
cargo run --manifest-path md/Cargo.toml --release --example field -- field.png
```

The color is the pair energy capped to the interval `[-1, 1]`. The arrows show
the radial force, and the dashed circle marks the zero-force separation
`r0 = 2^(1/6)`.
