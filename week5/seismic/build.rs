//! Compile the Enzyme kernel as an isolated `no_std` static library.
//!
//! This nightly's autodiff pass fails when it sees the whole program, so only
//! this one file is compiled with `-Zautodiff=Enable`; the main crate links it
//! through a C ABI and never enables autodiff globally.
use std::{env, path::PathBuf, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=src/kernel.rs");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let status = Command::new(env::var_os("RUSTC").unwrap())
        .args([
            "--edition=2024",
            "--crate-name",
            "enzyme_kernel",
            "--crate-type",
            "staticlib",
            "-C",
            "opt-level=3",
            "-C",
            "lto=fat",
            "-C",
            "panic=abort",
            "-Zautodiff=Enable",
            "src/kernel.rs",
            "-o",
        ])
        .arg(out.join("libenzyme_kernel.a"))
        .status()
        .expect("compile the Enzyme kernel with the pinned rustc");
    assert!(status.success(), "Enzyme kernel compilation failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=enzyme_kernel");
}
