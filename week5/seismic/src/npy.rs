//! Minimal NumPy `.npy` reader and writer (version 1.0, little-endian, C order).

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

fn header(descr: &str, shape: &[usize]) -> Vec<u8> {
    let inner = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let tuple = if shape.len() == 1 {
        format!("({inner},)")
    } else {
        format!("({inner})")
    };
    let mut text =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {tuple}, }}");
    let base = 6 + 2 + 2 + text.len() + 1;
    let pad = (64 - (base % 64)) % 64;
    text.push_str(&" ".repeat(pad));
    text.push('\n');
    text.into_bytes()
}

fn write_bytes(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = BufWriter::new(File::create(path)?);
    let text = header(descr, shape);
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&(text.len() as u16).to_le_bytes())?;
    file.write_all(&text)?;
    file.write_all(bytes)?;
    file.flush()
}

pub fn write_f64(path: &Path, shape: &[usize], data: &[f64]) -> std::io::Result<()> {
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "npy shape does not match the data length");
    let mut bytes = Vec::with_capacity(total * 8);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_bytes(path, "<f8", shape, &bytes)
}

pub fn write_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let total: usize = shape.iter().product();
    assert_eq!(data.len(), total, "npy shape does not match the data length");
    let mut bytes = Vec::with_capacity(total * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_bytes(path, "<f4", shape, &bytes)
}

pub fn read_f64(path: &Path) -> std::io::Result<(Vec<usize>, Vec<f64>)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "not a NumPy .npy file");
    let major = bytes[6];
    let (header_len, offset) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let text = std::str::from_utf8(&bytes[offset..offset + header_len]).unwrap();
    assert!(text.contains("'fortran_order': False"), "Fortran order is not supported");
    let descr = text
        .split("'descr': '")
        .nth(1)
        .and_then(|s| s.split('\'').next())
        .expect("missing descr");
    assert!(descr == "<f8" || descr == "|f8", "expected little-endian float64");
    let shape_text = text
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .expect("missing shape");
    let shape: Vec<usize> = shape_text
        .split(',')
        .filter_map(|s| {
            let s = s.trim();
            if s.is_empty() { None } else { Some(s.parse().unwrap()) }
        })
        .collect();
    let data_bytes = &bytes[offset + header_len..];
    let total: usize = shape.iter().product();
    assert_eq!(data_bytes.len(), total * 8, "unexpected .npy payload size");
    let data = (0..total)
        .map(|i| {
            let mut chunk = [0u8; 8];
            chunk.copy_from_slice(&data_bytes[i * 8..i * 8 + 8]);
            f64::from_le_bytes(chunk)
        })
        .collect();
    Ok((shape, data))
}
