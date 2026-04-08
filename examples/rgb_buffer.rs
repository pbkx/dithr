use dithr::diffusion::floyd_steinberg_in_place;
use dithr::{rgb_u8, QuantizeMode, Result};
use std::collections::BTreeSet;

fn main() -> Result<()> {
    let width = 96_usize;
    let height = 72_usize;
    let mut data = vec![0_u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            let r = (x * 255 / (width - 1)) as u8;
            let g = (y * 255 / (height - 1)) as u8;
            let b = ((x + (2 * y)) * 255 / (width + (2 * height) - 3)) as u8;
            data[offset] = r;
            data[offset + 1] = g;
            data[offset + 2] = b;
        }
    }

    let mut buffer = rgb_u8(&mut data, width, height, width * 3)?;
    floyd_steinberg_in_place(&mut buffer, QuantizeMode::RgbLevels(4))?;

    let allowed = [0_u8, 85_u8, 170_u8, 255_u8];
    let quantized = data.iter().all(|value| allowed.contains(value));
    assert!(quantized);

    let mut r_values = BTreeSet::new();
    let mut g_values = BTreeSet::new();
    let mut b_values = BTreeSet::new();
    for pixel in data.chunks_exact(3) {
        r_values.insert(pixel[0]);
        g_values.insert(pixel[1]);
        b_values.insert(pixel[2]);
    }

    assert!(r_values.len() > 1 && g_values.len() > 1 && b_values.len() > 1);

    println!(
        "{}x{} pixels={} distinct_r={} distinct_g={} distinct_b={} quantized={}",
        width,
        height,
        width * height,
        r_values.len(),
        g_values.len(),
        b_values.len(),
        quantized
    );
    Ok(())
}
