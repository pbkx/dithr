use dithr::{yliluoma_1_in_place, Buffer, IndexedImage, Palette, PixelFormat, Result};
use std::collections::BTreeSet;

fn main() -> Result<()> {
    let width = 32_usize;
    let height = 32_usize;
    let mut data = vec![0_u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = (x * 255 / (width - 1)) as u8;
            data[offset + 1] = (y * 255 / (height - 1)) as u8;
            data[offset + 2] = ((x * y) * 255 / ((width - 1) * (height - 1))) as u8;
        }
    }

    let palette = Palette::new(vec![
        [0, 0, 0],
        [255, 255, 255],
        [255, 64, 64],
        [64, 192, 255],
    ])?;
    let mut buffer = Buffer::new(
        &mut data,
        width,
        height,
        width * PixelFormat::Rgb8.bytes_per_pixel(),
        PixelFormat::Rgb8,
    )?;

    yliluoma_1_in_place(&mut buffer, &palette)?;

    let mut indices = Vec::with_capacity(width * height);
    for pixel in data.chunks_exact(3) {
        let rgb = [pixel[0], pixel[1], pixel[2]];
        assert!(palette.contains(rgb));
        indices.push(palette.nearest_rgb_index(rgb) as u8);
    }

    let indexed = IndexedImage {
        indices,
        width,
        height,
        palette: palette.clone(),
    };
    let used = indexed
        .indices
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .len();

    println!(
        "pixels={} palette_entries={} used_indices={}",
        indexed.indices.len(),
        indexed.palette.len(),
        used
    );
    Ok(())
}
