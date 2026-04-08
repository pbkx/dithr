use dithr::ordered::yliluoma_1_in_place;
use dithr::{rgb_u8, IndexedImage, Palette, Result};
use std::collections::BTreeSet;

fn main() -> Result<()> {
    let width = 64_usize;
    let height = 48_usize;
    let mut data = vec![0_u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            let r = (x * 255 / (width - 1)) as u8;
            let g = (y * 255 / (height - 1)) as u8;
            let b = ((x + y * 2) * 255 / (width + height * 2 - 3)) as u8;
            data[offset] = r;
            data[offset + 1] = g;
            data[offset + 2] = b;
        }
    }

    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
    ])?;
    let mut buffer = rgb_u8(&mut data, width, height, width * 3)?;

    yliluoma_1_in_place(&mut buffer, &palette)?;

    let mut indices = Vec::with_capacity(width * height);
    for pixel in data.chunks_exact(3) {
        let rgb = [pixel[0], pixel[1], pixel[2]];
        assert!(palette.contains(rgb));

        let index = palette.nearest_rgb_index(rgb) as u8;
        assert_eq!(palette.get(usize::from(index)), Some(rgb));
        indices.push(index);
    }

    let indexed = IndexedImage::new(indices, width, height, palette.clone())?;
    let used_indices = indexed.indices().iter().copied().collect::<BTreeSet<_>>();

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            let rgb = [data[offset], data[offset + 1], data[offset + 2]];
            assert_eq!(indexed.color_at(x, y), Some(rgb));
        }
    }

    assert!(used_indices.len() > 1);

    println!(
        "{}x{} pixels={} palette={} used={} indices={:?}",
        width,
        height,
        indexed.len(),
        indexed.palette().len(),
        used_indices.len(),
        used_indices
    );
    Ok(())
}
