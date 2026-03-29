use dithr::{bayer_8x8_rgb16_in_place, rgb_u16, Result};

fn main() -> Result<()> {
    let width = 64_usize;
    let height = 64_usize;
    let mut data = vec![0_u16; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = (x * 65_535 / (width - 1)) as u16;
            data[offset + 1] = (y * 65_535 / (height - 1)) as u16;
            data[offset + 2] = ((x + y) * 65_535 / (width + height - 2)) as u16;
        }
    }

    let mut buffer = rgb_u16(&mut data, width, height, width * 3)?;
    bayer_8x8_rgb16_in_place(&mut buffer, 2)?;

    let binary_channels = data.iter().all(|&value| value == 0 || value == 65_535);
    assert!(binary_channels);

    println!("pixels={} binary_channels=true", width * height);
    Ok(())
}
