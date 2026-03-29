use dithr::{cga_palette, floyd_steinberg_in_place, Buffer, PixelFormat, QuantizeMode, Result};

fn main() -> Result<()> {
    let width = 64_usize;
    let height = 64_usize;
    let mut data = vec![0_u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = (x * 255 / (width - 1)) as u8;
            data[offset + 1] = (y * 255 / (height - 1)) as u8;
            data[offset + 2] = ((x + y) * 255 / (width + height - 2)) as u8;
        }
    }

    let palette = cga_palette();
    let mut buffer: Buffer<'_, u8> = Buffer::new(
        &mut data,
        width,
        height,
        width * PixelFormat::<()>::Rgb8.bytes_per_pixel(),
        PixelFormat::<()>::Rgb8,
    )?;
    floyd_steinberg_in_place(&mut buffer, QuantizeMode::Palette(&palette))?;

    let only_palette_colors = data
        .chunks_exact(3)
        .all(|pixel| palette.contains([pixel[0], pixel[1], pixel[2]]));
    assert!(only_palette_colors);

    println!(
        "pixels={} palette_entries={}",
        width * height,
        palette.len()
    );
    Ok(())
}
