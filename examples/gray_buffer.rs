use dithr::{bayer_8x8_in_place, gray_u8, QuantizeMode, Result};

fn main() -> Result<()> {
    let width = 16_usize;
    let height = 16_usize;
    let mut data = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let value = ((x + y * width) * 255 / (width * height - 1)) as u8;
            data.push(value);
        }
    }

    let mut buffer = gray_u8(&mut data, width, height, width)?;
    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1))?;

    let black = data.iter().filter(|&&value| value == 0).count();
    let white = data.iter().filter(|&&value| value == 255).count();
    assert_eq!(black + white, data.len());

    println!("pixels={} black={} white={}", data.len(), black, white);
    Ok(())
}
