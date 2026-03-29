use dithr::{floyd_steinberg_gray_u16_in_place, gray_u16, Result};

fn main() -> Result<()> {
    let width = 16_usize;
    let height = 16_usize;
    let mut data = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let value = ((x + y * width) * 65_535 / (width * height - 1)) as u16;
            data.push(value);
        }
    }

    let mut buffer = gray_u16(&mut data, width, height, width)?;
    floyd_steinberg_gray_u16_in_place(&mut buffer, 2)?;

    let black = data.iter().filter(|&&value| value == 0).count();
    let white = data.iter().filter(|&&value| value == 65_535).count();
    assert_eq!(black + white, data.len());

    println!("pixels={} black={} white={}", data.len(), black, white);
    Ok(())
}
