use dithr::{floyd_steinberg_in_place, gray_u8, QuantizeMode, Result};

fn main() -> Result<()> {
    let width = 128_usize;
    let height = 96_usize;
    let mut data = Vec::with_capacity(width * height);

    for y in 0..height {
        for x in 0..width {
            let horizontal = (x * 255 / (width - 1)) as u8;
            let vertical = (y * 48 / (height - 1)) as u8;
            data.push(horizontal.saturating_add(vertical));
        }
    }

    let mut buffer = gray_u8(&mut data, width, height, width)?;
    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayLevels(2))?;

    let low = data.iter().filter(|&&value| value == 0).count();
    let high = data.iter().filter(|&&value| value == 255).count();
    let is_binary = low + high == data.len();
    let min = data.iter().copied().min().unwrap_or(0);
    let max = data.iter().copied().max().unwrap_or(0);

    assert!(is_binary);
    assert!(low > 0 && high > 0);

    println!(
        "{}x{} pixels={} min={} max={} low={} high={} binary={}",
        width,
        height,
        data.len(),
        min,
        max,
        low,
        high,
        is_binary
    );
    Ok(())
}
