use crate::{math::color::luma_u8, Palette, PixelFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeMode<'a> {
    GrayBits(u8),
    RgbBits(u8),
    Palette(&'a Palette),
    SingleColor { fg: [u8; 3], bits: u8 },
}

#[must_use]
pub fn quantize_gray_u8(value: u8, bits: u8) -> u8 {
    let bits = normalize_bits(bits);
    let levels = (1_u16 << bits) - 1;
    let q = ((u32::from(value) * u32::from(levels)) + 127) / 255;

    (((q * 255) + (u32::from(levels) / 2)) / u32::from(levels)) as u8
}

#[must_use]
pub fn quantize_rgb_u8(rgb: [u8; 3], bits: u8) -> [u8; 3] {
    [
        quantize_gray_u8(rgb[0], bits),
        quantize_gray_u8(rgb[1], bits),
        quantize_gray_u8(rgb[2], bits),
    ]
}

#[must_use]
pub fn quantize_pixel(format: PixelFormat, pixel: &[u8], mode: QuantizeMode<'_>) -> [u8; 4] {
    let (rgb, alpha) = pixel_rgb_alpha(format, pixel);

    match mode {
        QuantizeMode::GrayBits(bits) => {
            let g = quantize_gray_u8(luma_u8(rgb), bits);
            [g, g, g, alpha]
        }
        QuantizeMode::RgbBits(bits) => {
            let q = quantize_rgb_u8(rgb, bits);
            [q[0], q[1], q[2], alpha]
        }
        QuantizeMode::Palette(palette) => {
            let nearest = palette.nearest_rgb(rgb);
            let q = palette.as_slice()[nearest];
            [q[0], q[1], q[2], alpha]
        }
        QuantizeMode::SingleColor { fg, bits } => {
            let g = quantize_gray_u8(luma_u8(rgb), bits);
            [
                scale_channel_by_gray(fg[0], g),
                scale_channel_by_gray(fg[1], g),
                scale_channel_by_gray(fg[2], g),
                alpha,
            ]
        }
    }
}

#[must_use]
pub fn quantize_error(original: &[u8], quantized: &[u8]) -> [i16; 4] {
    let mut out = [0_i16; 4];

    for (idx, item) in out.iter_mut().enumerate() {
        let o = original.get(idx).copied().unwrap_or_default();
        let q = quantized.get(idx).copied().unwrap_or_default();
        *item = i16::from(o) - i16::from(q);
    }

    out
}

#[must_use]
fn normalize_bits(bits: u8) -> u8 {
    bits.clamp(1, 8)
}

#[must_use]
fn pixel_rgb_alpha(format: PixelFormat, pixel: &[u8]) -> ([u8; 3], u8) {
    match format {
        PixelFormat::Gray8 => {
            assert!(!pixel.is_empty(), "pixel slice too short for Gray8");
            let g = pixel[0];
            ([g, g, g], 255)
        }
        PixelFormat::Rgb8 => {
            assert!(pixel.len() >= 3, "pixel slice too short for Rgb8");
            ([pixel[0], pixel[1], pixel[2]], 255)
        }
        PixelFormat::Rgba8 => {
            assert!(pixel.len() >= 4, "pixel slice too short for Rgba8");
            ([pixel[0], pixel[1], pixel[2]], pixel[3])
        }
    }
}

#[must_use]
fn scale_channel_by_gray(channel: u8, gray: u8) -> u8 {
    let scaled = u16::from(channel)
        .checked_mul(u16::from(gray))
        .expect("scaled channel overflow");
    let rounded = scaled.checked_add(127).expect("rounded channel overflow");

    (rounded / 255) as u8
}
