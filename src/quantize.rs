use crate::{math::color::luma_u8, Error, Palette, PixelFormat, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeMode<'a> {
    GrayBits(u8),
    RgbBits(u8),
    Palette(&'a Palette),
    SingleColor { fg: [u8; 3], bits: u8 },
}

#[must_use]
#[inline]
pub fn quantize_gray_u8(value: u8, bits: u8) -> u8 {
    let bits = normalize_bits(bits);
    let levels = (1_u16 << bits) - 1;
    let q = ((u32::from(value) * u32::from(levels)) + 127) / 255;

    (((q * 255) + (u32::from(levels) / 2)) / u32::from(levels)) as u8
}

#[must_use]
#[inline]
pub fn quantize_rgb_u8(rgb: [u8; 3], bits: u8) -> [u8; 3] {
    [
        quantize_gray_u8(rgb[0], bits),
        quantize_gray_u8(rgb[1], bits),
        quantize_gray_u8(rgb[2], bits),
    ]
}

pub fn quantize_pixel(
    format: PixelFormat,
    pixel: &[u8],
    mode: QuantizeMode<'_>,
) -> Result<[u8; 4]> {
    let (rgb, alpha) = pixel_rgb_alpha(format, pixel)?;

    let out = match mode {
        QuantizeMode::GrayBits(bits) => {
            let g = quantize_gray_u8(luma_u8(rgb), bits);
            [g, g, g, alpha]
        }
        QuantizeMode::RgbBits(bits) => {
            let q = quantize_rgb_u8(rgb, bits);
            [q[0], q[1], q[2], alpha]
        }
        QuantizeMode::Palette(palette) => {
            let nearest = palette.nearest_rgb_index(rgb);
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
    };

    Ok(out)
}

#[inline]
pub fn quantize_error(original: &[u8], quantized: &[u8]) -> Result<[i16; 4]> {
    let len = original.len();
    if len != quantized.len() {
        return Err(Error::InvalidArgument(
            "original and quantized pixel lengths must match",
        ));
    }
    if len == 0 || len > 4 {
        return Err(Error::InvalidArgument("pixel length must be in 1..=4"));
    }

    let mut out = [0_i16; 4];
    for idx in 0..len {
        out[idx] = i16::from(original[idx]) - i16::from(quantized[idx]);
    }

    Ok(out)
}

#[must_use]
fn normalize_bits(bits: u8) -> u8 {
    bits.clamp(1, 8)
}

#[must_use]
fn expected_pixel_len(format: PixelFormat) -> usize {
    format.bytes_per_pixel()
}

fn pixel_rgb_alpha(format: PixelFormat, pixel: &[u8]) -> Result<([u8; 3], u8)> {
    let expected = expected_pixel_len(format);
    if pixel.len() != expected {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match format",
        ));
    }

    match format {
        PixelFormat::Gray8 => {
            let g = pixel[0];
            Ok(([g, g, g], 255))
        }
        PixelFormat::Rgb8 => Ok(([pixel[0], pixel[1], pixel[2]], 255)),
        PixelFormat::Rgba8 => Ok(([pixel[0], pixel[1], pixel[2]], pixel[3])),
    }
}

#[must_use]
#[inline]
fn scale_channel_by_gray(channel: u8, gray: u8) -> u8 {
    let scaled = (u32::from(channel) * u32::from(gray)) + 127;

    (scaled / 255) as u8
}
