use crate::{
    math::{
        color::luma_u8,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, BufferError, Error, PixelFormat, QuantizeMode, Result,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(crate) fn ordered_dither_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> Result<()> {
    buffer.validate()?;
    validate_map(map, map_w, map_h)?;

    let map_min = map.iter().copied().min().unwrap_or_default();
    let map_max = map.iter().copied().max().unwrap_or_default();
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = buffer.format.bytes_per_pixel();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let threshold = ordered_threshold_for_xy(x, y, map, map_w, map_h)?;
            let bias = threshold_bias(threshold, map_min, map_max, strength);
            let offset = x.checked_mul(bpp).ok_or(BufferError::OutOfBounds)?;

            match format {
                PixelFormat::Gray8 => {
                    let value = apply_bias(row[offset], bias);
                    let quantized = quantize_pixel(PixelFormat::Gray8, &[value], mode)?;
                    row[offset] = luma_u8([quantized[0], quantized[1], quantized[2]]);
                }
                PixelFormat::Rgb8 => {
                    let adjusted = [
                        apply_bias(row[offset], bias),
                        apply_bias(row[offset + 1], bias),
                        apply_bias(row[offset + 2], bias),
                    ];
                    let quantized = quantize_pixel(PixelFormat::Rgb8, &adjusted, mode)?;
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                }
                PixelFormat::Rgba8 => {
                    let alpha = row[offset + 3];
                    let adjusted = [
                        apply_bias(row[offset], bias),
                        apply_bias(row[offset + 1], bias),
                        apply_bias(row[offset + 2], bias),
                        alpha,
                    ];
                    let quantized = quantize_pixel(PixelFormat::Rgba8, &adjusted, mode)?;
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                    row[offset + 3] = alpha;
                }
                _ => {
                    return Err(Error::UnsupportedFormat(
                        "ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "rayon")]
pub(crate) fn ordered_dither_in_place_par(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> Result<()> {
    buffer.validate()?;
    validate_map(map, map_w, map_h)?;

    let map_min = map.iter().copied().min().unwrap_or_default();
    let map_max = map.iter().copied().max().unwrap_or_default();
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = buffer.format.bytes_per_pixel();
    let stride = buffer.stride;

    buffer
        .data
        .par_chunks_mut(stride)
        .take(height)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<()> {
            for x in 0..width {
                let threshold = ordered_threshold_for_xy(x, y, map, map_w, map_h)?;
                let bias = threshold_bias(threshold, map_min, map_max, strength);
                let offset = x * bpp;

                match format {
                    PixelFormat::Gray8 => {
                        let value = apply_bias(row[offset], bias);
                        let quantized = quantize_pixel(PixelFormat::Gray8, &[value], mode)?;
                        row[offset] = luma_u8([quantized[0], quantized[1], quantized[2]]);
                    }
                    PixelFormat::Rgb8 => {
                        let adjusted = [
                            apply_bias(row[offset], bias),
                            apply_bias(row[offset + 1], bias),
                            apply_bias(row[offset + 2], bias),
                        ];
                        let quantized = quantize_pixel(PixelFormat::Rgb8, &adjusted, mode)?;
                        row[offset] = quantized[0];
                        row[offset + 1] = quantized[1];
                        row[offset + 2] = quantized[2];
                    }
                    PixelFormat::Rgba8 => {
                        let alpha = row[offset + 3];
                        let adjusted = [
                            apply_bias(row[offset], bias),
                            apply_bias(row[offset + 1], bias),
                            apply_bias(row[offset + 2], bias),
                            alpha,
                        ];
                        let quantized = quantize_pixel(PixelFormat::Rgba8, &adjusted, mode)?;
                        row[offset] = quantized[0];
                        row[offset + 1] = quantized[1];
                        row[offset + 2] = quantized[2];
                        row[offset + 3] = alpha;
                    }
                    _ => {
                        return Err(Error::UnsupportedFormat(
                            "ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                        ));
                    }
                }
            }
            Ok(())
        })?;

    Ok(())
}

pub(crate) fn ordered_threshold_for_xy(
    x: usize,
    y: usize,
    map: &[u8],
    map_w: usize,
    map_h: usize,
) -> Result<u8> {
    if map_w == 0 || map_h == 0 {
        return Err(Error::InvalidArgument(
            "ordered map dimensions must be positive",
        ));
    }
    let Some(map_len) = map_w.checked_mul(map_h) else {
        return Err(Error::InvalidArgument("ordered map dimensions overflow"));
    };
    if map.len() != map_len {
        return Err(Error::InvalidArgument(
            "ordered map length must match dimensions",
        ));
    }

    let map_x = x % map_w;
    let map_y = y % map_h;
    let Some(index) = map_y
        .checked_mul(map_w)
        .and_then(|row_start| row_start.checked_add(map_x))
    else {
        return Err(Error::InvalidArgument("ordered threshold index overflow"));
    };

    map.get(index).copied().ok_or(Error::InvalidArgument(
        "ordered threshold index out of bounds",
    ))
}

fn validate_map(map: &[u8], map_w: usize, map_h: usize) -> Result<()> {
    if map_w == 0 || map_h == 0 {
        return Err(Error::InvalidArgument(
            "ordered map dimensions must be positive",
        ));
    }

    let expected_len = map_w
        .checked_mul(map_h)
        .ok_or(Error::InvalidArgument("ordered map dimensions overflow"))?;
    if map.len() != expected_len {
        return Err(Error::InvalidArgument(
            "ordered map length must match dimensions",
        ));
    }

    Ok(())
}

fn threshold_bias(threshold: u8, map_min: u8, map_max: u8, strength: i16) -> i16 {
    if map_min >= map_max || strength == 0 {
        return 0;
    }

    let min = i32::from(map_min);
    let max = i32::from(map_max);
    let centered = i32::from(threshold) * 2 - (min + max);
    let range = max - min;
    let scaled = centered * i32::from(strength) / range;

    clamp_i16(scaled)
}

fn apply_bias(value: u8, bias: i16) -> u8 {
    clamp_u8(i32::from(value) + i32::from(bias))
}
