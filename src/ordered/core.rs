use crate::{
    core::{PixelLayout, Sample},
    quantize_pixel, Buffer, BufferError, Error, PixelFormat, QuantizeMode, Result,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(crate) fn ordered_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u16],
    map_w: usize,
    map_h: usize,
    strength: f32,
) -> Result<()> {
    buffer.validate()?;
    validate_map(map, map_w, map_h)?;

    let map_min = map.iter().copied().min().unwrap_or_default();
    let map_max = map.iter().copied().max().unwrap_or_default();
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let channels = format.channels();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let threshold = ordered_threshold_for_xy(x, y, map, map_w, map_h);
            let bias = threshold_bias_unit(threshold, map_min, map_max, strength);
            let offset = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;

            match format {
                PixelFormat::Gray8 | PixelFormat::Gray16 => {
                    let value = apply_bias_unit(row[offset], bias);
                    let quantized = quantize_pixel::<S, crate::core::Gray>(&[value], mode)?;
                    row[offset] = quantized[0];
                }
                PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::Rgb32F => {
                    let adjusted = [
                        apply_bias_unit(row[offset], bias),
                        apply_bias_unit(row[offset + 1], bias),
                        apply_bias_unit(row[offset + 2], bias),
                    ];
                    let quantized = quantize_pixel::<S, crate::core::Rgb>(&adjusted, mode)?;
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                }
                PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::Rgba32F => {
                    let alpha = row[offset + 3];
                    let adjusted = [
                        apply_bias_unit(row[offset], bias),
                        apply_bias_unit(row[offset + 1], bias),
                        apply_bias_unit(row[offset + 2], bias),
                        alpha,
                    ];
                    let quantized = quantize_pixel::<S, crate::core::Rgba>(&adjusted, mode)?;
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                    row[offset + 3] = alpha;
                }
                _ => {
                    return Err(Error::UnsupportedFormat(
                        "ordered dithering supports Gray, Rgb, and Rgba formats only",
                    ));
                }
            }
        }
    }

    Ok(())
}

#[cfg(feature = "rayon")]
pub(crate) fn ordered_dither_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u16],
    map_w: usize,
    map_h: usize,
    strength: f32,
) -> Result<()> {
    buffer.validate()?;
    validate_map(map, map_w, map_h)?;

    let map_min = map.iter().copied().min().unwrap_or_default();
    let map_max = map.iter().copied().max().unwrap_or_default();
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let channels = format.channels();
    let stride = buffer.stride;

    buffer
        .data
        .par_chunks_mut(stride)
        .take(height)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<()> {
            for x in 0..width {
                let threshold = ordered_threshold_for_xy(x, y, map, map_w, map_h);
                let bias = threshold_bias_unit(threshold, map_min, map_max, strength);
                let offset = x * channels;

                match format {
                    PixelFormat::Gray8 | PixelFormat::Gray16 => {
                        let value = apply_bias_unit(row[offset], bias);
                        let quantized = quantize_pixel::<S, crate::core::Gray>(&[value], mode)?;
                        row[offset] = quantized[0];
                    }
                    PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::Rgb32F => {
                        let adjusted = [
                            apply_bias_unit(row[offset], bias),
                            apply_bias_unit(row[offset + 1], bias),
                            apply_bias_unit(row[offset + 2], bias),
                        ];
                        let quantized = quantize_pixel::<S, crate::core::Rgb>(&adjusted, mode)?;
                        row[offset] = quantized[0];
                        row[offset + 1] = quantized[1];
                        row[offset + 2] = quantized[2];
                    }
                    PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::Rgba32F => {
                        let alpha = row[offset + 3];
                        let adjusted = [
                            apply_bias_unit(row[offset], bias),
                            apply_bias_unit(row[offset + 1], bias),
                            apply_bias_unit(row[offset + 2], bias),
                            alpha,
                        ];
                        let quantized = quantize_pixel::<S, crate::core::Rgba>(&adjusted, mode)?;
                        row[offset] = quantized[0];
                        row[offset + 1] = quantized[1];
                        row[offset + 2] = quantized[2];
                        row[offset + 3] = alpha;
                    }
                    _ => {
                        return Err(Error::UnsupportedFormat(
                            "ordered dithering supports Gray, Rgb, and Rgba formats only",
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
    map: &[u16],
    map_w: usize,
    map_h: usize,
) -> u16 {
    debug_assert!(map_w > 0);
    debug_assert!(map_h > 0);
    debug_assert_eq!(map.len(), map_w * map_h);

    let map_x = x % map_w;
    let map_y = y % map_h;
    map[map_y * map_w + map_x]
}

fn validate_map(map: &[u16], map_w: usize, map_h: usize) -> Result<()> {
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

fn threshold_bias_unit(threshold: u16, map_min: u16, map_max: u16, strength: f32) -> f32 {
    if map_min >= map_max || strength == 0.0 {
        return 0.0;
    }

    let min = i32::from(map_min);
    let max = i32::from(map_max);
    let centered = i32::from(threshold) * 2 - (min + max);
    let range = (max - min) as f32;
    let strength_steps = strength * 255.0;
    let scaled_steps = (centered as f32 * strength_steps / range).trunc();
    scaled_steps / 255.0
}

fn apply_bias_unit<S: Sample>(value: S, bias: f32) -> S {
    S::from_unit_f32((value.to_unit_f32() + bias).clamp(0.0, 1.0))
}
