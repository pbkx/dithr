use crate::{
    math::{
        color::luma_u8,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, BufferError, DithrError, DithrResult, PixelFormat, QuantizeMode,
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
) -> DithrResult<()> {
    buffer.validate()?;
    validate_map(map, map_w, map_h)?;

    let map_max = map.iter().copied().max().unwrap_or_default();
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = buffer.format.bytes_per_pixel();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let threshold = ordered_threshold_for_xy(x, y, map, map_w, map_h);
            let bias = threshold_bias(threshold, map_max, strength);
            let offset = x.checked_mul(bpp).ok_or(BufferError::OutOfBounds)?;

            match format {
                PixelFormat::Gray8 => {
                    let value = apply_bias(row[offset], bias);
                    let quantized = quantize_pixel(PixelFormat::Gray8, &[value], mode);
                    row[offset] = luma_u8([quantized[0], quantized[1], quantized[2]]);
                }
                PixelFormat::Rgb8 => {
                    let adjusted = [
                        apply_bias(row[offset], bias),
                        apply_bias(row[offset + 1], bias),
                        apply_bias(row[offset + 2], bias),
                    ];
                    let quantized = quantize_pixel(PixelFormat::Rgb8, &adjusted, mode);
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
                    let quantized = quantize_pixel(PixelFormat::Rgba8, &adjusted, mode);
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                    row[offset + 3] = alpha;
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
) -> DithrResult<()> {
    buffer.validate()?;
    validate_map(map, map_w, map_h)?;

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
        .for_each(|(y, row)| {
            for x in 0..width {
                let threshold = ordered_threshold_for_xy(x, y, map, map_w, map_h);
                let bias = threshold_bias(threshold, map_max, strength);
                let offset = x * bpp;

                match format {
                    PixelFormat::Gray8 => {
                        let value = apply_bias(row[offset], bias);
                        let quantized = quantize_pixel(PixelFormat::Gray8, &[value], mode);
                        row[offset] = luma_u8([quantized[0], quantized[1], quantized[2]]);
                    }
                    PixelFormat::Rgb8 => {
                        let adjusted = [
                            apply_bias(row[offset], bias),
                            apply_bias(row[offset + 1], bias),
                            apply_bias(row[offset + 2], bias),
                        ];
                        let quantized = quantize_pixel(PixelFormat::Rgb8, &adjusted, mode);
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
                        let quantized = quantize_pixel(PixelFormat::Rgba8, &adjusted, mode);
                        row[offset] = quantized[0];
                        row[offset + 1] = quantized[1];
                        row[offset + 2] = quantized[2];
                        row[offset + 3] = alpha;
                    }
                }
            }
        });

    Ok(())
}

pub(crate) fn ordered_threshold_for_xy(
    x: usize,
    y: usize,
    map: &[u8],
    map_w: usize,
    map_h: usize,
) -> u8 {
    if map_w == 0 || map_h == 0 {
        return 0;
    }
    let Some(map_len) = map_w.checked_mul(map_h) else {
        return 0;
    };
    if map.len() != map_len {
        return 0;
    }

    let map_x = x % map_w;
    let map_y = y % map_h;
    let Some(index) = map_y
        .checked_mul(map_w)
        .and_then(|row_start| row_start.checked_add(map_x))
    else {
        return 0;
    };

    map.get(index).copied().unwrap_or_default()
}

fn validate_map(map: &[u8], map_w: usize, map_h: usize) -> DithrResult<()> {
    if map_w == 0 || map_h == 0 {
        return Err(DithrError::InvalidArgument(
            "ordered map dimensions must be positive",
        ));
    }

    let expected_len = map_w.checked_mul(map_h).ok_or(DithrError::InvalidArgument(
        "ordered map dimensions overflow",
    ))?;
    if map.len() != expected_len {
        return Err(DithrError::InvalidArgument(
            "ordered map length must match dimensions",
        ));
    }

    Ok(())
}

fn threshold_bias(threshold: u8, map_max: u8, strength: i16) -> i16 {
    if map_max == 0 || strength == 0 {
        return 0;
    }

    let centered = i32::from(threshold) * 2 - i32::from(map_max);
    let scaled = centered * i32::from(strength) / i32::from(map_max);

    clamp_i16(scaled)
}

fn apply_bias(value: u8, bias: i16) -> u8 {
    clamp_u8(i32::from(value) + i32::from(bias))
}
