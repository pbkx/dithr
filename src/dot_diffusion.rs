use crate::{
    core::{PixelLayout, Sample},
    math::fixed::mul_div_i32,
    quantize_pixel, Buffer, Error, PixelFormat, QuantizeMode, Result,
};
use std::mem::size_of;

const CLASS_MATRIX_W: usize = 4;
const CLASS_MATRIX_H: usize = 4;
const CLASS_COUNT: usize = CLASS_MATRIX_W * CLASS_MATRIX_H;
const CLASS_MATRIX: [[u8; CLASS_MATRIX_W]; CLASS_MATRIX_H] =
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
const DIFFUSION_WEIGHTS_3X3: [[i16; 3]; 3] = [[1, 2, 1], [2, 0, 2], [1, 2, 1]];

#[derive(Clone, Copy)]
struct DiffusionDims {
    width: usize,
    height: usize,
    max_value: i32,
}

pub fn knuth_dot_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    match buffer.format {
        PixelFormat::Gray8 | PixelFormat::Gray16 => dot_diffuse_gray(buffer, mode),
        PixelFormat::Rgb8
        | PixelFormat::Rgba8
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Rgb32F
        | PixelFormat::Rgba32F => dot_diffuse_rgb(buffer, mode),
        _ => Err(Error::UnsupportedFormat(
            "knuth dot diffusion supports Gray, Rgb, and Rgba formats only",
        )),
    }
}

fn dot_diffuse_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    if S::IS_FLOAT {
        return dot_diffuse_gray_float(buffer, mode);
    }

    dot_diffuse_gray_int(buffer, mode)
}

fn dot_diffuse_gray_int<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let max_value = integer_sample_max::<S>()?;
    let dims = DiffusionDims {
        width,
        height,
        max_value,
    };
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut errors = vec![0.0_f32; pixel_count];

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for (x, value) in row.iter_mut().take(width).enumerate() {
                if class_at(x, y) != class {
                    continue;
                }

                let idx = y * width + x;
                let adjusted = (sample_to_domain(*value, max_value)
                    + unit_to_domain(errors[idx], max_value))
                .clamp(0, max_value);
                let quantized = quantize_pixel::<S, crate::core::Gray>(
                    &[domain_to_sample::<S>(adjusted, max_value)],
                    mode,
                )?;
                let quantized_gray = quantized[0];
                *value = quantized_gray;

                let err = adjusted - sample_to_domain(quantized_gray, max_value);
                diffuse_gray_to_higher_classes_int(&mut errors, dims, x, y, class, err)?;
            }
        }
    }

    Ok(())
}

fn dot_diffuse_gray_float<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut errors = vec![0.0_f32; pixel_count];

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for (x, value) in row.iter_mut().take(width).enumerate() {
                if class_at(x, y) != class {
                    continue;
                }

                let idx = y * width + x;
                let adjusted = (value.to_unit_f32() + errors[idx]).clamp(0.0, 1.0);
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                let quantized_gray = quantized[0];
                *value = quantized_gray;

                let err = adjusted - quantized_gray.to_unit_f32();
                diffuse_gray_to_higher_classes_float(&mut errors, width, height, x, y, class, err);
            }
        }
    }

    Ok(())
}

fn dot_diffuse_rgb<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    if S::IS_FLOAT {
        return dot_diffuse_rgb_float(buffer, mode);
    }

    dot_diffuse_rgb_int(buffer, mode)
}

fn dot_diffuse_rgb_int<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let max_value = integer_sample_max::<S>()?;
    let dims = DiffusionDims {
        width,
        height,
        max_value,
    };
    let sample_channels = format.channels();
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];
    let is_rgba = format.has_alpha();

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for x in 0..width {
                if class_at(x, y) != class {
                    continue;
                }

                let offset = x * sample_channels;
                let base = (y * width + x) * channels;
                let adjusted = [
                    (sample_to_domain(row[offset], max_value)
                        + unit_to_domain(errors[base], max_value))
                    .clamp(0, max_value),
                    (sample_to_domain(row[offset + 1], max_value)
                        + unit_to_domain(errors[base + 1], max_value))
                    .clamp(0, max_value),
                    (sample_to_domain(row[offset + 2], max_value)
                        + unit_to_domain(errors[base + 2], max_value))
                    .clamp(0, max_value),
                ];
                let alpha = if is_rgba {
                    row[offset + 3]
                } else {
                    S::from_unit_f32(1.0)
                };
                let quantized = if is_rgba {
                    let pixel = [
                        domain_to_sample::<S>(adjusted[0], max_value),
                        domain_to_sample::<S>(adjusted[1], max_value),
                        domain_to_sample::<S>(adjusted[2], max_value),
                        alpha,
                    ];
                    quantize_pixel::<S, crate::core::Rgba>(&pixel, mode)?
                } else {
                    let pixel = [
                        domain_to_sample::<S>(adjusted[0], max_value),
                        domain_to_sample::<S>(adjusted[1], max_value),
                        domain_to_sample::<S>(adjusted[2], max_value),
                    ];
                    quantize_pixel::<S, crate::core::Rgb>(&pixel, mode)?
                };

                row[offset] = quantized[0];
                row[offset + 1] = quantized[1];
                row[offset + 2] = quantized[2];
                if is_rgba {
                    row[offset + 3] = alpha;
                }

                let err = [
                    adjusted[0] - sample_to_domain(quantized[0], max_value),
                    adjusted[1] - sample_to_domain(quantized[1], max_value),
                    adjusted[2] - sample_to_domain(quantized[2], max_value),
                ];
                diffuse_rgb_to_higher_classes_int(&mut errors, dims, x, y, class, err)?;
            }
        }
    }

    Ok(())
}

fn dot_diffuse_rgb_float<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let sample_channels = format.channels();
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];
    let is_rgba = format.has_alpha();

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for x in 0..width {
                if class_at(x, y) != class {
                    continue;
                }

                let offset = x * sample_channels;
                let base = (y * width + x) * channels;
                let adjusted = [
                    (row[offset].to_unit_f32() + errors[base]).clamp(0.0, 1.0),
                    (row[offset + 1].to_unit_f32() + errors[base + 1]).clamp(0.0, 1.0),
                    (row[offset + 2].to_unit_f32() + errors[base + 2]).clamp(0.0, 1.0),
                ];
                let alpha = if is_rgba {
                    row[offset + 3]
                } else {
                    S::from_unit_f32(1.0)
                };
                let quantized = if is_rgba {
                    let pixel = [
                        S::from_unit_f32(adjusted[0]),
                        S::from_unit_f32(adjusted[1]),
                        S::from_unit_f32(adjusted[2]),
                        alpha,
                    ];
                    quantize_pixel::<S, crate::core::Rgba>(&pixel, mode)?
                } else {
                    let pixel = [
                        S::from_unit_f32(adjusted[0]),
                        S::from_unit_f32(adjusted[1]),
                        S::from_unit_f32(adjusted[2]),
                    ];
                    quantize_pixel::<S, crate::core::Rgb>(&pixel, mode)?
                };

                row[offset] = quantized[0];
                row[offset + 1] = quantized[1];
                row[offset + 2] = quantized[2];
                if is_rgba {
                    row[offset + 3] = alpha;
                }

                let err = [
                    adjusted[0] - quantized[0].to_unit_f32(),
                    adjusted[1] - quantized[1].to_unit_f32(),
                    adjusted[2] - quantized[2].to_unit_f32(),
                ];
                diffuse_rgb_to_higher_classes_float(&mut errors, width, height, x, y, class, err);
            }
        }
    }

    Ok(())
}

fn diffuse_gray_to_higher_classes_float(
    errors: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
    err: f32,
) {
    let targets = higher_class_neighbors(width, height, x, y, class);
    if targets.is_empty() {
        return;
    }
    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| f32::from(weight))
        .sum::<f32>();

    for (nx, ny, weight) in targets {
        let distributed = err * f32::from(weight) / total_weight;
        let idx = ny * width + nx;
        errors[idx] += distributed;
    }
}

fn diffuse_gray_to_higher_classes_int(
    errors: &mut [f32],
    dims: DiffusionDims,
    x: usize,
    y: usize,
    class: u8,
    err: i32,
) -> Result<()> {
    let targets = higher_class_neighbors(dims.width, dims.height, x, y, class);
    if targets.is_empty() {
        return Ok(());
    }
    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| i32::from(weight))
        .sum::<i32>();

    for (nx, ny, weight) in targets {
        let distributed = mul_div_i32(err, i32::from(weight), total_weight)?;
        let idx = ny * dims.width + nx;
        errors[idx] += domain_to_unit(distributed, dims.max_value);
    }

    Ok(())
}

fn diffuse_rgb_to_higher_classes_float(
    errors: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
    err: [f32; 3],
) {
    let targets = higher_class_neighbors(width, height, x, y, class);
    if targets.is_empty() {
        return;
    }

    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| f32::from(weight))
        .sum::<f32>();

    for (nx, ny, weight) in targets {
        let w = f32::from(weight) / total_weight;
        let base = (ny * width + nx) * 3;
        errors[base] += err[0] * w;
        errors[base + 1] += err[1] * w;
        errors[base + 2] += err[2] * w;
    }
}

fn diffuse_rgb_to_higher_classes_int(
    errors: &mut [f32],
    dims: DiffusionDims,
    x: usize,
    y: usize,
    class: u8,
    err: [i32; 3],
) -> Result<()> {
    let targets = higher_class_neighbors(dims.width, dims.height, x, y, class);
    if targets.is_empty() {
        return Ok(());
    }

    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| i32::from(weight))
        .sum::<i32>();

    for (nx, ny, weight) in targets {
        let share = [
            mul_div_i32(err[0], i32::from(weight), total_weight)?,
            mul_div_i32(err[1], i32::from(weight), total_weight)?,
            mul_div_i32(err[2], i32::from(weight), total_weight)?,
        ];
        let base = (ny * dims.width + nx) * 3;
        errors[base] += domain_to_unit(share[0], dims.max_value);
        errors[base + 1] += domain_to_unit(share[1], dims.max_value);
        errors[base + 2] += domain_to_unit(share[2], dims.max_value);
    }

    Ok(())
}

fn higher_class_neighbors(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
) -> Vec<(usize, usize, i16)> {
    let mut targets = Vec::with_capacity(8);

    let x0 = x.saturating_sub(1);
    let y0 = y.saturating_sub(1);
    let x1 = (x + 1).min(width - 1);
    let y1 = (y + 1).min(height - 1);

    for ny in y0..=y1 {
        for nx in x0..=x1 {
            if nx == x && ny == y {
                continue;
            }
            let kernel_x = (nx as isize - x as isize + 1) as usize;
            let kernel_y = (ny as isize - y as isize + 1) as usize;
            let weight = DIFFUSION_WEIGHTS_3X3[kernel_y][kernel_x];
            if weight <= 0 {
                continue;
            }
            if class_at(nx, ny) > class {
                targets.push((nx, ny, weight));
            }
        }
    }

    targets
}

fn class_at(x: usize, y: usize) -> u8 {
    CLASS_MATRIX[y % CLASS_MATRIX_H][x % CLASS_MATRIX_W]
}

fn integer_sample_max<S: Sample>() -> Result<i32> {
    if S::IS_FLOAT {
        return Err(Error::UnsupportedFormat(
            "dot diffusion integer path does not support floating-point samples",
        ));
    }

    match size_of::<S>() {
        1 => Ok(255),
        2 => Ok(65_535),
        _ => Err(Error::UnsupportedFormat(
            "unsupported integer sample width for dot diffusion",
        )),
    }
}

fn sample_to_domain<S: Sample>(value: S, max_value: i32) -> i32 {
    (value.to_unit_f32().clamp(0.0, 1.0) * max_value as f32).round() as i32
}

fn domain_to_sample<S: Sample>(value: i32, max_value: i32) -> S {
    S::from_unit_f32(value.clamp(0, max_value) as f32 / max_value as f32)
}

fn unit_to_domain(unit: f32, max_value: i32) -> i32 {
    (unit * max_value as f32).round() as i32
}

fn domain_to_unit(value: i32, max_value: i32) -> f32 {
    value as f32 / max_value as f32
}
