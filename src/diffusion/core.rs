use crate::{
    core::{PixelLayout, Sample},
    data::ErrorKernel,
    math::fixed::mul_div_i32,
    quantize_pixel, Buffer, BufferError, Error, PixelFormat, QuantizeMode, Result,
};
use std::mem::size_of;

pub(crate) fn error_diffuse_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    buffer.validate()?;
    if kernel.weight_den <= 0 {
        return Err(Error::InvalidArgument(
            "kernel denominator must be positive",
        ));
    }

    match buffer.format {
        PixelFormat::Gray8 | PixelFormat::Gray16 => diffuse_gray_row_major(buffer, mode, kernel),
        PixelFormat::Rgb8
        | PixelFormat::Rgba8
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Rgb32F
        | PixelFormat::Rgba32F => diffuse_rgb_row_major(buffer, mode, kernel),
        _ => Err(Error::UnsupportedFormat(
            "error diffusion core supports Gray, Rgb, and Rgba formats only",
        )),
    }
}

pub(crate) fn add_error_to_pixel(
    errors: &mut [f32],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    channels: usize,
    delta: [f32; 3],
) {
    if x < 0 || y < 0 {
        return;
    }

    let x = x as usize;
    let y = y as usize;
    if x >= width || y >= height {
        return;
    }

    let pixel_index = y * width + x;
    let base = pixel_index * channels;

    if channels == 1 {
        errors[base] += delta[0];
        return;
    }

    errors[base] += delta[0];
    errors[base + 1] += delta[1];
    errors[base + 2] += delta[2];
}

fn add_error_to_pixel_i32(
    errors: &mut [i32],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    channels: usize,
    delta: [i32; 3],
) {
    if x < 0 || y < 0 {
        return;
    }

    let x = x as usize;
    let y = y as usize;
    if x >= width || y >= height {
        return;
    }

    let pixel_index = y * width + x;
    let base = pixel_index * channels;

    if channels == 1 {
        errors[base] += delta[0];
        return;
    }

    errors[base] += delta[0];
    errors[base + 1] += delta[1];
    errors[base + 2] += delta[2];
}

pub(crate) fn diffuse_gray_row_major<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    if S::IS_FLOAT {
        return diffuse_gray_row_major_float(buffer, mode, kernel);
    }

    diffuse_gray_row_major_int(buffer, mode, kernel)
}

fn diffuse_gray_row_major_int<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let max_value = integer_sample_max::<S>()?;
    let denominator = i32::from(kernel.weight_den);
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut errors = vec![0_i32; pixel_count];

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let row_base = y * width;

        for (x, value) in row.iter_mut().enumerate().take(width) {
            let idx = row_base + x;
            let adjusted = (sample_to_domain(*value, max_value) + errors[idx]).clamp(0, max_value);
            let adjusted_sample = domain_to_sample::<S>(adjusted, max_value);
            let quantized = quantize_pixel::<S, crate::core::Gray>(&[adjusted_sample], mode)?;
            let quantized_gray = quantized[0];
            *value = quantized_gray;

            let err = adjusted - sample_to_domain(quantized_gray, max_value);

            for tap in kernel.taps {
                let nx = x as isize + isize::from(tap.dx);
                let ny = y as isize + isize::from(tap.dy);
                let distributed = mul_div_i32(err, i32::from(tap.weight_num), denominator)?;
                add_error_to_pixel_i32(&mut errors, width, height, nx, ny, 1, [distributed, 0, 0]);
            }
        }
    }

    Ok(())
}

fn diffuse_gray_row_major_float<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let denominator = f32::from(kernel.weight_den);
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut errors = vec![0.0_f32; pixel_count];

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let row_base = y * width;

        for (x, value) in row.iter_mut().enumerate().take(width) {
            let idx = row_base + x;
            let adjusted_unit = (value.to_unit_f32() + errors[idx]).clamp(0.0, 1.0);
            let adjusted = S::from_unit_f32(adjusted_unit);
            let quantized = quantize_pixel::<S, crate::core::Gray>(&[adjusted], mode)?;
            let quantized_gray = quantized[0];
            *value = quantized_gray;

            let err = adjusted_unit - quantized_gray.to_unit_f32();

            for tap in kernel.taps {
                let nx = x as isize + isize::from(tap.dx);
                let ny = y as isize + isize::from(tap.dy);
                let distributed = err * f32::from(tap.weight_num) / denominator;
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    nx,
                    ny,
                    1,
                    [distributed, 0.0, 0.0],
                );
            }
        }
    }

    Ok(())
}

pub(crate) fn diffuse_rgb_row_major<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    if S::IS_FLOAT {
        return diffuse_rgb_row_major_float(buffer, mode, kernel);
    }

    diffuse_rgb_row_major_int(buffer, mode, kernel)
}

fn diffuse_rgb_row_major_int<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let max_value = integer_sample_max::<S>()?;
    let denominator = i32::from(kernel.weight_den);
    let channels = 3_usize;
    let sample_channels = format.channels();
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0_i32; error_len];
    let is_rgba = format.has_alpha();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let row_error_base = y * width * channels;

        for x in 0..width {
            let offset = x
                .checked_mul(sample_channels)
                .ok_or(BufferError::OutOfBounds)?;
            let error_base = row_error_base + x * channels;
            let adjusted = [
                (sample_to_domain(row[offset], max_value) + errors[error_base]).clamp(0, max_value),
                (sample_to_domain(row[offset + 1], max_value) + errors[error_base + 1])
                    .clamp(0, max_value),
                (sample_to_domain(row[offset + 2], max_value) + errors[error_base + 2])
                    .clamp(0, max_value),
            ];
            let adjusted_sample = [
                domain_to_sample::<S>(adjusted[0], max_value),
                domain_to_sample::<S>(adjusted[1], max_value),
                domain_to_sample::<S>(adjusted[2], max_value),
            ];
            let alpha = if is_rgba {
                row[offset + 3]
            } else {
                S::from_unit_f32(1.0)
            };
            let quantized = if is_rgba {
                let pixel = [
                    adjusted_sample[0],
                    adjusted_sample[1],
                    adjusted_sample[2],
                    alpha,
                ];
                quantize_pixel::<S, crate::core::Rgba>(&pixel, mode)?
            } else {
                quantize_pixel::<S, crate::core::Rgb>(&adjusted_sample, mode)?
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

            for tap in kernel.taps {
                let nx = x as isize + isize::from(tap.dx);
                let ny = y as isize + isize::from(tap.dy);
                let distributed = [
                    mul_div_i32(err[0], i32::from(tap.weight_num), denominator)?,
                    mul_div_i32(err[1], i32::from(tap.weight_num), denominator)?,
                    mul_div_i32(err[2], i32::from(tap.weight_num), denominator)?,
                ];
                add_error_to_pixel_i32(&mut errors, width, height, nx, ny, channels, distributed);
            }
        }
    }

    Ok(())
}

fn diffuse_rgb_row_major_float<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let denominator = f32::from(kernel.weight_den);
    let channels = 3_usize;
    let sample_channels = format.channels();
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];
    let is_rgba = format.has_alpha();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let row_error_base = y * width * channels;

        for x in 0..width {
            let offset = x
                .checked_mul(sample_channels)
                .ok_or(BufferError::OutOfBounds)?;
            let error_base = row_error_base + x * channels;
            let adjusted_unit = [
                (row[offset].to_unit_f32() + errors[error_base]).clamp(0.0, 1.0),
                (row[offset + 1].to_unit_f32() + errors[error_base + 1]).clamp(0.0, 1.0),
                (row[offset + 2].to_unit_f32() + errors[error_base + 2]).clamp(0.0, 1.0),
            ];
            let adjusted = [
                S::from_unit_f32(adjusted_unit[0]),
                S::from_unit_f32(adjusted_unit[1]),
                S::from_unit_f32(adjusted_unit[2]),
            ];
            let alpha = if is_rgba {
                row[offset + 3]
            } else {
                S::from_unit_f32(1.0)
            };
            let quantized = if is_rgba {
                let pixel = [adjusted[0], adjusted[1], adjusted[2], alpha];
                quantize_pixel::<S, crate::core::Rgba>(&pixel, mode)?
            } else {
                quantize_pixel::<S, crate::core::Rgb>(&adjusted, mode)?
            };

            row[offset] = quantized[0];
            row[offset + 1] = quantized[1];
            row[offset + 2] = quantized[2];
            if is_rgba {
                row[offset + 3] = alpha;
            }

            let err = [
                adjusted_unit[0] - quantized[0].to_unit_f32(),
                adjusted_unit[1] - quantized[1].to_unit_f32(),
                adjusted_unit[2] - quantized[2].to_unit_f32(),
            ];

            for tap in kernel.taps {
                let nx = x as isize + isize::from(tap.dx);
                let ny = y as isize + isize::from(tap.dy);
                let distributed = [
                    err[0] * f32::from(tap.weight_num) / denominator,
                    err[1] * f32::from(tap.weight_num) / denominator,
                    err[2] * f32::from(tap.weight_num) / denominator,
                ];
                add_error_to_pixel(&mut errors, width, height, nx, ny, channels, distributed);
            }
        }
    }

    Ok(())
}

fn integer_sample_max<S: Sample>() -> Result<i32> {
    if S::IS_FLOAT {
        return Err(Error::UnsupportedFormat(
            "integer diffusion path does not support floating-point samples",
        ));
    }

    match size_of::<S>() {
        1 => Ok(255),
        2 => Ok(65_535),
        _ => Err(Error::UnsupportedFormat(
            "unsupported integer sample width for diffusion core",
        )),
    }
}

fn sample_to_domain<S: Sample>(value: S, max_value: i32) -> i32 {
    (value.to_unit_f32().clamp(0.0, 1.0) * max_value as f32).round() as i32
}

fn domain_to_sample<S: Sample>(value: i32, max_value: i32) -> S {
    S::from_unit_f32(value.clamp(0, max_value) as f32 / max_value as f32)
}
