use super::core::{diffuse_error_forward, read_pixel_with_error, write_quantized_pixel};
use crate::{
    core::{PixelLayout, Sample},
    data::{OSTROMOUKHOV_COEFFS, ZHOU_FANG_MODULATION},
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};

#[derive(Clone, Copy)]
enum GrayOnlyVariableAlgorithm {
    Ostromoukhov,
    ZhouFang,
    GradientBased,
}

pub fn ostromoukhov_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::Ostromoukhov)
}

pub fn zhou_fang_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::ZhouFang)
}

pub fn gradient_based_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::GradientBased)
}

fn diffuse_gray_only_variable<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    algorithm: GrayOnlyVariableAlgorithm,
) -> Result<()> {
    buffer.validate()?;

    if L::CHANNELS != 1 || L::HAS_ALPHA {
        return Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only",
        ));
    }

    match algorithm {
        GrayOnlyVariableAlgorithm::Ostromoukhov => diffuse_variable_gray(buffer, mode, None),
        GrayOnlyVariableAlgorithm::ZhouFang => {
            diffuse_variable_gray(buffer, mode, Some(&ZHOU_FANG_MODULATION))
        }
        GrayOnlyVariableAlgorithm::GradientBased => diffuse_gradient_gray(buffer, mode),
    }
}

fn diffuse_gradient_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let stride = buffer.stride;
    let source = buffer.data.to_vec();
    let mut errors = allocate_gray_error_buffer(width, height)?;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let pixel = row.get_mut(x..=x).ok_or(BufferError::OutOfBounds)?;
            let err_idx = (y * width + x) * 4;
            let err = [
                errors[err_idx],
                errors[err_idx + 1],
                errors[err_idx + 2],
                errors[err_idx + 3],
            ];
            let adjusted = read_pixel_with_error::<S, L>(pixel, &err)[0];
            let quantized =
                quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
            write_quantized_pixel::<S, L>(pixel, quantized);
            let residual = adjusted - quantized[0].to_unit_f32();

            let gradient = local_gradient_unit(&source, x, y, width, height, stride);
            let scale = 8.0 + (255.0 - f32::from(gradient)) * 8.0 / 255.0;
            let right = residual * (7.0 * scale) / (16.0 * 16.0);
            let down_left = residual * (3.0 * scale) / (16.0 * 16.0);
            let down = residual * (5.0 * scale) / (16.0 * 16.0);
            let down_right = residual * scale / (16.0 * 16.0);

            diffuse_error_forward::<L>(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize,
                [right, 0.0, 0.0, 0.0],
            );
            diffuse_error_forward::<L>(
                &mut errors,
                width,
                height,
                x as isize - 1,
                y as isize + 1,
                [down_left, 0.0, 0.0, 0.0],
            );
            diffuse_error_forward::<L>(
                &mut errors,
                width,
                height,
                x as isize,
                y as isize + 1,
                [down, 0.0, 0.0, 0.0],
            );
            diffuse_error_forward::<L>(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize + 1,
                [down_right, 0.0, 0.0, 0.0],
            );
        }
    }

    Ok(())
}

fn diffuse_variable_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    modulation: Option<&[u8; 256]>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let mut errors = allocate_gray_error_buffer(width, height)?;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let reverse = (y & 1) == 1;

        if reverse {
            for x in (0..width).rev() {
                let pixel = row.get_mut(x..=x).ok_or(BufferError::OutOfBounds)?;
                let err_idx = (y * width + x) * 4;
                let err = [
                    errors[err_idx],
                    errors[err_idx + 1],
                    errors[err_idx + 2],
                    errors[err_idx + 3],
                ];
                let adjusted = read_pixel_with_error::<S, L>(pixel, &err)[0];
                let thresholded = if let Some(table) = modulation {
                    zhou_fang_thresholded_unit(adjusted, x, y, table)
                } else {
                    adjusted
                };
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(thresholded)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = adjusted - quantized[0].to_unit_f32();
                let coeff = coefficient_for_luma(luma_bucket_unit(adjusted));
                let den = f32::from(coeff.3);
                let forward = residual * f32::from(coeff.0) / den;
                let down_diag = residual * f32::from(coeff.1) / den;
                let down = residual * f32::from(coeff.2) / den;
                let xi = x as isize;
                let yi = y as isize;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi,
                    [forward, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    [down_diag, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    [down, 0.0, 0.0, 0.0],
                );
            }
        } else {
            for x in 0..width {
                let pixel = row.get_mut(x..=x).ok_or(BufferError::OutOfBounds)?;
                let err_idx = (y * width + x) * 4;
                let err = [
                    errors[err_idx],
                    errors[err_idx + 1],
                    errors[err_idx + 2],
                    errors[err_idx + 3],
                ];
                let adjusted = read_pixel_with_error::<S, L>(pixel, &err)[0];
                let thresholded = if let Some(table) = modulation {
                    zhou_fang_thresholded_unit(adjusted, x, y, table)
                } else {
                    adjusted
                };
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(thresholded)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = adjusted - quantized[0].to_unit_f32();
                let coeff = coefficient_for_luma(luma_bucket_unit(adjusted));
                let den = f32::from(coeff.3);
                let forward = residual * f32::from(coeff.0) / den;
                let down_diag = residual * f32::from(coeff.1) / den;
                let down = residual * f32::from(coeff.2) / den;
                let xi = x as isize;
                let yi = y as isize;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi,
                    [forward, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    [down_diag, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    [down, 0.0, 0.0, 0.0],
                );
            }
        }
    }

    Ok(())
}

fn allocate_gray_error_buffer(width: usize, height: usize) -> Result<Vec<f32>> {
    let len = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(4))
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    Ok(vec![0.0_f32; len])
}

fn coefficient_for_luma(luma: u8) -> (i16, i16, i16, i16) {
    OSTROMOUKHOV_COEFFS[usize::from(luma)]
}

fn modulation_for_luma(luma: u8, table: &[u8; 256]) -> u8 {
    table[usize::from(luma)]
}

fn zhou_fang_thresholded_unit(value_unit: f32, x: usize, y: usize, table: &[u8; 256]) -> f32 {
    let luma = luma_bucket_unit(value_unit);
    let amplitude = f32::from(modulation_for_luma(luma, table)) / 255.0;
    let noise = zhou_fang_noise_unit(x, y);
    (value_unit + (noise - 0.5) * amplitude).clamp(0.0, 1.0)
}

fn zhou_fang_noise_unit(x: usize, y: usize) -> f32 {
    let mut state = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ 0x94D0_49BB_1331_11EB_u64;
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^= state >> 31;

    (state >> 40) as f32 / 16_777_215.0
}

fn local_gradient_unit<S: Sample>(
    source: &[S],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
) -> u8 {
    let center = sample_to_byte(source[y * stride + x]);
    let left = if x > 0 {
        sample_to_byte(source[y * stride + (x - 1)])
    } else {
        center
    };
    let right = if x + 1 < width {
        sample_to_byte(source[y * stride + (x + 1)])
    } else {
        center
    };
    let up = if y > 0 {
        sample_to_byte(source[(y - 1) * stride + x])
    } else {
        center
    };
    let down = if y + 1 < height {
        sample_to_byte(source[(y + 1) * stride + x])
    } else {
        center
    };

    let gx = i32::from(right) - i32::from(left);
    let gy = i32::from(down) - i32::from(up);
    let magnitude = gx.abs() + gy.abs();
    (magnitude / 2).clamp(0, 255) as u8
}

fn sample_to_byte<S: Sample>(value: S) -> u8 {
    (value.to_unit_f32().clamp(0.0, 1.0) * 255.0).round() as u8
}

fn luma_bucket_unit(value_unit: f32) -> u8 {
    (value_unit.clamp(0.0, 1.0) * 255.0).round() as u8
}
