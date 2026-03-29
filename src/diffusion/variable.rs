use crate::{
    core::{PixelLayout, Sample},
    data::{OSTROMOUKHOV_COEFFS, ZHOU_FANG_MODULATION},
    math::fixed::mul_div_i32,
    quantize_pixel, Buffer, Error, QuantizeMode, Result,
};
use std::mem::size_of;

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
    let max_value = integer_sample_max::<S>()?;
    let source = buffer.data.to_vec();
    let mut errors = allocate_gray_error_buffer(width, height)?;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let adjusted = (sample_to_domain(*value, max_value) + errors[idx]).clamp(0, max_value);
            let adjusted_sample = domain_to_sample::<S>(adjusted, max_value);
            let quantized = quantize_pixel::<S, crate::core::Gray>(&[adjusted_sample], mode)?;
            let quantized_gray = quantized[0];
            *value = quantized_gray;

            let gradient = local_gradient(&source, x, y, width, height, stride, max_value);
            let scale_num = 8 + ((255 - i32::from(gradient)) * 8) / 255;
            let err = adjusted - sample_to_domain(quantized_gray, max_value);

            let right = mul_div_i32(err, 7 * scale_num, 16 * 16)?;
            let down_left = mul_div_i32(err, 3 * scale_num, 16 * 16)?;
            let down = mul_div_i32(err, 5 * scale_num, 16 * 16)?;
            let down_right = mul_div_i32(err, scale_num, 16 * 16)?;

            add_error_to_pixel_i32(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize,
                1,
                [right, 0, 0],
            );
            add_error_to_pixel_i32(
                &mut errors,
                width,
                height,
                x as isize - 1,
                y as isize + 1,
                1,
                [down_left, 0, 0],
            );
            add_error_to_pixel_i32(
                &mut errors,
                width,
                height,
                x as isize,
                y as isize + 1,
                1,
                [down, 0, 0],
            );
            add_error_to_pixel_i32(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize + 1,
                1,
                [down_right, 0, 0],
            );
        }
    }

    Ok(())
}

fn diffuse_variable_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    modulation: Option<&[i16; 16]>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let max_value = integer_sample_max::<S>()?;
    let mut errors = allocate_gray_error_buffer(width, height)?;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let reverse = (y & 1) == 1;

        if reverse {
            let mut x = width;
            while x > 0 {
                x -= 1;
                let idx = y * width + x;
                let adjusted =
                    (sample_to_domain(row[x], max_value) + errors[idx]).clamp(0, max_value);
                let thresholded = if let Some(table) = modulation {
                    (adjusted + scaled_modulation(adjusted, max_value, table)).clamp(0, max_value)
                } else {
                    adjusted
                };
                let thresholded_sample = domain_to_sample::<S>(thresholded, max_value);
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[thresholded_sample], mode)?;
                let quantized_gray = quantized[0];
                row[x] = quantized_gray;

                let err = adjusted - sample_to_domain(quantized_gray, max_value);
                let coeff = coefficient_for_luma(luma_bucket(adjusted, max_value));
                let den = i32::from(coeff.3);
                let forward = mul_div_i32(err, i32::from(coeff.0), den)?;
                let down_diag = mul_div_i32(err, i32::from(coeff.1), den)?;
                let down = mul_div_i32(err, i32::from(coeff.2), den)?;
                let xi = x as isize;
                let yi = y as isize;

                add_error_to_pixel_i32(&mut errors, width, height, xi - 1, yi, 1, [forward, 0, 0]);
                add_error_to_pixel_i32(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    1,
                    [down_diag, 0, 0],
                );
                add_error_to_pixel_i32(&mut errors, width, height, xi, yi + 1, 1, [down, 0, 0]);
            }
        } else {
            let mut x = 0;
            while x < width {
                let idx = y * width + x;
                let adjusted =
                    (sample_to_domain(row[x], max_value) + errors[idx]).clamp(0, max_value);
                let thresholded = if let Some(table) = modulation {
                    (adjusted + scaled_modulation(adjusted, max_value, table)).clamp(0, max_value)
                } else {
                    adjusted
                };
                let thresholded_sample = domain_to_sample::<S>(thresholded, max_value);
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[thresholded_sample], mode)?;
                let quantized_gray = quantized[0];
                row[x] = quantized_gray;

                let err = adjusted - sample_to_domain(quantized_gray, max_value);
                let coeff = coefficient_for_luma(luma_bucket(adjusted, max_value));
                let den = i32::from(coeff.3);
                let forward = mul_div_i32(err, i32::from(coeff.0), den)?;
                let down_diag = mul_div_i32(err, i32::from(coeff.1), den)?;
                let down = mul_div_i32(err, i32::from(coeff.2), den)?;
                let xi = x as isize;
                let yi = y as isize;

                add_error_to_pixel_i32(&mut errors, width, height, xi + 1, yi, 1, [forward, 0, 0]);
                add_error_to_pixel_i32(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    1,
                    [down_diag, 0, 0],
                );
                add_error_to_pixel_i32(&mut errors, width, height, xi, yi + 1, 1, [down, 0, 0]);
                x += 1;
            }
        }
    }

    Ok(())
}

fn allocate_gray_error_buffer(width: usize, height: usize) -> Result<Vec<i32>> {
    let len = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    Ok(vec![0_i32; len])
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

fn coefficient_for_luma(luma: u8) -> (i16, i16, i16, i16) {
    OSTROMOUKHOV_COEFFS[usize::from(luma)]
}

fn modulation_for_luma(luma: u8, table: &[i16; 16]) -> i16 {
    let idx = (usize::from(luma) * table.len()) / 256;
    table[idx]
}

fn scaled_modulation(value: i32, max_value: i32, table: &[i16; 16]) -> i32 {
    let modulation = i32::from(modulation_for_luma(luma_bucket(value, max_value), table));
    if max_value == 255 {
        modulation
    } else {
        let scaled = i64::from(modulation) * i64::from(max_value);
        (scaled / 255_i64) as i32
    }
}

fn local_gradient<S: Sample>(
    source: &[S],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
    max_value: i32,
) -> u8 {
    let center = sample_to_domain(source[y * stride + x], max_value);
    let left = if x > 0 {
        sample_to_domain(source[y * stride + (x - 1)], max_value)
    } else {
        center
    };
    let right = if x + 1 < width {
        sample_to_domain(source[y * stride + (x + 1)], max_value)
    } else {
        center
    };
    let up = if y > 0 {
        sample_to_domain(source[(y - 1) * stride + x], max_value)
    } else {
        center
    };
    let down = if y + 1 < height {
        sample_to_domain(source[(y + 1) * stride + x], max_value)
    } else {
        center
    };

    let gx = right - left;
    let gy = down - up;
    let magnitude = gx.abs() + gy.abs();

    if max_value == 255 {
        (magnitude / 2).clamp(0, 255) as u8
    } else {
        let numer = i64::from(magnitude) * 255_i64;
        let denom = i64::from(max_value) * 2_i64;
        ((numer / denom) as i32).clamp(0, 255) as u8
    }
}

fn integer_sample_max<S: Sample>() -> Result<i32> {
    if S::IS_FLOAT {
        return Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support integer grayscale formats only",
        ));
    }

    match size_of::<S>() {
        1 => Ok(255),
        2 => Ok(65_535),
        _ => Err(Error::UnsupportedFormat(
            "unsupported integer sample width for variable diffusion",
        )),
    }
}

fn sample_to_domain<S: Sample>(value: S, max_value: i32) -> i32 {
    (value.to_unit_f32().clamp(0.0, 1.0) * max_value as f32).round() as i32
}

fn domain_to_sample<S: Sample>(value: i32, max_value: i32) -> S {
    S::from_unit_f32(value.clamp(0, max_value) as f32 / max_value as f32)
}

fn luma_bucket(value: i32, max_value: i32) -> u8 {
    if max_value == 255 {
        value.clamp(0, 255) as u8
    } else {
        let scaled = i64::from(value.clamp(0, max_value)) * 255_i64;
        (scaled / i64::from(max_value)) as u8
    }
}
