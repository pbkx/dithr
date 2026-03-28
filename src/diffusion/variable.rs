use super::core::add_error_to_pixel;
use crate::{
    data::{OSTROMOUKHOV_COEFFS, ZHOU_FANG_MODULATION},
    math::{
        color::luma_u8,
        fixed::mul_div_i32,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, Error, PixelFormat, QuantizeMode, Result,
};

#[derive(Clone, Copy)]
enum GrayOnlyVariableAlgorithm {
    Ostromoukhov,
    ZhouFang,
    GradientBased,
}

pub fn ostromoukhov_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::Ostromoukhov)
}

pub fn zhou_fang_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::ZhouFang)
}

pub fn gradient_based_error_diffusion_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::GradientBased)
}

fn diffuse_gray_only_variable(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    algorithm: GrayOnlyVariableAlgorithm,
) -> Result<()> {
    buffer.validate()?;

    if buffer.format != PixelFormat::Gray8 {
        return Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support Gray8 only",
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

fn diffuse_gradient_gray(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let stride = buffer.stride;
    let source = buffer.data.to_vec();
    let mut errors = allocate_gray_error_buffer(width, height)?;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let original = *value;
            let adjusted = clamp_u8(i32::from(original) + errors[idx]);
            let quantized = quantize_pixel(PixelFormat::Gray8, &[adjusted], mode)?;
            let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
            *value = quantized_gray;

            let gradient = local_gradient(&source, x, y, width, height, stride);
            let scale_num = 8 + ((255 - i32::from(gradient)) * 8) / 255;
            let err = i32::from(adjusted) - i32::from(quantized_gray);

            let right = mul_div_i32(err, 7 * scale_num, 16 * 16);
            let down_left = mul_div_i32(err, 3 * scale_num, 16 * 16);
            let down = mul_div_i32(err, 5 * scale_num, 16 * 16);
            let down_right = mul_div_i32(err, scale_num, 16 * 16);

            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize,
                1,
                [i32::from(clamp_i16(right)), 0, 0],
            );
            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize - 1,
                y as isize + 1,
                1,
                [i32::from(clamp_i16(down_left)), 0, 0],
            );
            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize,
                y as isize + 1,
                1,
                [i32::from(clamp_i16(down)), 0, 0],
            );
            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize + 1,
                1,
                [i32::from(clamp_i16(down_right)), 0, 0],
            );
        }
    }

    Ok(())
}

fn diffuse_variable_gray(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    modulation: Option<&[i16; 16]>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let mut errors = allocate_gray_error_buffer(width, height)?;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let reverse = (y & 1) == 1;

        if reverse {
            let mut x = width;
            while x > 0 {
                x -= 1;
                let idx = y * width + x;
                let original = row[x];
                let adjusted = clamp_u8(i32::from(original) + errors[idx]);
                let thresholded = if let Some(table) = modulation {
                    clamp_u8(i32::from(adjusted) + i32::from(modulation_for_luma(adjusted, table)))
                } else {
                    adjusted
                };
                let quantized = quantize_pixel(PixelFormat::Gray8, &[thresholded], mode)?;
                let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
                row[x] = quantized_gray;

                let err = i32::from(adjusted) - i32::from(quantized_gray);
                let coeff = coefficient_for_luma(adjusted);
                let den = i32::from(coeff.3);
                let forward = mul_div_i32(err, i32::from(coeff.0), den);
                let down_diag = mul_div_i32(err, i32::from(coeff.1), den);
                let down = mul_div_i32(err, i32::from(coeff.2), den);
                let xi = x as isize;
                let yi = y as isize;

                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi,
                    1,
                    [i32::from(clamp_i16(forward)), 0, 0],
                );
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    1,
                    [i32::from(clamp_i16(down_diag)), 0, 0],
                );
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    1,
                    [i32::from(clamp_i16(down)), 0, 0],
                );
            }
        } else {
            let mut x = 0;
            while x < width {
                let idx = y * width + x;
                let original = row[x];
                let adjusted = clamp_u8(i32::from(original) + errors[idx]);
                let thresholded = if let Some(table) = modulation {
                    clamp_u8(i32::from(adjusted) + i32::from(modulation_for_luma(adjusted, table)))
                } else {
                    adjusted
                };
                let quantized = quantize_pixel(PixelFormat::Gray8, &[thresholded], mode)?;
                let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
                row[x] = quantized_gray;

                let err = i32::from(adjusted) - i32::from(quantized_gray);
                let coeff = coefficient_for_luma(adjusted);
                let den = i32::from(coeff.3);
                let forward = mul_div_i32(err, i32::from(coeff.0), den);
                let down_diag = mul_div_i32(err, i32::from(coeff.1), den);
                let down = mul_div_i32(err, i32::from(coeff.2), den);
                let xi = x as isize;
                let yi = y as isize;

                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi,
                    1,
                    [i32::from(clamp_i16(forward)), 0, 0],
                );
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    1,
                    [i32::from(clamp_i16(down_diag)), 0, 0],
                );
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    1,
                    [i32::from(clamp_i16(down)), 0, 0],
                );
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

fn coefficient_for_luma(luma: u8) -> (i16, i16, i16, i16) {
    OSTROMOUKHOV_COEFFS[usize::from(luma)]
}

fn modulation_for_luma(luma: u8, table: &[i16; 16]) -> i16 {
    let idx = (usize::from(luma) * table.len()) / 256;
    table[idx]
}

fn local_gradient(
    source: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    stride: usize,
) -> u8 {
    let center = source[y * stride + x];
    let left = if x > 0 {
        source[y * stride + (x - 1)]
    } else {
        center
    };
    let right = if x + 1 < width {
        source[y * stride + (x + 1)]
    } else {
        center
    };
    let up = if y > 0 {
        source[(y - 1) * stride + x]
    } else {
        center
    };
    let down = if y + 1 < height {
        source[(y + 1) * stride + x]
    } else {
        center
    };

    let gx = i32::from(right) - i32::from(left);
    let gy = i32::from(down) - i32::from(up);
    let magnitude = gx.abs() + gy.abs();

    clamp_u8(magnitude / 2)
}
