use crate::{
    math::{
        color::luma_u8,
        fixed::mul_div_i32,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, Error, PixelFormat, QuantizeMode, Result,
};

const CLASS_MATRIX_W: usize = 4;
const CLASS_MATRIX_H: usize = 4;
const CLASS_COUNT: usize = CLASS_MATRIX_W * CLASS_MATRIX_H;
const CLASS_MATRIX: [[u8; CLASS_MATRIX_W]; CLASS_MATRIX_H] =
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
const DIFFUSION_WEIGHTS_3X3: [[i16; 3]; 3] = [[1, 2, 1], [2, 0, 2], [1, 2, 1]];

pub fn knuth_dot_diffusion_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    buffer.validate()?;
    match buffer.format {
        PixelFormat::Gray8 => dot_diffuse_gray(buffer, mode),
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => dot_diffuse_rgb(buffer, mode),
    }
}

fn dot_diffuse_gray(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut errors = vec![0_i32; pixel_count];

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for (x, value) in row.iter_mut().take(width).enumerate() {
                if class_at(x, y) != class {
                    continue;
                }

                let idx = y * width + x;

                let adjusted = clamp_u8(i32::from(*value) + errors[idx]);
                let quantized = quantize_pixel(PixelFormat::Gray8, &[adjusted], mode)?;
                let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
                *value = quantized_gray;

                let err = i32::from(adjusted) - i32::from(quantized_gray);
                diffuse_gray_to_higher_classes(&mut errors, width, height, x, y, class, err)?;
            }
        }
    }

    Ok(())
}

fn dot_diffuse_rgb(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0_i32; error_len];

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for x in 0..width {
                if class_at(x, y) != class {
                    continue;
                }

                let offset = x * bpp;
                let base = (y * width + x) * channels;

                let adjusted = [
                    clamp_u8(i32::from(row[offset]) + errors[base]),
                    clamp_u8(i32::from(row[offset + 1]) + errors[base + 1]),
                    clamp_u8(i32::from(row[offset + 2]) + errors[base + 2]),
                ];
                let alpha = if format == PixelFormat::Rgba8 {
                    row[offset + 3]
                } else {
                    255
                };
                let quantized = if format == PixelFormat::Rgba8 {
                    let pixel = [adjusted[0], adjusted[1], adjusted[2], alpha];
                    quantize_pixel(PixelFormat::Rgba8, &pixel, mode)?
                } else {
                    quantize_pixel(PixelFormat::Rgb8, &adjusted, mode)?
                };

                row[offset] = quantized[0];
                row[offset + 1] = quantized[1];
                row[offset + 2] = quantized[2];
                if format == PixelFormat::Rgba8 {
                    row[offset + 3] = alpha;
                }

                let err = [
                    i32::from(adjusted[0]) - i32::from(quantized[0]),
                    i32::from(adjusted[1]) - i32::from(quantized[1]),
                    i32::from(adjusted[2]) - i32::from(quantized[2]),
                ];
                diffuse_rgb_to_higher_classes(&mut errors, width, height, x, y, class, err)?;
            }
        }
    }

    Ok(())
}

fn diffuse_gray_to_higher_classes(
    errors: &mut [i32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
    err: i32,
) -> Result<()> {
    let targets = higher_class_neighbors(width, height, x, y, class);
    if targets.is_empty() {
        return Ok(());
    }
    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| i32::from(weight))
        .sum();

    for (nx, ny, weight) in targets {
        let distributed = mul_div_i32(err, i32::from(weight), total_weight)?;
        let share = i32::from(clamp_i16(distributed));
        let idx = ny * width + nx;
        errors[idx] += share;
    }

    Ok(())
}

fn diffuse_rgb_to_higher_classes(
    errors: &mut [i32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
    err: [i32; 3],
) -> Result<()> {
    let targets = higher_class_neighbors(width, height, x, y, class);
    if targets.is_empty() {
        return Ok(());
    }

    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| i32::from(weight))
        .sum();

    for (nx, ny, weight) in targets {
        let share = [
            i32::from(clamp_i16(mul_div_i32(
                err[0],
                i32::from(weight),
                total_weight,
            )?)),
            i32::from(clamp_i16(mul_div_i32(
                err[1],
                i32::from(weight),
                total_weight,
            )?)),
            i32::from(clamp_i16(mul_div_i32(
                err[2],
                i32::from(weight),
                total_weight,
            )?)),
        ];
        let base = (ny * width + nx) * 3;
        errors[base] += share[0];
        errors[base + 1] += share[1];
        errors[base + 2] += share[2];
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
