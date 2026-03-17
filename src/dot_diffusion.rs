use crate::{
    math::{
        color::luma_u8,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, DithrError, DithrResult, PixelFormat, QuantizeMode,
};

const CLASS_MATRIX_W: usize = 4;
const CLASS_MATRIX_H: usize = 4;
const CLASS_COUNT: usize = CLASS_MATRIX_W * CLASS_MATRIX_H;
const CLASS_MATRIX: [[u8; CLASS_MATRIX_W]; CLASS_MATRIX_H] =
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

pub fn knuth_dot_diffusion_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
) -> DithrResult<()> {
    buffer.validate()?;
    match buffer.format {
        PixelFormat::Gray8 => dot_diffuse_gray(buffer, mode),
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => dot_diffuse_rgb(buffer, mode),
    }
}

fn dot_diffuse_gray(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(DithrError::InvalidArgument("image dimensions overflow"))?;
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
                let quantized = quantize_pixel(PixelFormat::Gray8, &[adjusted], mode);
                let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
                *value = quantized_gray;

                let err = i32::from(adjusted) - i32::from(quantized_gray);
                diffuse_gray_to_higher_classes(&mut errors, width, height, x, y, class, err);
            }
        }
    }

    Ok(())
}

fn dot_diffuse_rgb(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(DithrError::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(DithrError::InvalidArgument("error buffer size overflow"))?;
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
                let pixel = if format == PixelFormat::Rgba8 {
                    [adjusted[0], adjusted[1], adjusted[2], alpha]
                } else {
                    [adjusted[0], adjusted[1], adjusted[2], 255]
                };
                let quantized = quantize_pixel(format, &pixel, mode);

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
                diffuse_rgb_to_higher_classes(&mut errors, width, height, x, y, class, err);
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
) {
    let targets = higher_class_neighbors(width, height, x, y, class);
    if targets.is_empty() {
        return;
    }
    let share = i32::from(clamp_i16(err / targets.len() as i32));

    for (nx, ny) in targets {
        let idx = ny * width + nx;
        errors[idx] += share;
    }
}

fn diffuse_rgb_to_higher_classes(
    errors: &mut [i32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
    err: [i32; 3],
) {
    let targets = higher_class_neighbors(width, height, x, y, class);
    if targets.is_empty() {
        return;
    }

    let count = targets.len() as i32;
    let share = [
        i32::from(clamp_i16(err[0] / count)),
        i32::from(clamp_i16(err[1] / count)),
        i32::from(clamp_i16(err[2] / count)),
    ];

    for (nx, ny) in targets {
        let base = (ny * width + nx) * 3;
        errors[base] += share[0];
        errors[base + 1] += share[1];
        errors[base + 2] += share[2];
    }
}

fn higher_class_neighbors(
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    class: u8,
) -> Vec<(usize, usize)> {
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
            if class_at(nx, ny) > class {
                targets.push((nx, ny));
            }
        }
    }

    targets
}

fn class_at(x: usize, y: usize) -> u8 {
    CLASS_MATRIX[y % CLASS_MATRIX_H][x % CLASS_MATRIX_W]
}
