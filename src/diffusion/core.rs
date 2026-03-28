use crate::{
    data::ErrorKernel,
    math::{
        color::luma_u8,
        fixed::mul_div_i32,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, Error, PixelFormat, QuantizeMode, Result,
};

pub(crate) fn error_diffuse_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    kernel: &ErrorKernel,
) -> Result<()> {
    buffer.validate()?;
    if kernel.weight_den <= 0 {
        return Err(Error::InvalidArgument(
            "kernel denominator must be positive",
        ));
    }

    match buffer.format {
        PixelFormat::Gray8 => diffuse_gray_row_major(buffer, mode, kernel),
        PixelFormat::Rgb8 | PixelFormat::Rgba8 => diffuse_rgb_row_major(buffer, mode, kernel),
    }
}

pub(crate) fn add_error_to_pixel(
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

pub(crate) fn diffuse_gray_row_major(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
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
            let original = *value;
            let adjusted = clamp_u8(i32::from(original) + errors[idx]);
            let quantized = quantize_pixel(PixelFormat::Gray8, &[adjusted], mode)?;
            let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
            *value = quantized_gray;

            let err = i32::from(adjusted) - i32::from(quantized_gray);

            for tap in kernel.taps {
                let nx = x as isize + isize::from(tap.dx);
                let ny = y as isize + isize::from(tap.dy);
                let distributed = mul_div_i32(err, i32::from(tap.weight_num), denominator);
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    nx,
                    ny,
                    1,
                    [i32::from(clamp_i16(distributed)), 0, 0],
                );
            }
        }
    }

    Ok(())
}

pub(crate) fn diffuse_rgb_row_major(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let denominator = i32::from(kernel.weight_den);
    let bytes_per_pixel = format.bytes_per_pixel();
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0_i32; error_len];
    let is_rgba = format == PixelFormat::Rgba8;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        let row_error_base = y * width * channels;

        for x in 0..width {
            let offset = x * bytes_per_pixel;
            let error_base = row_error_base + x * channels;
            let adjusted = [
                clamp_u8(i32::from(row[offset]) + errors[error_base]),
                clamp_u8(i32::from(row[offset + 1]) + errors[error_base + 1]),
                clamp_u8(i32::from(row[offset + 2]) + errors[error_base + 2]),
            ];
            let alpha = if is_rgba { row[offset + 3] } else { 255 };
            let quantized = if is_rgba {
                let pixel = [adjusted[0], adjusted[1], adjusted[2], alpha];
                quantize_pixel(PixelFormat::Rgba8, &pixel, mode)?
            } else {
                quantize_pixel(PixelFormat::Rgb8, &adjusted, mode)?
            };

            row[offset] = quantized[0];
            row[offset + 1] = quantized[1];
            row[offset + 2] = quantized[2];
            if is_rgba {
                row[offset + 3] = alpha;
            }

            let err = [
                i32::from(adjusted[0]) - i32::from(quantized[0]),
                i32::from(adjusted[1]) - i32::from(quantized[1]),
                i32::from(adjusted[2]) - i32::from(quantized[2]),
            ];

            for tap in kernel.taps {
                let nx = x as isize + isize::from(tap.dx);
                let ny = y as isize + isize::from(tap.dy);
                let distributed = [
                    mul_div_i32(err[0], i32::from(tap.weight_num), denominator),
                    mul_div_i32(err[1], i32::from(tap.weight_num), denominator),
                    mul_div_i32(err[2], i32::from(tap.weight_num), denominator),
                ];
                add_error_to_pixel(
                    &mut errors,
                    width,
                    height,
                    nx,
                    ny,
                    channels,
                    [
                        i32::from(clamp_i16(distributed[0])),
                        i32::from(clamp_i16(distributed[1])),
                        i32::from(clamp_i16(distributed[2])),
                    ],
                );
            }
        }
    }

    Ok(())
}
