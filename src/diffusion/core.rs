use crate::{
    core::{alpha_index, read_unit_pixel, PixelLayout, Sample},
    data::ErrorKernel,
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};

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

    if L::COLOR_CHANNELS == 1 && !L::HAS_ALPHA {
        return diffuse_gray_row_major(buffer, mode, kernel);
    }

    if L::COLOR_CHANNELS == 3 && (L::CHANNELS == 3 || L::CHANNELS == 4) {
        return diffuse_rgb_row_major(buffer, mode, kernel);
    }

    Err(Error::UnsupportedFormat(
        "error diffusion core supports Gray, Rgb, and Rgba formats only",
    ))
}

pub(crate) fn read_pixel_with_error<S: Sample, L: PixelLayout>(
    pixel: &[S],
    err: &[f32; 4],
) -> [f32; 4] {
    let mut rgba = read_unit_pixel::<S, L>(pixel);

    for channel in 0..L::COLOR_CHANNELS {
        rgba[channel] = (rgba[channel] + err[channel]).clamp(0.0, 1.0);
    }

    if L::COLOR_CHANNELS == 1 {
        rgba[1] = rgba[0];
        rgba[2] = rgba[0];
    }

    rgba
}

pub(crate) fn write_quantized_pixel<S: Sample, L: PixelLayout>(pixel: &mut [S], quantized: [S; 4]) {
    let preserved_alpha = alpha_index::<L>().and_then(|idx| pixel.get(idx).copied());
    pixel[..L::COLOR_CHANNELS].copy_from_slice(&quantized[..L::COLOR_CHANNELS]);

    if let Some(alpha_lane) = alpha_index::<L>() {
        if let Some(alpha) = preserved_alpha {
            pixel[alpha_lane] = alpha;
        }
    }
}

pub(crate) fn diffuse_error_forward<L: PixelLayout>(
    errors: &mut [f32],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    delta: [f32; 4],
) {
    if x < 0 || y < 0 {
        return;
    }

    let x = x as usize;
    let y = y as usize;
    if x >= width || y >= height {
        return;
    }

    let base = (y * width + x) * 4;
    for channel in 0..L::COLOR_CHANNELS {
        errors[base + channel] += delta[channel];
    }
}

pub(crate) fn diffuse_gray_row_major<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    if L::COLOR_CHANNELS != 1 || L::HAS_ALPHA {
        return Err(Error::UnsupportedFormat(
            "error diffusion gray path supports Gray formats only",
        ));
    }

    diffuse_row_major(buffer, mode, kernel)
}

pub(crate) fn diffuse_rgb_row_major<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    if !(L::COLOR_CHANNELS == 3 && (L::CHANNELS == 3 || L::CHANNELS == 4)) {
        return Err(Error::UnsupportedFormat(
            "error diffusion rgb path supports Rgb and Rgba formats only",
        ));
    }

    diffuse_row_major(buffer, mode, kernel)
}

fn diffuse_row_major<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let sample_channels = L::CHANNELS;
    let denominator = f32::from(kernel.weight_den);
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let offset = x
                .checked_mul(sample_channels)
                .ok_or(BufferError::OutOfBounds)?;
            let end = offset
                .checked_add(sample_channels)
                .ok_or(BufferError::OutOfBounds)?;
            let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
            let err_idx = (y * width + x) * 4;
            let err = [
                errors[err_idx],
                errors[err_idx + 1],
                errors[err_idx + 2],
                errors[err_idx + 3],
            ];
            let adjusted_unit = read_pixel_with_error::<S, L>(pixel, &err);
            let adjusted = [
                S::from_unit_f32(adjusted_unit[0]),
                S::from_unit_f32(adjusted_unit[1]),
                S::from_unit_f32(adjusted_unit[2]),
                S::from_unit_f32(adjusted_unit[3]),
            ];
            let quantized = quantize_pixel::<S, L>(&adjusted[..sample_channels], mode)?;
            let quantized_unit = read_unit_pixel::<S, L>(&quantized[..sample_channels]);
            write_quantized_pixel::<S, L>(pixel, quantized);

            let residual = [
                adjusted_unit[0] - quantized_unit[0],
                adjusted_unit[1] - quantized_unit[1],
                adjusted_unit[2] - quantized_unit[2],
                0.0,
            ];

            for tap in kernel.taps {
                let weight = f32::from(tap.weight_num) / denominator;
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize + isize::from(tap.dx),
                    y as isize + isize::from(tap.dy),
                    [
                        residual[0] * weight,
                        residual[1] * weight,
                        residual[2] * weight,
                        0.0,
                    ],
                );
            }
        }
    }

    Ok(())
}
