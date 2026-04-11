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
    Multiscale,
    FeaturePreservingMsed,
    GreenNoiseMsed,
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

pub fn multiscale_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::Multiscale)
}

pub fn feature_preserving_msed_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(
        buffer,
        mode,
        GrayOnlyVariableAlgorithm::FeaturePreservingMsed,
    )
}

pub fn green_noise_msed_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::GreenNoiseMsed)
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
        GrayOnlyVariableAlgorithm::Multiscale => diffuse_multiscale_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::FeaturePreservingMsed => {
            diffuse_feature_preserving_msed_gray(buffer, mode)
        }
        GrayOnlyVariableAlgorithm::GreenNoiseMsed => diffuse_green_noise_msed_gray(buffer, mode),
    }
}

#[derive(Clone, Copy)]
enum MultiscaleProfile {
    Baseline,
    FeaturePreserving,
    GreenNoise,
}

struct MultiscaleLevel {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

const GREEN_NOISE_BASE_SEED: u64 = 0xD1B5_4A32_7C6E_9F01;
const GREEN_NOISE_SMALL_RADIUS: usize = 1;
const GREEN_NOISE_LARGE_RADIUS: usize = 3;
const GREEN_NOISE_THRESHOLD_AMPLITUDE: f32 = 0.0625;
const GREEN_NOISE_DIFFUSION_AMPLITUDE: f32 = 0.2;

fn diffuse_multiscale_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_multiscale_gray_with_profile(buffer, mode, MultiscaleProfile::Baseline)
}

fn diffuse_feature_preserving_msed_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_multiscale_gray_with_profile(buffer, mode, MultiscaleProfile::FeaturePreserving)
}

fn diffuse_green_noise_msed_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_multiscale_gray_with_profile(buffer, mode, MultiscaleProfile::GreenNoise)
}

fn diffuse_multiscale_gray_with_profile<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    profile: MultiscaleProfile,
) -> Result<()> {
    let mut levels = build_multiscale_pyramid(buffer)?;
    let mut propagated_error: Option<MultiscaleLevel> = None;

    for level_index in (0..levels.len()).rev() {
        let level = levels
            .get_mut(level_index)
            .ok_or(Error::InvalidArgument("multiscale level indexing failed"))?;
        if let Some(coarser_error) = propagated_error.take() {
            add_upsampled_error(
                &mut level.data,
                level.width,
                level.height,
                &coarser_error.data,
                coarser_error.width,
                coarser_error.height,
            )?;
        }

        let residual = diffuse_multiscale_level::<S>(
            &mut level.data,
            level.width,
            level.height,
            mode,
            profile,
        )?;
        if level_index > 0 {
            propagated_error = Some(MultiscaleLevel {
                width: level.width,
                height: level.height,
                data: residual,
            });
        }
    }

    let finest = levels
        .first()
        .ok_or(Error::InvalidArgument("multiscale pyramid is empty"))?;
    write_multiscale_output(buffer, &finest.data)
}

fn build_multiscale_pyramid<S: Sample, L: PixelLayout>(
    buffer: &Buffer<'_, S, L>,
) -> Result<Vec<MultiscaleLevel>> {
    let mut levels = Vec::new();
    let mut current = MultiscaleLevel {
        width: buffer.width,
        height: buffer.height,
        data: read_gray_units(buffer)?,
    };
    levels.push(MultiscaleLevel {
        width: current.width,
        height: current.height,
        data: current.data.clone(),
    });

    while current.width > 1 || current.height > 1 {
        let next_width = current.width.div_ceil(2);
        let next_height = current.height.div_ceil(2);
        let next_data = downsample_gray_units(&current.data, current.width, current.height)?;
        current = MultiscaleLevel {
            width: next_width,
            height: next_height,
            data: next_data,
        };
        levels.push(MultiscaleLevel {
            width: current.width,
            height: current.height,
            data: current.data.clone(),
        });
    }

    Ok(levels)
}

fn read_gray_units<S: Sample, L: PixelLayout>(buffer: &Buffer<'_, S, L>) -> Result<Vec<f32>> {
    let len = checked_gray_len(buffer.width, buffer.height)?;
    let mut out = vec![0.0_f32; len];

    for y in 0..buffer.height {
        let row = buffer.try_row(y)?;
        for x in 0..buffer.width {
            let value = row.get(x).ok_or(BufferError::OutOfBounds)?;
            out[y * buffer.width + x] = value.to_unit_f32();
        }
    }

    Ok(out)
}

fn downsample_gray_units(source: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let next_width = width.div_ceil(2);
    let next_height = height.div_ceil(2);
    let len = checked_gray_len(next_width, next_height)?;
    let mut out = vec![0.0_f32; len];

    for y in 0..next_height {
        for x in 0..next_width {
            let x0 = x * 2;
            let y0 = y * 2;
            let mut sum = 0.0_f32;
            let mut count = 0_u32;

            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = x0 + dx;
                    let sy = y0 + dy;
                    if sx < width && sy < height {
                        sum += source[sy * width + sx];
                        count += 1;
                    }
                }
            }

            out[y * next_width + x] = if count > 0 { sum / count as f32 } else { 0.0 };
        }
    }

    Ok(out)
}

fn add_upsampled_error(
    target: &mut [f32],
    target_width: usize,
    target_height: usize,
    source: &[f32],
    source_width: usize,
    source_height: usize,
) -> Result<()> {
    let expected_target = checked_gray_len(target_width, target_height)?;
    let expected_source = checked_gray_len(source_width, source_height)?;
    if target.len() != expected_target || source.len() != expected_source {
        return Err(Error::InvalidArgument(
            "multiscale level shape does not match level buffers",
        ));
    }

    for y in 0..target_height {
        for x in 0..target_width {
            let sample = bilinear_sample(
                source,
                source_width,
                source_height,
                x,
                y,
                target_width,
                target_height,
            );
            let idx = y * target_width + x;
            target[idx] = (target[idx] + sample).clamp(0.0, 1.0);
        }
    }

    Ok(())
}

fn bilinear_sample(
    source: &[f32],
    source_width: usize,
    source_height: usize,
    x: usize,
    y: usize,
    target_width: usize,
    target_height: usize,
) -> f32 {
    let fx = if target_width > 1 && source_width > 1 {
        x as f32 * (source_width - 1) as f32 / (target_width - 1) as f32
    } else {
        0.0
    };
    let fy = if target_height > 1 && source_height > 1 {
        y as f32 * (source_height - 1) as f32 / (target_height - 1) as f32
    } else {
        0.0
    };

    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(source_width.saturating_sub(1));
    let y1 = (y0 + 1).min(source_height.saturating_sub(1));
    let tx = fx - x0 as f32;
    let ty = fy - y0 as f32;

    let p00 = source[y0 * source_width + x0];
    let p10 = source[y0 * source_width + x1];
    let p01 = source[y1 * source_width + x0];
    let p11 = source[y1 * source_width + x1];

    let top = p00 + (p10 - p00) * tx;
    let bottom = p01 + (p11 - p01) * tx;
    top + (bottom - top) * ty
}

fn diffuse_multiscale_level<S: Sample>(
    data: &mut [f32],
    width: usize,
    height: usize,
    mode: QuantizeMode<'_, S>,
    profile: MultiscaleProfile,
) -> Result<Vec<f32>> {
    let len = checked_gray_len(width, height)?;
    if data.len() != len {
        return Err(Error::InvalidArgument(
            "multiscale diffusion level buffer has invalid shape",
        ));
    }

    let features = if matches!(profile, MultiscaleProfile::FeaturePreserving) {
        Some(build_feature_map(data, width, height)?)
    } else {
        None
    };
    let green_noise = if matches!(profile, MultiscaleProfile::GreenNoise) {
        Some(build_green_noise_map(
            width,
            height,
            GREEN_NOISE_BASE_SEED ^ ((width as u64) << 32) ^ height as u64,
        )?)
    } else {
        None
    };
    let mut errors = vec![0.0_f32; len];
    let mut residual = vec![0.0_f32; len];

    for y in 0..height {
        if (y & 1) == 1 {
            for x in (0..width).rev() {
                let idx = y * width + x;
                let adjusted = (data[idx] + errors[idx]).clamp(0.0, 1.0);
                let threshold_adjusted =
                    green_noise_threshold_adjusted(green_noise.as_deref(), idx, adjusted);
                let quantized = quantize_unit_gray::<S>(threshold_adjusted, mode)?;
                let err = adjusted - quantized;
                data[idx] = quantized;
                residual[idx] = err;
                let diffusion_scale = feature_preservation_scale(features.as_deref(), idx, profile)
                    * green_noise_diffusion_scale(green_noise.as_deref(), idx, profile);

                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize - 1,
                    y as isize,
                    err * diffusion_scale * 7.0 / 16.0,
                );
                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize + 1,
                    y as isize + 1,
                    err * diffusion_scale * 3.0 / 16.0,
                );
                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize,
                    y as isize + 1,
                    err * diffusion_scale * 5.0 / 16.0,
                );
                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize - 1,
                    y as isize + 1,
                    err * diffusion_scale / 16.0,
                );
            }
        } else {
            for x in 0..width {
                let idx = y * width + x;
                let adjusted = (data[idx] + errors[idx]).clamp(0.0, 1.0);
                let threshold_adjusted =
                    green_noise_threshold_adjusted(green_noise.as_deref(), idx, adjusted);
                let quantized = quantize_unit_gray::<S>(threshold_adjusted, mode)?;
                let err = adjusted - quantized;
                data[idx] = quantized;
                residual[idx] = err;
                let diffusion_scale = feature_preservation_scale(features.as_deref(), idx, profile)
                    * green_noise_diffusion_scale(green_noise.as_deref(), idx, profile);

                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize + 1,
                    y as isize,
                    err * diffusion_scale * 7.0 / 16.0,
                );
                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize - 1,
                    y as isize + 1,
                    err * diffusion_scale * 3.0 / 16.0,
                );
                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize,
                    y as isize + 1,
                    err * diffusion_scale * 5.0 / 16.0,
                );
                diffuse_scalar_error(
                    &mut errors,
                    width,
                    height,
                    x as isize + 1,
                    y as isize + 1,
                    err * diffusion_scale / 16.0,
                );
            }
        }
    }

    Ok(residual)
}

fn build_feature_map(data: &[f32], width: usize, height: usize) -> Result<Vec<f32>> {
    let len = checked_gray_len(width, height)?;
    if data.len() != len {
        return Err(Error::InvalidArgument(
            "feature map source shape does not match level dimensions",
        ));
    }

    let mut features = vec![0.0_f32; len];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let center = data[idx];
            let left = if x > 0 {
                data[y * width + (x - 1)]
            } else {
                center
            };
            let right = if x + 1 < width {
                data[y * width + (x + 1)]
            } else {
                center
            };
            let up = if y > 0 {
                data[(y - 1) * width + x]
            } else {
                center
            };
            let down = if y + 1 < height {
                data[(y + 1) * width + x]
            } else {
                center
            };
            let gx = (right - left).abs();
            let gy = (down - up).abs();
            features[idx] = (0.5 * (gx + gy)).clamp(0.0, 1.0);
        }
    }

    Ok(features)
}

fn feature_preservation_scale(
    features: Option<&[f32]>,
    idx: usize,
    profile: MultiscaleProfile,
) -> f32 {
    if matches!(profile, MultiscaleProfile::FeaturePreserving) {
        if let Some(feature_map) = features {
            let edge = feature_map.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            return (1.0 - 0.65 * edge).clamp(0.35, 1.0);
        }
    }
    1.0
}

fn green_noise_threshold_adjusted(green_noise: Option<&[f32]>, idx: usize, value: f32) -> f32 {
    let modulation = green_noise
        .and_then(|map| map.get(idx))
        .copied()
        .unwrap_or(0.0)
        .clamp(-1.0, 1.0);
    (value + modulation * GREEN_NOISE_THRESHOLD_AMPLITUDE).clamp(0.0, 1.0)
}

fn green_noise_diffusion_scale(
    green_noise: Option<&[f32]>,
    idx: usize,
    profile: MultiscaleProfile,
) -> f32 {
    if matches!(profile, MultiscaleProfile::GreenNoise) {
        let modulation = green_noise
            .and_then(|map| map.get(idx))
            .copied()
            .unwrap_or(0.0)
            .clamp(-1.0, 1.0);
        return (1.0 + modulation * GREEN_NOISE_DIFFUSION_AMPLITUDE).clamp(0.8, 1.2);
    }
    1.0
}

fn build_green_noise_map(width: usize, height: usize, seed: u64) -> Result<Vec<f32>> {
    let len = checked_gray_len(width, height)?;
    let mut base = vec![0.0_f32; len];
    for y in 0..height {
        for x in 0..width {
            base[y * width + x] = unit_hash_noise(x, y, seed) - 0.5;
        }
    }

    let low = box_blur_scalar(&base, width, height, GREEN_NOISE_LARGE_RADIUS)?;
    let mut band = vec![0.0_f32; len];
    for idx in 0..len {
        band[idx] = base[idx] - low[idx];
    }

    let shaped = box_blur_scalar(&band, width, height, GREEN_NOISE_SMALL_RADIUS)?;
    let mean = shaped.iter().copied().sum::<f32>() / len as f32;
    let max_abs = shaped
        .iter()
        .map(|&value| (value - mean).abs())
        .fold(0.0_f32, f32::max);

    if max_abs <= f32::EPSILON {
        return Ok(vec![0.0_f32; len]);
    }

    let mut normalized = vec![0.0_f32; len];
    for idx in 0..len {
        normalized[idx] = ((shaped[idx] - mean) / max_abs).clamp(-1.0, 1.0);
    }

    Ok(normalized)
}

fn box_blur_scalar(source: &[f32], width: usize, height: usize, radius: usize) -> Result<Vec<f32>> {
    let len = checked_gray_len(width, height)?;
    if source.len() != len {
        return Err(Error::InvalidArgument(
            "green-noise scalar blur source shape mismatch",
        ));
    }
    if radius == 0 || len == 0 {
        return Ok(source.to_vec());
    }

    let integral_width = width
        .checked_add(1)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let integral_height = height
        .checked_add(1)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let integral_len = integral_width
        .checked_mul(integral_height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut integral = vec![0.0_f64; integral_len];

    for y in 0..height {
        let mut row_sum = 0.0_f64;
        for x in 0..width {
            row_sum += f64::from(source[y * width + x]);
            let idx = (y + 1) * integral_width + (x + 1);
            integral[idx] = integral[y * integral_width + (x + 1)] + row_sum;
        }
    }

    let mut out = vec![0.0_f32; len];
    for y in 0..height {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(height.saturating_sub(1));
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width.saturating_sub(1));
            let a = integral[y0 * integral_width + x0];
            let b = integral[y0 * integral_width + (x1 + 1)];
            let c = integral[(y1 + 1) * integral_width + x0];
            let d = integral[(y1 + 1) * integral_width + (x1 + 1)];
            let area = (x1 - x0 + 1)
                .checked_mul(y1 - y0 + 1)
                .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
            out[y * width + x] = ((d - b - c + a) / area as f64) as f32;
        }
    }

    Ok(out)
}

fn unit_hash_noise(x: usize, y: usize, seed: u64) -> f32 {
    let mut state = seed
        ^ (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 30;
    state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state ^= state >> 27;
    state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
    state ^= state >> 31;
    (state >> 40) as f32 / 16_777_215.0
}

fn quantize_unit_gray<S: Sample>(value: f32, mode: QuantizeMode<'_, S>) -> Result<f32> {
    let quantized = quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(value)], mode)?;
    Ok(quantized[0].to_unit_f32())
}

fn diffuse_scalar_error(
    errors: &mut [f32],
    width: usize,
    height: usize,
    x: isize,
    y: isize,
    delta: f32,
) {
    if x < 0 || y < 0 {
        return;
    }

    let x = x as usize;
    let y = y as usize;
    if x >= width || y >= height {
        return;
    }

    errors[y * width + x] += delta;
}

fn write_multiscale_output<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    output: &[f32],
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let expected = checked_gray_len(width, height)?;
    if output.len() != expected {
        return Err(Error::InvalidArgument(
            "multiscale output shape does not match image dimensions",
        ));
    }

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for x in 0..width {
            let src = output[y * width + x].clamp(0.0, 1.0);
            let dst = row.get_mut(x).ok_or(BufferError::OutOfBounds)?;
            *dst = S::from_unit_f32(src);
        }
    }

    Ok(())
}

fn checked_gray_len(width: usize, height: usize) -> Result<usize> {
    width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))
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
            let adjusted = read_pixel_with_error::<S, L>(pixel, &err)?[0];
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
                let adjusted = read_pixel_with_error::<S, L>(pixel, &err)?[0];
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
                let adjusted = read_pixel_with_error::<S, L>(pixel, &err)?[0];
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
