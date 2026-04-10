use crate::{
    core::{layout::validate_layout_invariants, PixelLayout, Sample},
    math::fixed::mul_div_i32,
    quantize_pixel, Buffer, Error, QuantizeMode, Result,
};
use std::{mem::size_of, sync::OnceLock};

const CLASS_MATRIX_W: usize = 8;
const CLASS_MATRIX_H: usize = 8;
const CLASS_COUNT: usize = CLASS_MATRIX_W * CLASS_MATRIX_H;

const KNUTH_CLASS_MATRIX: [u8; CLASS_COUNT] = [
    34, 48, 40, 32, 29, 15, 23, 31, 42, 58, 56, 53, 21, 5, 7, 10, 50, 62, 61, 45, 13, 1, 2, 18, 38,
    46, 54, 37, 25, 17, 9, 26, 28, 14, 22, 30, 35, 49, 41, 33, 20, 4, 6, 11, 43, 59, 57, 52, 12, 0,
    3, 19, 51, 63, 60, 44, 24, 16, 8, 27, 39, 47, 55, 36,
];

const KNUTH_DIFFUSION_WEIGHTS_3X3: [i16; 9] = [1, 2, 1, 2, 0, 2, 1, 2, 1];
const OPTIMIZED_DIFFUSION_WEIGHTS_5X5: [i16; 25] = [
    1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 0, 6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1,
];

const CLASS_OBJECTIVE_WEIGHTS_5X5: [u16; 25] = [
    1, 2, 3, 2, 1, 2, 4, 6, 4, 2, 3, 6, 0, 6, 3, 2, 4, 6, 4, 2, 1, 2, 3, 2, 1,
];
const CLASS_OBJECTIVE_LEVELS: [u8; 8] = [3, 7, 11, 15, 23, 31, 47, 63];
const OPTIMIZATION_MAX_PASSES: usize = 6;
const BARON_ZERO_PENALTY: u64 = 50_000;
const BARON_ONE_PENALTY: u64 = 5_000;

#[derive(Clone, Copy)]
struct DiffusionDims {
    width: usize,
    height: usize,
    max_value: i32,
}

#[derive(Clone, Copy)]
struct ImageDims {
    width: usize,
    height: usize,
}

#[derive(Clone, Copy)]
struct DotDiffusionConfig {
    class_matrix: &'static [u8; CLASS_COUNT],
    class_w: usize,
    class_h: usize,
    diffusion_weights: &'static [i16],
    diffusion_w: usize,
    diffusion_h: usize,
}

const KNUTH_CONFIG: DotDiffusionConfig = DotDiffusionConfig {
    class_matrix: &KNUTH_CLASS_MATRIX,
    class_w: CLASS_MATRIX_W,
    class_h: CLASS_MATRIX_H,
    diffusion_weights: &KNUTH_DIFFUSION_WEIGHTS_3X3,
    diffusion_w: 3,
    diffusion_h: 3,
};

static OPTIMIZED_CLASS_MATRIX: OnceLock<[u8; CLASS_COUNT]> = OnceLock::new();

pub fn knuth_dot_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    dot_diffusion_in_place(
        buffer,
        mode,
        KNUTH_CONFIG,
        "knuth dot diffusion supports Gray, Rgb, and Rgba formats only",
    )
}

pub fn optimized_dot_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    dot_diffusion_in_place(
        buffer,
        mode,
        optimized_dot_diffusion_config(),
        "optimized dot diffusion supports Gray, Rgb, and Rgba formats only",
    )
}

fn optimized_dot_diffusion_config() -> DotDiffusionConfig {
    DotDiffusionConfig {
        class_matrix: optimized_class_matrix_8x8(),
        class_w: CLASS_MATRIX_W,
        class_h: CLASS_MATRIX_H,
        diffusion_weights: &OPTIMIZED_DIFFUSION_WEIGHTS_5X5,
        diffusion_w: 5,
        diffusion_h: 5,
    }
}

fn optimized_class_matrix_8x8() -> &'static [u8; CLASS_COUNT] {
    OPTIMIZED_CLASS_MATRIX.get_or_init(generate_optimized_class_matrix_8x8)
}

fn generate_optimized_class_matrix_8x8() -> [u8; CLASS_COUNT] {
    let mut matrix = KNUTH_CLASS_MATRIX;
    let mut best_score = class_matrix_objective(&matrix);

    for _ in 0..OPTIMIZATION_MAX_PASSES {
        let mut best_swap = None;
        let mut best_swap_score = best_score;

        for i in 0..(CLASS_COUNT - 1) {
            for j in (i + 1)..CLASS_COUNT {
                matrix.swap(i, j);
                let score = class_matrix_objective(&matrix);
                matrix.swap(i, j);

                if score < best_swap_score {
                    best_swap_score = score;
                    best_swap = Some((i, j));
                }
            }
        }

        match best_swap {
            Some((i, j)) => {
                matrix.swap(i, j);
                best_score = best_swap_score;
            }
            None => break,
        }
    }

    matrix
}

fn class_matrix_objective(matrix: &[u8; CLASS_COUNT]) -> u64 {
    let mut score = 0_u64;

    for &threshold in &CLASS_OBJECTIVE_LEVELS {
        for y in 0..CLASS_MATRIX_H {
            for x in 0..CLASS_MATRIX_W {
                let idx = y * CLASS_MATRIX_W + x;
                if matrix[idx] > threshold {
                    continue;
                }

                for ky in 0..5 {
                    for kx in 0..5 {
                        let weight = u64::from(CLASS_OBJECTIVE_WEIGHTS_5X5[ky * 5 + kx]);
                        if weight == 0 {
                            continue;
                        }

                        let nx = wrap_index(x as isize + kx as isize - 2, CLASS_MATRIX_W);
                        let ny = wrap_index(y as isize + ky as isize - 2, CLASS_MATRIX_H);
                        let nidx = ny * CLASS_MATRIX_W + nx;

                        if matrix[nidx] <= threshold {
                            score += weight;
                        }
                    }
                }
            }
        }
    }

    score + class_matrix_baron_penalty(matrix)
}

fn class_matrix_baron_penalty(matrix: &[u8; CLASS_COUNT]) -> u64 {
    let mut positions = [usize::MAX; CLASS_COUNT];
    for (idx, &class_value) in matrix.iter().enumerate() {
        positions[class_value as usize] = idx;
    }

    let mut penalty = 0_u64;

    for (class, &pos) in positions.iter().enumerate() {
        if pos == usize::MAX {
            return u64::MAX / 4;
        }

        let x = pos % CLASS_MATRIX_W;
        let y = pos / CLASS_MATRIX_W;
        let mut higher_count = 0_u8;

        for dy in -1_isize..=1 {
            for dx in -1_isize..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let nx = wrap_index(x as isize + dx, CLASS_MATRIX_W);
                let ny = wrap_index(y as isize + dy, CLASS_MATRIX_H);
                let nidx = ny * CLASS_MATRIX_W + nx;
                if usize::from(matrix[nidx]) > class {
                    higher_count = higher_count.saturating_add(1);
                }
            }
        }

        if higher_count == 0 {
            penalty = penalty.saturating_add(BARON_ZERO_PENALTY);
        } else if higher_count == 1 {
            penalty = penalty.saturating_add(BARON_ONE_PENALTY);
        }
    }

    penalty
}

fn wrap_index(value: isize, size: usize) -> usize {
    value.rem_euclid(size as isize) as usize
}

fn dot_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
    unsupported_message: &'static str,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;
    validate_dot_diffusion_config(config)?;

    if L::CHANNELS == 1 && !L::HAS_ALPHA {
        dot_diffuse_gray(buffer, mode, config)
    } else if (L::CHANNELS == 3 && !L::HAS_ALPHA) || (L::CHANNELS == 4 && L::HAS_ALPHA) {
        dot_diffuse_rgb(buffer, mode, config)
    } else {
        Err(Error::UnsupportedFormat(unsupported_message))
    }
}

fn validate_dot_diffusion_config(config: DotDiffusionConfig) -> Result<()> {
    if config.class_w == 0 || config.class_h == 0 {
        return Err(Error::InvalidArgument(
            "dot diffusion class matrix dimensions must be non-zero",
        ));
    }
    if config.class_w.checked_mul(config.class_h) != Some(CLASS_COUNT) {
        return Err(Error::InvalidArgument(
            "dot diffusion class matrix dimensions do not match storage size",
        ));
    }
    if config.diffusion_w == 0 || config.diffusion_h == 0 {
        return Err(Error::InvalidArgument(
            "dot diffusion diffusion-kernel dimensions must be non-zero",
        ));
    }
    if config.diffusion_w % 2 == 0 || config.diffusion_h % 2 == 0 {
        return Err(Error::InvalidArgument(
            "dot diffusion diffusion-kernel dimensions must be odd",
        ));
    }
    if config.diffusion_weights.len()
        != config
            .diffusion_w
            .checked_mul(config.diffusion_h)
            .ok_or(Error::InvalidArgument(
                "dot diffusion kernel dimensions overflow",
            ))?
    {
        return Err(Error::InvalidArgument(
            "dot diffusion diffusion-kernel dimensions do not match storage size",
        ));
    }
    Ok(())
}

fn dot_diffuse_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
) -> Result<()> {
    if S::IS_FLOAT {
        return dot_diffuse_gray_float(buffer, mode, config);
    }

    dot_diffuse_gray_int(buffer, mode, config)
}

fn dot_diffuse_gray_int<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
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
                if class_at(config, x, y) != class {
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
                diffuse_gray_to_higher_classes_int(&mut errors, dims, x, y, class, err, config)?;
            }
        }
    }

    Ok(())
}

fn dot_diffuse_gray_float<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let mut errors = vec![0.0_f32; pixel_count];
    let dims = ImageDims { width, height };

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for (x, value) in row.iter_mut().take(width).enumerate() {
                if class_at(config, x, y) != class {
                    continue;
                }

                let idx = y * width + x;
                let adjusted = (value.to_unit_f32() + errors[idx]).clamp(0.0, 1.0);
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                let quantized_gray = quantized[0];
                *value = quantized_gray;

                let err = adjusted - quantized_gray.to_unit_f32();
                diffuse_gray_to_higher_classes_float(&mut errors, dims, x, y, class, err, config);
            }
        }
    }

    Ok(())
}

fn dot_diffuse_rgb<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
) -> Result<()> {
    if S::IS_FLOAT {
        return dot_diffuse_rgb_float(buffer, mode, config);
    }

    dot_diffuse_rgb_int(buffer, mode, config)
}

fn dot_diffuse_rgb_int<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let max_value = integer_sample_max::<S>()?;
    let dims = DiffusionDims {
        width,
        height,
        max_value,
    };
    let sample_channels = L::CHANNELS;
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];
    let is_rgba = L::HAS_ALPHA;

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for x in 0..width {
                if class_at(config, x, y) != class {
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
                diffuse_rgb_to_higher_classes_int(&mut errors, dims, x, y, class, err, config)?;
            }
        }
    }

    Ok(())
}

fn dot_diffuse_rgb_float<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    config: DotDiffusionConfig,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let sample_channels = L::CHANNELS;
    let channels = 3_usize;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(channels)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];
    let dims = ImageDims { width, height };
    let is_rgba = L::HAS_ALPHA;

    for class in 0..CLASS_COUNT as u8 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;

            for x in 0..width {
                if class_at(config, x, y) != class {
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
                diffuse_rgb_to_higher_classes_float(&mut errors, dims, x, y, class, err, config);
            }
        }
    }

    Ok(())
}

fn diffuse_gray_to_higher_classes_float(
    errors: &mut [f32],
    dims: ImageDims,
    x: usize,
    y: usize,
    class: u8,
    err: f32,
    config: DotDiffusionConfig,
) {
    let targets = higher_class_neighbors(dims.width, dims.height, x, y, class, config);
    if targets.is_empty() {
        return;
    }
    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| f32::from(weight))
        .sum::<f32>();

    for (nx, ny, weight) in targets {
        let distributed = err * f32::from(weight) / total_weight;
        let idx = ny * dims.width + nx;
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
    config: DotDiffusionConfig,
) -> Result<()> {
    let targets = higher_class_neighbors(dims.width, dims.height, x, y, class, config);
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
    dims: ImageDims,
    x: usize,
    y: usize,
    class: u8,
    err: [f32; 3],
    config: DotDiffusionConfig,
) {
    let targets = higher_class_neighbors(dims.width, dims.height, x, y, class, config);
    if targets.is_empty() {
        return;
    }

    let total_weight = targets
        .iter()
        .map(|&(_, _, weight)| f32::from(weight))
        .sum::<f32>();

    for (nx, ny, weight) in targets {
        let w = f32::from(weight) / total_weight;
        let base = (ny * dims.width + nx) * 3;
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
    config: DotDiffusionConfig,
) -> Result<()> {
    let targets = higher_class_neighbors(dims.width, dims.height, x, y, class, config);
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
    config: DotDiffusionConfig,
) -> Vec<(usize, usize, i16)> {
    let mut targets = Vec::new();
    let radius_x = config.diffusion_w / 2;
    let radius_y = config.diffusion_h / 2;

    let x0 = x.saturating_sub(radius_x);
    let y0 = y.saturating_sub(radius_y);
    let x1 = (x + radius_x).min(width - 1);
    let y1 = (y + radius_y).min(height - 1);

    for ny in y0..=y1 {
        for nx in x0..=x1 {
            if nx == x && ny == y {
                continue;
            }

            let kernel_x = (radius_x as isize + nx as isize - x as isize) as usize;
            let kernel_y = (radius_y as isize + ny as isize - y as isize) as usize;
            let weight = config.diffusion_weights[kernel_y * config.diffusion_w + kernel_x];
            if weight <= 0 {
                continue;
            }
            if class_at(config, nx, ny) > class {
                targets.push((nx, ny, weight));
            }
        }
    }

    targets
}

fn class_at(config: DotDiffusionConfig, x: usize, y: usize) -> u8 {
    let row = y % config.class_h;
    let col = x % config.class_w;
    config.class_matrix[row * config.class_w + col]
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
