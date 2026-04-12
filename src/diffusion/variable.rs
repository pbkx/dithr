use super::core::{diffuse_error_forward, read_pixel_with_error, write_quantized_pixel};
use crate::{
    core::{read_unit_pixel, PixelLayout, Sample},
    data::{OSTROMOUKHOV_COEFFS, ZHOU_FANG_MODULATION},
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};

#[derive(Clone, Copy)]
enum GrayOnlyVariableAlgorithm {
    Ostromoukhov,
    ZhouFang,
    ToneDependent,
    StructureAware,
    HvsOptimized,
    GradientBased,
    Multiscale,
    FeaturePreservingMsed,
    GreenNoiseMsed,
    LinearPixelShuffling,
}

#[derive(Clone, Copy)]
enum ColorVectorAlgorithm {
    Vector,
    SemiVector,
    Hierarchical,
    Mbvq,
    Neugebauer,
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

pub fn hvs_optimized_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::HvsOptimized)
}

pub fn tone_dependent_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::ToneDependent)
}

pub fn structure_aware_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(buffer, mode, GrayOnlyVariableAlgorithm::StructureAware)
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

pub fn linear_pixel_shuffling_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_gray_only_variable(
        buffer,
        mode,
        GrayOnlyVariableAlgorithm::LinearPixelShuffling,
    )
}

pub fn adaptive_vector_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;

    if !(L::COLOR_CHANNELS == 3 && (L::CHANNELS == 3 || L::CHANNELS == 4)) {
        return Err(Error::UnsupportedFormat(
            "adaptive vector error diffusion supports Rgb and Rgba formats only",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;
    let channels = L::CHANNELS;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];
    let mut weights = ADAPTIVE_VECTOR_BASE_WEIGHTS;
    let mut previous_residual = [0.0_f32; 3];

    for y in 0..height {
        let reverse = (y & 1) == 1;
        let row = buffer.try_row_mut(y)?;

        if reverse {
            for x in (0..width).rev() {
                let offset = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;
                let end = offset
                    .checked_add(channels)
                    .ok_or(BufferError::OutOfBounds)?;
                let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
                let err_idx = (y * width + x) * 4;
                let err = [
                    errors[err_idx],
                    errors[err_idx + 1],
                    errors[err_idx + 2],
                    errors[err_idx + 3],
                ];
                let adjusted_unit = read_pixel_with_error::<S, L>(pixel, &err)?;
                let adjusted = [
                    S::from_unit_f32(adjusted_unit[0]),
                    S::from_unit_f32(adjusted_unit[1]),
                    S::from_unit_f32(adjusted_unit[2]),
                    S::from_unit_f32(adjusted_unit[3]),
                ];
                let quantized = quantize_pixel::<S, L>(&adjusted[..channels], mode)?;
                let quantized_unit = read_unit_pixel::<S, L>(&quantized[..channels])?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = [
                    adjusted_unit[0] - quantized_unit[0],
                    adjusted_unit[1] - quantized_unit[1],
                    adjusted_unit[2] - quantized_unit[2],
                ];
                update_adaptive_vector_weights(&mut weights, residual, previous_residual)?;
                previous_residual = residual;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize - 1,
                    y as isize,
                    [
                        residual[0] * weights[0],
                        residual[1] * weights[0],
                        residual[2] * weights[0],
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize + 1,
                    y as isize + 1,
                    [
                        residual[0] * weights[1],
                        residual[1] * weights[1],
                        residual[2] * weights[1],
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize,
                    y as isize + 1,
                    [
                        residual[0] * weights[2],
                        residual[1] * weights[2],
                        residual[2] * weights[2],
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize - 1,
                    y as isize + 1,
                    [
                        residual[0] * weights[3],
                        residual[1] * weights[3],
                        residual[2] * weights[3],
                        0.0,
                    ],
                );
            }
        } else {
            for x in 0..width {
                let offset = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;
                let end = offset
                    .checked_add(channels)
                    .ok_or(BufferError::OutOfBounds)?;
                let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
                let err_idx = (y * width + x) * 4;
                let err = [
                    errors[err_idx],
                    errors[err_idx + 1],
                    errors[err_idx + 2],
                    errors[err_idx + 3],
                ];
                let adjusted_unit = read_pixel_with_error::<S, L>(pixel, &err)?;
                let adjusted = [
                    S::from_unit_f32(adjusted_unit[0]),
                    S::from_unit_f32(adjusted_unit[1]),
                    S::from_unit_f32(adjusted_unit[2]),
                    S::from_unit_f32(adjusted_unit[3]),
                ];
                let quantized = quantize_pixel::<S, L>(&adjusted[..channels], mode)?;
                let quantized_unit = read_unit_pixel::<S, L>(&quantized[..channels])?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = [
                    adjusted_unit[0] - quantized_unit[0],
                    adjusted_unit[1] - quantized_unit[1],
                    adjusted_unit[2] - quantized_unit[2],
                ];
                update_adaptive_vector_weights(&mut weights, residual, previous_residual)?;
                previous_residual = residual;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize + 1,
                    y as isize,
                    [
                        residual[0] * weights[0],
                        residual[1] * weights[0],
                        residual[2] * weights[0],
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize - 1,
                    y as isize + 1,
                    [
                        residual[0] * weights[1],
                        residual[1] * weights[1],
                        residual[2] * weights[1],
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize,
                    y as isize + 1,
                    [
                        residual[0] * weights[2],
                        residual[1] * weights[2],
                        residual[2] * weights[2],
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    x as isize + 1,
                    y as isize + 1,
                    [
                        residual[0] * weights[3],
                        residual[1] * weights[3],
                        residual[2] * weights[3],
                        0.0,
                    ],
                );
            }
        }
    }

    Ok(())
}

pub fn vector_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_color_vector(buffer, mode, ColorVectorAlgorithm::Vector)
}

pub fn semivector_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_color_vector(buffer, mode, ColorVectorAlgorithm::SemiVector)
}

pub fn hierarchical_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_color_vector(buffer, mode, ColorVectorAlgorithm::Hierarchical)
}

pub fn mbvq_color_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_color_vector(buffer, mode, ColorVectorAlgorithm::Mbvq)
}

pub fn neugebauer_color_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    diffuse_color_vector(buffer, mode, ColorVectorAlgorithm::Neugebauer)
}

fn diffuse_color_vector<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    algorithm: ColorVectorAlgorithm,
) -> Result<()> {
    buffer.validate()?;

    if !(L::COLOR_CHANNELS == 3 && (L::CHANNELS == 3 || L::CHANNELS == 4)) {
        return Err(Error::UnsupportedFormat(
            "vector diffusion algorithms support Rgb and Rgba formats only",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;
    let channels = L::CHANNELS;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let error_len = pixel_count
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("error buffer size overflow"))?;
    let mut errors = vec![0.0_f32; error_len];

    for y in 0..height {
        let reverse = (y & 1) == 1;
        let row = buffer.try_row_mut(y)?;

        if reverse {
            for x in (0..width).rev() {
                let offset = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;
                let end = offset
                    .checked_add(channels)
                    .ok_or(BufferError::OutOfBounds)?;
                let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
                let err_idx = (y * width + x) * 4;
                let err = [
                    errors[err_idx],
                    errors[err_idx + 1],
                    errors[err_idx + 2],
                    errors[err_idx + 3],
                ];
                let adjusted_unit = read_pixel_with_error::<S, L>(pixel, &err)?;
                let adjusted = [
                    S::from_unit_f32(adjusted_unit[0]),
                    S::from_unit_f32(adjusted_unit[1]),
                    S::from_unit_f32(adjusted_unit[2]),
                    S::from_unit_f32(adjusted_unit[3]),
                ];
                let quantized = quantize_color_vector_pixel::<S, L>(
                    pixel,
                    adjusted_unit,
                    adjusted,
                    mode,
                    algorithm,
                )?;
                let quantized_unit = read_unit_pixel::<S, L>(&quantized[..channels])?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = [
                    adjusted_unit[0] - quantized_unit[0],
                    adjusted_unit[1] - quantized_unit[1],
                    adjusted_unit[2] - quantized_unit[2],
                ];
                let transformed = color_residual_transform(residual, x, y, algorithm);
                let xi = x as isize;
                let yi = y as isize;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_FORWARD,
                        transformed[1] * VECTOR_ED_WEIGHT_FORWARD,
                        transformed[2] * VECTOR_ED_WEIGHT_FORWARD,
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_DOWN_DIAGONAL,
                        transformed[1] * VECTOR_ED_WEIGHT_DOWN_DIAGONAL,
                        transformed[2] * VECTOR_ED_WEIGHT_DOWN_DIAGONAL,
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_DOWN,
                        transformed[1] * VECTOR_ED_WEIGHT_DOWN,
                        transformed[2] * VECTOR_ED_WEIGHT_DOWN,
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_DOWN_FORWARD,
                        transformed[1] * VECTOR_ED_WEIGHT_DOWN_FORWARD,
                        transformed[2] * VECTOR_ED_WEIGHT_DOWN_FORWARD,
                        0.0,
                    ],
                );
            }
        } else {
            for x in 0..width {
                let offset = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;
                let end = offset
                    .checked_add(channels)
                    .ok_or(BufferError::OutOfBounds)?;
                let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
                let err_idx = (y * width + x) * 4;
                let err = [
                    errors[err_idx],
                    errors[err_idx + 1],
                    errors[err_idx + 2],
                    errors[err_idx + 3],
                ];
                let adjusted_unit = read_pixel_with_error::<S, L>(pixel, &err)?;
                let adjusted = [
                    S::from_unit_f32(adjusted_unit[0]),
                    S::from_unit_f32(adjusted_unit[1]),
                    S::from_unit_f32(adjusted_unit[2]),
                    S::from_unit_f32(adjusted_unit[3]),
                ];
                let quantized = quantize_color_vector_pixel::<S, L>(
                    pixel,
                    adjusted_unit,
                    adjusted,
                    mode,
                    algorithm,
                )?;
                let quantized_unit = read_unit_pixel::<S, L>(&quantized[..channels])?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = [
                    adjusted_unit[0] - quantized_unit[0],
                    adjusted_unit[1] - quantized_unit[1],
                    adjusted_unit[2] - quantized_unit[2],
                ];
                let transformed = color_residual_transform(residual, x, y, algorithm);
                let xi = x as isize;
                let yi = y as isize;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_FORWARD,
                        transformed[1] * VECTOR_ED_WEIGHT_FORWARD,
                        transformed[2] * VECTOR_ED_WEIGHT_FORWARD,
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_DOWN_DIAGONAL,
                        transformed[1] * VECTOR_ED_WEIGHT_DOWN_DIAGONAL,
                        transformed[2] * VECTOR_ED_WEIGHT_DOWN_DIAGONAL,
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_DOWN,
                        transformed[1] * VECTOR_ED_WEIGHT_DOWN,
                        transformed[2] * VECTOR_ED_WEIGHT_DOWN,
                        0.0,
                    ],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    [
                        transformed[0] * VECTOR_ED_WEIGHT_DOWN_FORWARD,
                        transformed[1] * VECTOR_ED_WEIGHT_DOWN_FORWARD,
                        transformed[2] * VECTOR_ED_WEIGHT_DOWN_FORWARD,
                        0.0,
                    ],
                );
            }
        }
    }

    Ok(())
}

fn color_residual_matrix_apply(error: [f32; 3], matrix: [[f32; 3]; 3]) -> [f32; 3] {
    [
        matrix[0][0] * error[0] + matrix[0][1] * error[1] + matrix[0][2] * error[2],
        matrix[1][0] * error[0] + matrix[1][1] * error[1] + matrix[1][2] * error[2],
        matrix[2][0] * error[0] + matrix[2][1] * error[1] + matrix[2][2] * error[2],
    ]
}

fn hierarchical_color_residual_transform(residual: [f32; 3], x: usize, y: usize) -> [f32; 3] {
    let base = (residual[0] + residual[1] + residual[2]) / 3.0;
    let chroma = [residual[0] - base, residual[1] - base, residual[2] - base];

    let phase = ((x & 1) | ((y & 1) << 1)) & 3;
    let sequence = HIERARCHICAL_COLOR_SEQUENCE[phase];
    let mut sequenced = [
        chroma[sequence[0]],
        chroma[sequence[1]],
        chroma[sequence[2]],
    ];
    let overlap = 0.5 * (sequenced[0] + sequenced[1]);
    sequenced[0] += overlap * HIERARCHICAL_OVERLAP_PRIMARY;
    sequenced[1] += overlap * HIERARCHICAL_OVERLAP_SECONDARY;
    sequenced[2] -= overlap * HIERARCHICAL_OVERLAP_TERTIARY;

    let position_gain = if ((x ^ y) & 1) == 0 {
        1.0 + HIERARCHICAL_POSITION_GAIN
    } else {
        1.0 - HIERARCHICAL_POSITION_GAIN
    };

    let medium = color_residual_matrix_apply(chroma, VECTOR_DIFFUSION_MATRIX);
    let sequenced_fine = color_residual_matrix_apply(sequenced, SEMIVECTOR_DIFFUSION_MATRIX);
    let mut fine = [0.0_f32; 3];
    fine[sequence[0]] = sequenced_fine[0];
    fine[sequence[1]] = sequenced_fine[1];
    fine[sequence[2]] = sequenced_fine[2];

    [
        base * HIERARCHICAL_BASE_WEIGHT
            + medium[0] * HIERARCHICAL_MEDIUM_WEIGHT * position_gain
            + fine[0] * HIERARCHICAL_FINE_WEIGHT,
        base * HIERARCHICAL_BASE_WEIGHT
            + medium[1] * HIERARCHICAL_MEDIUM_WEIGHT * position_gain
            + fine[1] * HIERARCHICAL_FINE_WEIGHT,
        base * HIERARCHICAL_BASE_WEIGHT
            + medium[2] * HIERARCHICAL_MEDIUM_WEIGHT * position_gain
            + fine[2] * HIERARCHICAL_FINE_WEIGHT,
    ]
}

fn color_residual_transform(
    residual: [f32; 3],
    x: usize,
    y: usize,
    algorithm: ColorVectorAlgorithm,
) -> [f32; 3] {
    match algorithm {
        ColorVectorAlgorithm::Vector => {
            color_residual_matrix_apply(residual, VECTOR_DIFFUSION_MATRIX)
        }
        ColorVectorAlgorithm::SemiVector => {
            color_residual_matrix_apply(residual, SEMIVECTOR_DIFFUSION_MATRIX)
        }
        ColorVectorAlgorithm::Hierarchical => hierarchical_color_residual_transform(residual, x, y),
        ColorVectorAlgorithm::Mbvq => {
            color_residual_matrix_apply(residual, MBVQ_COLOR_DIFFUSION_MATRIX)
        }
        ColorVectorAlgorithm::Neugebauer => {
            color_residual_matrix_apply(residual, NEUGEBAUER_COLOR_DIFFUSION_MATRIX)
        }
    }
}

fn quantize_color_vector_pixel<S: Sample, L: PixelLayout>(
    pixel: &[S],
    adjusted_unit: [f32; 4],
    adjusted: [S; 4],
    mode: QuantizeMode<'_, S>,
    algorithm: ColorVectorAlgorithm,
) -> Result<[S; 4]> {
    match algorithm {
        ColorVectorAlgorithm::Vector
        | ColorVectorAlgorithm::SemiVector
        | ColorVectorAlgorithm::Hierarchical => {
            quantize_pixel::<S, L>(&adjusted[..L::CHANNELS], mode)
        }
        ColorVectorAlgorithm::Mbvq | ColorVectorAlgorithm::Neugebauer => {
            let quantized_hint = quantize_pixel::<S, L>(&adjusted[..L::CHANNELS], mode)?;
            let hint_unit = read_unit_pixel::<S, L>(&quantized_hint[..L::CHANNELS])?;
            let source_unit = read_unit_pixel::<S, L>(pixel)?;
            let state = select_neugebauer_primary(
                [source_unit[0], source_unit[1], source_unit[2]],
                [hint_unit[0], hint_unit[1], hint_unit[2]],
                algorithm,
            );
            let primary = neugebauer_primary_unit(state);
            Ok([
                S::from_unit_f32(primary[0]),
                S::from_unit_f32(primary[1]),
                S::from_unit_f32(primary[2]),
                S::from_unit_f32(adjusted_unit[3]),
            ])
        }
    }
}

fn select_neugebauer_primary(
    source_rgb: [f32; 3],
    target_rgb: [f32; 3],
    algorithm: ColorVectorAlgorithm,
) -> usize {
    let candidates = match algorithm {
        ColorVectorAlgorithm::Mbvq => CandidateStates::Mbvq(mbvq_states_for_rgb(source_rgb)),
        ColorVectorAlgorithm::Neugebauer => CandidateStates::All,
        _ => CandidateStates::All,
    };

    let mut best_state = 0_usize;
    let mut best_distance = f32::INFINITY;
    match candidates {
        CandidateStates::Mbvq(states) => {
            for &state in &states {
                let candidate = neugebauer_primary_unit(state);
                let distance = neugebauer_color_distance(candidate, target_rgb);
                if distance < best_distance || (distance == best_distance && state < best_state) {
                    best_distance = distance;
                    best_state = state;
                }
            }
        }
        CandidateStates::All => {
            for state in 0..8_usize {
                let candidate = neugebauer_primary_unit(state);
                let distance = neugebauer_color_distance(candidate, target_rgb);
                if distance < best_distance || (distance == best_distance && state < best_state) {
                    best_distance = distance;
                    best_state = state;
                }
            }
        }
    }

    best_state
}

fn neugebauer_color_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    NEUGEBAUER_DISTANCE_WEIGHTS[0] * dr * dr
        + NEUGEBAUER_DISTANCE_WEIGHTS[1] * dg * dg
        + NEUGEBAUER_DISTANCE_WEIGHTS[2] * db * db
}

fn neugebauer_primary_unit(state: usize) -> [f32; 3] {
    [
        if state & 0b001 != 0 { 1.0 } else { 0.0 },
        if state & 0b010 != 0 { 1.0 } else { 0.0 },
        if state & 0b100 != 0 { 1.0 } else { 0.0 },
    ]
}

fn mbvq_states_for_rgb(rgb: [f32; 3]) -> [usize; 4] {
    match mbvq_region_for_rgb(rgb) {
        MbvqRegion::Cmyw => [0b110, 0b101, 0b011, 0b111],
        MbvqRegion::Mygc => [0b101, 0b011, 0b010, 0b110],
        MbvqRegion::Rgmy => [0b001, 0b010, 0b101, 0b011],
        MbvqRegion::Krgb => [0b000, 0b001, 0b010, 0b100],
        MbvqRegion::Rgbm => [0b001, 0b010, 0b100, 0b101],
        MbvqRegion::Cmgb => [0b110, 0b101, 0b010, 0b100],
    }
}

fn mbvq_region_for_rgb(rgb: [f32; 3]) -> MbvqRegion {
    let r = rgb[0];
    let g = rgb[1];
    let b = rgb[2];

    if r + g > 1.0 {
        if g + b > 1.0 {
            if r + g + b > 2.0 {
                MbvqRegion::Cmyw
            } else {
                MbvqRegion::Mygc
            }
        } else {
            MbvqRegion::Rgmy
        }
    } else if g + b <= 1.0 {
        if r + g + b <= 1.0 {
            MbvqRegion::Krgb
        } else {
            MbvqRegion::Rgbm
        }
    } else {
        MbvqRegion::Cmgb
    }
}

#[derive(Clone, Copy)]
enum CandidateStates {
    Mbvq([usize; 4]),
    All,
}

#[derive(Clone, Copy)]
enum MbvqRegion {
    Cmyw,
    Mygc,
    Rgmy,
    Krgb,
    Rgbm,
    Cmgb,
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
        GrayOnlyVariableAlgorithm::ToneDependent => diffuse_tone_dependent_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::StructureAware => diffuse_structure_aware_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::HvsOptimized => diffuse_hvs_optimized_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::GradientBased => diffuse_gradient_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::Multiscale => diffuse_multiscale_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::FeaturePreservingMsed => {
            diffuse_feature_preserving_msed_gray(buffer, mode)
        }
        GrayOnlyVariableAlgorithm::GreenNoiseMsed => diffuse_green_noise_msed_gray(buffer, mode),
        GrayOnlyVariableAlgorithm::LinearPixelShuffling => {
            diffuse_linear_pixel_shuffling_gray(buffer, mode)
        }
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
const ADAPTIVE_VECTOR_BASE_WEIGHTS: [f32; 4] = [7.0 / 16.0, 3.0 / 16.0, 5.0 / 16.0, 1.0 / 16.0];
const ADAPTIVE_VECTOR_MIN_WEIGHT: f32 = 0.05;
const ADAPTIVE_VECTOR_MAX_WEIGHT: f32 = 0.75;
const ADAPTIVE_VECTOR_LEARNING_RATE: f32 = 0.09;
const ADAPTIVE_VECTOR_RELAX_RATE: f32 = 0.035;
const ADAPTIVE_VECTOR_EPSILON: f32 = 1.0e-7;
const VECTOR_ED_WEIGHT_FORWARD: f32 = 7.0 / 16.0;
const VECTOR_ED_WEIGHT_DOWN_DIAGONAL: f32 = 3.0 / 16.0;
const VECTOR_ED_WEIGHT_DOWN: f32 = 5.0 / 16.0;
const VECTOR_ED_WEIGHT_DOWN_FORWARD: f32 = 1.0 / 16.0;
const VECTOR_DIFFUSION_MATRIX: [[f32; 3]; 3] =
    [[0.84, 0.10, 0.06], [0.10, 0.84, 0.06], [0.08, 0.08, 0.84]];
const SEMIVECTOR_DIFFUSION_MATRIX: [[f32; 3]; 3] = [
    [0.77568, 0.18784, 0.03648],
    [0.09568, 0.86784, 0.03648],
    [0.09568, 0.18784, 0.71648],
];
const MBVQ_COLOR_DIFFUSION_MATRIX: [[f32; 3]; 3] =
    [[0.86, 0.09, 0.05], [0.09, 0.86, 0.05], [0.07, 0.07, 0.86]];
const NEUGEBAUER_COLOR_DIFFUSION_MATRIX: [[f32; 3]; 3] =
    [[0.82, 0.12, 0.06], [0.12, 0.82, 0.06], [0.10, 0.10, 0.82]];
const NEUGEBAUER_DISTANCE_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];
const HIERARCHICAL_COLOR_SEQUENCE: [[usize; 3]; 4] = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1]];
const HIERARCHICAL_BASE_WEIGHT: f32 = 0.56;
const HIERARCHICAL_MEDIUM_WEIGHT: f32 = 0.30;
const HIERARCHICAL_FINE_WEIGHT: f32 = 0.14;
const HIERARCHICAL_POSITION_GAIN: f32 = 0.08;
const HIERARCHICAL_OVERLAP_PRIMARY: f32 = 0.34;
const HIERARCHICAL_OVERLAP_SECONDARY: f32 = 0.16;
const HIERARCHICAL_OVERLAP_TERTIARY: f32 = 0.24;
const TONE_DEPENDENT_DIFFUSION_COEFFS: [[f32; 3]; 16] = [
    [0.50, 0.26, 0.24],
    [0.52, 0.24, 0.24],
    [0.54, 0.23, 0.23],
    [0.56, 0.22, 0.22],
    [0.58, 0.21, 0.21],
    [0.60, 0.20, 0.20],
    [0.62, 0.19, 0.19],
    [0.64, 0.18, 0.18],
    [0.64, 0.18, 0.18],
    [0.62, 0.19, 0.19],
    [0.60, 0.20, 0.20],
    [0.58, 0.21, 0.21],
    [0.56, 0.22, 0.22],
    [0.54, 0.23, 0.23],
    [0.52, 0.24, 0.24],
    [0.50, 0.26, 0.24],
];
const STRUCTURE_AWARE_FORWARD_REDUCTION: f32 = 0.22;
const STRUCTURE_AWARE_DIAGONAL_BOOST: f32 = 0.33;
const STRUCTURE_AWARE_DOWN_BOOST: f32 = 0.28;
const STRUCTURE_AWARE_TONE_CENTER_BOOST: f32 = 0.18;
const STRUCTURE_AWARE_GAIN_MIN: f32 = 0.82;
const STRUCTURE_AWARE_GAIN_SCALE: f32 = 0.32;
const HVS_OPTIMIZED_FORWARD: f32 = 0.7770;
const HVS_OPTIMIZED_DOWN_DIAGONAL: f32 = -0.0090;
const HVS_OPTIMIZED_DOWN: f32 = 0.7861;
const HVS_OPTIMIZED_DOWN_FORWARD: f32 = -0.6098;
const LPS_BASE_SEED: u64 = 0x6C8E_9CF5_42A1_7D3B;
const LPS_NEIGHBORS: [(isize, isize, f32); 8] = [
    (-1, -1, 1.0 / 12.0),
    (0, -1, 2.0 / 12.0),
    (1, -1, 1.0 / 12.0),
    (-1, 0, 2.0 / 12.0),
    (1, 0, 2.0 / 12.0),
    (-1, 1, 1.0 / 12.0),
    (0, 1, 2.0 / 12.0),
    (1, 1, 1.0 / 12.0),
];

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

fn diffuse_linear_pixel_shuffling_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let len = checked_gray_len(width, height)?;
    let mut data = read_gray_units(buffer)?;
    if data.len() != len {
        return Err(Error::InvalidArgument(
            "linear pixel shuffling source shape mismatch",
        ));
    }

    let order = linear_pixel_shuffling_permutation(width, height, LPS_BASE_SEED)?;
    if order.len() != len {
        return Err(Error::InvalidArgument(
            "linear pixel shuffling traversal size mismatch",
        ));
    }

    let mut errors = vec![0.0_f32; len];
    let mut processed = vec![false; len];

    for &idx in &order {
        if idx >= len {
            return Err(Error::InvalidArgument(
                "linear pixel shuffling traversal contains invalid index",
            ));
        }
        if processed[idx] {
            return Err(Error::InvalidArgument(
                "linear pixel shuffling traversal repeats index",
            ));
        }

        let x = idx % width;
        let y = idx / width;
        let adjusted = (data[idx] + errors[idx]).clamp(0.0, 1.0);
        let quantized = quantize_unit_gray::<S>(adjusted, mode)?;
        let residual = adjusted - quantized;
        data[idx] = quantized;
        processed[idx] = true;

        let mut available = [(0_usize, 0.0_f32); 8];
        let mut available_len = 0_usize;
        let mut available_weight_sum = 0.0_f32;
        for &(dx, dy, weight) in &LPS_NEIGHBORS {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let nx = nx as usize;
            let ny = ny as usize;
            if nx >= width || ny >= height {
                continue;
            }

            let nidx = ny * width + nx;
            if processed[nidx] {
                continue;
            }

            available[available_len] = (nidx, weight);
            available_len += 1;
            available_weight_sum += weight;
        }

        if available_len > 0 && available_weight_sum > 0.0 {
            for &(nidx, weight) in available.iter().take(available_len) {
                errors[nidx] += residual * (weight / available_weight_sum);
            }
        }
    }

    write_multiscale_output(buffer, &data)
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

fn diffuse_hvs_optimized_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
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
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);
                let residual = adjusted - quantized[0].to_unit_f32();
                let xi = x as isize;
                let yi = y as isize;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi,
                    [residual * HVS_OPTIMIZED_FORWARD, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    [residual * HVS_OPTIMIZED_DOWN_DIAGONAL, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    [residual * HVS_OPTIMIZED_DOWN, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    [residual * HVS_OPTIMIZED_DOWN_FORWARD, 0.0, 0.0, 0.0],
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
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);
                let residual = adjusted - quantized[0].to_unit_f32();
                let xi = x as isize;
                let yi = y as isize;

                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi,
                    [residual * HVS_OPTIMIZED_FORWARD, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi - 1,
                    yi + 1,
                    [residual * HVS_OPTIMIZED_DOWN_DIAGONAL, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi,
                    yi + 1,
                    [residual * HVS_OPTIMIZED_DOWN, 0.0, 0.0, 0.0],
                );
                diffuse_error_forward::<L>(
                    &mut errors,
                    width,
                    height,
                    xi + 1,
                    yi + 1,
                    [residual * HVS_OPTIMIZED_DOWN_FORWARD, 0.0, 0.0, 0.0],
                );
            }
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

fn diffuse_tone_dependent_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
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
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = adjusted - quantized[0].to_unit_f32();
                let luma = luma_bucket_unit(adjusted);
                let coeff = tone_dependent_coeff_for_luma(luma)?;
                let tone = f32::from(luma) / 255.0;
                let center_bias = 1.0 - (2.0 * tone - 1.0).abs();
                let gain = 0.7 + 0.6 * center_bias;
                let forward = residual * coeff[0] * gain;
                let down_diag = residual * coeff[1] * gain;
                let down = residual * coeff[2] * gain;
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
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = adjusted - quantized[0].to_unit_f32();
                let luma = luma_bucket_unit(adjusted);
                let coeff = tone_dependent_coeff_for_luma(luma)?;
                let tone = f32::from(luma) / 255.0;
                let center_bias = 1.0 - (2.0 * tone - 1.0).abs();
                let gain = 0.7 + 0.6 * center_bias;
                let forward = residual * coeff[0] * gain;
                let down_diag = residual * coeff[1] * gain;
                let down = residual * coeff[2] * gain;
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

fn diffuse_structure_aware_gray<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    let width = buffer.width;
    let height = buffer.height;
    let source = read_gray_units(buffer)?;
    let expected = checked_gray_len(width, height)?;
    if source.len() != expected {
        return Err(Error::InvalidArgument(
            "structure-aware diffusion source shape mismatch",
        ));
    }
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
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = adjusted - quantized[0].to_unit_f32();
                let luma = luma_bucket_unit(adjusted);
                let structure = local_structure_strength_from_units(&source, width, height, x, y)?;
                let coeff = structure_aware_weights_for_pixel(luma, structure)?;
                let forward = residual * coeff[0];
                let down_diag = residual * coeff[1];
                let down = residual * coeff[2];
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
                let quantized =
                    quantize_pixel::<S, crate::core::Gray>(&[S::from_unit_f32(adjusted)], mode)?;
                write_quantized_pixel::<S, L>(pixel, quantized);

                let residual = adjusted - quantized[0].to_unit_f32();
                let luma = luma_bucket_unit(adjusted);
                let structure = local_structure_strength_from_units(&source, width, height, x, y)?;
                let coeff = structure_aware_weights_for_pixel(luma, structure)?;
                let forward = residual * coeff[0];
                let down_diag = residual * coeff[1];
                let down = residual * coeff[2];
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

fn tone_dependent_coeff_for_luma(luma: u8) -> Result<[f32; 3]> {
    let table_len = TONE_DEPENDENT_DIFFUSION_COEFFS.len();
    if table_len == 0 {
        return Err(Error::InvalidArgument(
            "tone-dependent diffusion coefficient table must be non-empty",
        ));
    }

    let idx = (usize::from(luma) * table_len) / 256;
    let coeff = TONE_DEPENDENT_DIFFUSION_COEFFS
        .get(idx)
        .ok_or(Error::InvalidArgument(
            "tone-dependent diffusion coefficient lookup out of range",
        ))?;

    let sum = coeff[0] + coeff[1] + coeff[2];
    if !sum.is_finite() || sum <= f32::EPSILON {
        return Err(Error::InvalidArgument(
            "tone-dependent diffusion coefficients must sum to a finite positive value",
        ));
    }

    Ok([coeff[0] / sum, coeff[1] / sum, coeff[2] / sum])
}

fn structure_aware_weights_for_pixel(luma: u8, structure: f32) -> Result<[f32; 3]> {
    let base = coefficient_for_luma(luma);
    let den = f32::from(base.3);
    if !den.is_finite() || den <= f32::EPSILON {
        return Err(Error::InvalidArgument(
            "structure-aware diffusion denominator must be positive",
        ));
    }

    let forward_base = f32::from(base.0) / den;
    let down_diag_base = f32::from(base.1) / den;
    let down_base = f32::from(base.2) / den;

    let structure_clamped = structure.clamp(0.0, 1.0);
    let tone = f32::from(luma) / 255.0;
    let tone_center = 1.0 - (2.0 * tone - 1.0).abs();
    let flat = 1.0 - structure_clamped;

    let forward = forward_base * (1.0 - STRUCTURE_AWARE_FORWARD_REDUCTION * structure_clamped);
    let down_diag = down_diag_base * (1.0 + STRUCTURE_AWARE_DIAGONAL_BOOST * flat);
    let down = down_base
        * (1.0
            + STRUCTURE_AWARE_DOWN_BOOST * structure_clamped
            + STRUCTURE_AWARE_TONE_CENTER_BOOST * tone_center);

    let sum = forward + down_diag + down;
    if !sum.is_finite() || sum <= f32::EPSILON {
        return Err(Error::InvalidArgument(
            "structure-aware diffusion coefficient normalization failed",
        ));
    }

    let gain = (STRUCTURE_AWARE_GAIN_MIN + STRUCTURE_AWARE_GAIN_SCALE * structure_clamped).clamp(
        STRUCTURE_AWARE_GAIN_MIN,
        STRUCTURE_AWARE_GAIN_MIN + STRUCTURE_AWARE_GAIN_SCALE,
    );

    Ok([
        forward / sum * gain,
        down_diag / sum * gain,
        down / sum * gain,
    ])
}

fn local_structure_strength_from_units(
    source: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
) -> Result<f32> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidArgument(
            "structure-aware diffusion requires non-empty dimensions",
        ));
    }
    let expected = checked_gray_len(width, height)?;
    if source.len() != expected {
        return Err(Error::InvalidArgument(
            "structure-aware diffusion source dimensions mismatch",
        ));
    }

    let idx = y
        .checked_mul(width)
        .and_then(|base| base.checked_add(x))
        .ok_or(Error::InvalidArgument(
            "structure-aware diffusion index overflow",
        ))?;
    let center = *source.get(idx).ok_or(Error::InvalidArgument(
        "structure-aware diffusion center index out of range",
    ))?;

    let sample = |sx: usize, sy: usize| -> f32 { source[sy * width + sx] };
    let left = sample(x.saturating_sub(1), y);
    let right = sample((x + 1).min(width - 1), y);
    let up = sample(x, y.saturating_sub(1));
    let down = sample(x, (y + 1).min(height - 1));
    let nw = sample(x.saturating_sub(1), y.saturating_sub(1));
    let ne = sample((x + 1).min(width - 1), y.saturating_sub(1));
    let sw = sample(x.saturating_sub(1), (y + 1).min(height - 1));
    let se = sample((x + 1).min(width - 1), (y + 1).min(height - 1));

    let gx = (right - left).abs();
    let gy = (down - up).abs();
    let diag = ((se - nw).abs() + (sw - ne).abs()) * 0.5;
    let local_var = ((left - center).abs()
        + (right - center).abs()
        + (up - center).abs()
        + (down - center).abs())
        * 0.25;

    Ok((0.45 * (gx + gy) + 0.30 * diag + 0.25 * local_var).clamp(0.0, 1.0))
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

fn linear_pixel_shuffling_permutation(
    width: usize,
    height: usize,
    seed: u64,
) -> Result<Vec<usize>> {
    let len = checked_gray_len(width, height)?;
    if len == 0 {
        return Err(Error::InvalidArgument(
            "linear pixel shuffling requires non-empty image",
        ));
    }
    if len == 1 {
        return Ok(vec![0]);
    }

    let mixed = splitmix64(seed ^ (width as u64).rotate_left(17) ^ (height as u64).rotate_left(49));
    let stride = coprime_stride(len, mixed as usize)?;
    let offset = (splitmix64(mixed ^ 0xA5A5_A5A5_A5A5_A5A5) as usize) % len;
    let mut order = Vec::with_capacity(len);
    let mut visited = vec![false; len];

    let mut idx = offset;
    for _ in 0..len {
        if visited[idx] {
            return Err(Error::InvalidArgument(
                "linear pixel shuffling permutation generation failed",
            ));
        }
        visited[idx] = true;
        order.push(idx);
        idx = (idx + stride) % len;
    }

    Ok(order)
}

fn coprime_stride(len: usize, candidate: usize) -> Result<usize> {
    if len <= 1 {
        return Ok(1);
    }

    let mut stride = (candidate % len).max(1);
    let mut attempts = 0_usize;
    while gcd_usize(stride, len) != 1 {
        stride += 1;
        if stride >= len {
            stride = 1;
        }
        attempts += 1;
        if attempts > len {
            return Err(Error::InvalidArgument(
                "linear pixel shuffling failed to derive coprime stride",
            ));
        }
    }

    Ok(stride)
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn update_adaptive_vector_weights(
    weights: &mut [f32; 4],
    residual: [f32; 3],
    previous_residual: [f32; 3],
) -> Result<()> {
    let residual_norm = vector_norm_sq(residual);
    let previous_norm = vector_norm_sq(previous_residual);

    if residual_norm > ADAPTIVE_VECTOR_EPSILON && previous_norm > ADAPTIVE_VECTOR_EPSILON {
        let correlation = vector_dot(residual, previous_residual)
            / ((residual_norm * previous_norm).sqrt() + ADAPTIVE_VECTOR_EPSILON);
        let chroma = ((residual[0] - residual[1]).abs()
            + (residual[1] - residual[2]).abs()
            + (residual[2] - residual[0]).abs())
            / 3.0;
        let step = ADAPTIVE_VECTOR_LEARNING_RATE
            * correlation
            * residual_norm.sqrt().clamp(0.0, 1.0)
            * (1.0 + 0.5 * chroma.clamp(0.0, 1.0));
        weights[0] += step;
        weights[2] += 0.5 * step;
        weights[1] -= 0.75 * step;
        weights[3] -= 0.75 * step;
    }

    for (idx, weight) in weights.iter_mut().enumerate() {
        *weight += (ADAPTIVE_VECTOR_BASE_WEIGHTS[idx] - *weight) * ADAPTIVE_VECTOR_RELAX_RATE;
        *weight = weight.clamp(ADAPTIVE_VECTOR_MIN_WEIGHT, ADAPTIVE_VECTOR_MAX_WEIGHT);
    }

    project_adaptive_vector_weights(weights)?;
    validate_adaptive_vector_weights(*weights)
}

fn project_adaptive_vector_weights(weights: &mut [f32; 4]) -> Result<()> {
    for weight in weights.iter_mut() {
        if !weight.is_finite() {
            return Err(Error::InvalidArgument(
                "adaptive vector coefficients must remain finite",
            ));
        }
        *weight = weight.clamp(ADAPTIVE_VECTOR_MIN_WEIGHT, ADAPTIVE_VECTOR_MAX_WEIGHT);
    }

    for _ in 0..8 {
        let sum = weights.iter().copied().sum::<f32>();
        if !sum.is_finite() {
            return Err(Error::InvalidArgument(
                "adaptive vector coefficient normalization failed",
            ));
        }
        let delta = 1.0 - sum;
        if delta.abs() <= 1.0e-6 {
            return Ok(());
        }

        if delta > 0.0 {
            let room = weights
                .iter()
                .map(|weight| ADAPTIVE_VECTOR_MAX_WEIGHT - *weight)
                .sum::<f32>();
            if room <= ADAPTIVE_VECTOR_EPSILON {
                return Err(Error::InvalidArgument(
                    "adaptive vector coefficient normalization failed",
                ));
            }
            for weight in weights.iter_mut() {
                let avail = ADAPTIVE_VECTOR_MAX_WEIGHT - *weight;
                *weight += delta * (avail / room);
            }
        } else {
            let room = weights
                .iter()
                .map(|weight| *weight - ADAPTIVE_VECTOR_MIN_WEIGHT)
                .sum::<f32>();
            if room <= ADAPTIVE_VECTOR_EPSILON {
                return Err(Error::InvalidArgument(
                    "adaptive vector coefficient normalization failed",
                ));
            }
            for weight in weights.iter_mut() {
                let avail = *weight - ADAPTIVE_VECTOR_MIN_WEIGHT;
                *weight += delta * (avail / room);
            }
        }

        for weight in weights.iter_mut() {
            *weight = weight.clamp(ADAPTIVE_VECTOR_MIN_WEIGHT, ADAPTIVE_VECTOR_MAX_WEIGHT);
        }
    }

    let final_sum = weights.iter().copied().sum::<f32>();
    if (final_sum - 1.0).abs() > 1.0e-4 {
        return Err(Error::InvalidArgument(
            "adaptive vector coefficient normalization failed",
        ));
    }

    Ok(())
}

fn validate_adaptive_vector_weights(weights: [f32; 4]) -> Result<()> {
    let mut sum = 0.0_f32;
    for weight in weights {
        if !weight.is_finite() {
            return Err(Error::InvalidArgument(
                "adaptive vector coefficients must remain finite",
            ));
        }
        if !(ADAPTIVE_VECTOR_MIN_WEIGHT..=ADAPTIVE_VECTOR_MAX_WEIGHT).contains(&weight) {
            return Err(Error::InvalidArgument(
                "adaptive vector coefficients out of bounds",
            ));
        }
        sum += weight;
    }

    if (sum - 1.0).abs() > 1.0e-4 {
        return Err(Error::InvalidArgument(
            "adaptive vector coefficients must sum to one",
        ));
    }

    Ok(())
}

fn vector_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vector_norm_sq(a: [f32; 3]) -> f32 {
    vector_dot(a, a)
}
