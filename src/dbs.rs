use crate::{
    core::{PixelLayout, Sample},
    data::CLUSTER_DOT_8X8_FLAT,
    Buffer, Error, Result,
};
use std::mem::size_of;

const DBS_HVS_RADIUS: usize = 3;
const DBS_HVS_SIZE: usize = DBS_HVS_RADIUS * 2 + 1;
const DBS_HVS_SIGMA: f64 = 1.0;
const DBS_SWAP_NEIGHBORS: [(isize, isize); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];
const LBM_DIRECTIONS: [(isize, isize); 9] = [
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
];
const LBM_T_WEIGHTS: [f64; 9] = [
    4.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 9.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];
const LBM_CONVERGENCE_EPSILON: f64 = 1e-6;
const ELECTRO_FORCE_EPSILON: f64 = 1e-6;
const ELECTRO_REPULSION_WEIGHT: f64 = 1.0;
const ELECTRO_ATTRACTION_WEIGHT: f64 = 1.0;
const ELECTRO_STEP_SIZE: f64 = 0.2;
const ELECTRO_SHAKE_BASE: f64 = 0.35;
const ELECTRO_CONVERGENCE_EPSILON: f64 = 1e-3;
const MODEL_PRINTER_KERNEL: [f32; 9] = [
    0.025, 0.07, 0.025, //
    0.07, 0.62, 0.07, //
    0.025, 0.07, 0.025,
];
const MODEL_EYE_RADIUS: usize = 3;
const MODEL_EYE_SIZE: usize = MODEL_EYE_RADIUS * 2 + 1;
const MODEL_EYE_SIGMA: f64 = 1.2;
const MODEL_MED_ERROR_LIMIT: f32 = 1.0;
const HIERARCHICAL_COLORANT_GROUP_MASKS: [usize; 7] =
    [0b111, 0b011, 0b101, 0b110, 0b001, 0b010, 0b100];

pub fn direct_binary_search_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_iters: usize,
) -> Result<()> {
    buffer.validate()?;
    ensure_grayscale_integer_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;

    let mut target_unit = Vec::with_capacity(pixel_count);
    let mut binary = Vec::with_capacity(pixel_count);

    for y in 0..height {
        let row = buffer.try_row(y)?;
        for &value in row.iter().take(width) {
            let unit = value.to_unit_f32().clamp(0.0, 1.0);
            target_unit.push(unit);
            binary.push(if unit >= 0.5 { 1_u8 } else { 0_u8 });
        }
    }

    let hvs = dbs_hvs_filter();
    let mut filtered_error =
        dbs_filtered_error_map(&target_unit, &binary, width, height, &hvs, DBS_HVS_RADIUS);

    for _ in 0..max_iters {
        let mut improved = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let mut best_delta = 0.0_f64;
                let mut best_move = CandidateMove::None;

                let toggle_delta = dbs_candidate_delta_energy(
                    width,
                    height,
                    &hvs,
                    DBS_HVS_RADIUS,
                    &filtered_error,
                    &[(x, y, toggle_delta_value(binary[idx]))],
                );
                if toggle_delta < best_delta {
                    best_delta = toggle_delta;
                    best_move = CandidateMove::Toggle(idx);
                }

                for (dx, dy) in DBS_SWAP_NEIGHBORS {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                        continue;
                    }
                    let nidx = ny as usize * width + nx as usize;
                    if binary[idx] == binary[nidx] {
                        continue;
                    }
                    let delta_primary = swap_primary_delta(binary[idx], binary[nidx]);
                    let swap_delta = dbs_candidate_delta_energy(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &filtered_error,
                        &[
                            (x, y, delta_primary),
                            (nx as usize, ny as usize, -delta_primary),
                        ],
                    );
                    if swap_delta < best_delta {
                        best_delta = swap_delta;
                        best_move = CandidateMove::Swap(idx, nidx);
                    }
                }

                match best_move {
                    CandidateMove::None => {}
                    CandidateMove::Toggle(i) => {
                        let px = i % width;
                        let py = i / width;
                        let delta = toggle_delta_value(binary[i]);
                        binary[i] ^= 1;
                        dbs_apply_delta_filtered_error(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &mut filtered_error,
                            &[(px, py, delta)],
                        );
                        improved = true;
                    }
                    CandidateMove::Swap(i, j) => {
                        let ix = i % width;
                        let iy = i / width;
                        let jx = j % width;
                        let jy = j / width;
                        let delta_primary = swap_primary_delta(binary[i], binary[j]);
                        binary.swap(i, j);
                        dbs_apply_delta_filtered_error(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &mut filtered_error,
                            &[(ix, iy, delta_primary), (jx, jy, -delta_primary)],
                        );
                        improved = true;
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }

    for y in 0..height {
        let start = y * width;
        let row = buffer.try_row_mut(y)?;
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let domain = if binary[start + x] == 0 { 0 } else { max_value };
            *value = domain_to_sample(domain, max_value);
        }
    }

    Ok(())
}

pub fn clustered_dot_direct_multibit_search_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_iters: usize,
    levels: u16,
) -> Result<()> {
    buffer.validate()?;
    ensure_grayscale_integer_format::<S, L>()?;
    if levels < 2 {
        return Err(Error::InvalidArgument(
            "clustered-dot direct multibit search requires at least 2 levels",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;
    let capacity = integer_sample_capacity::<S>()?;
    if usize::from(levels) > capacity {
        return Err(Error::InvalidArgument(
            "clustered-dot direct multibit search levels exceed sample capacity",
        ));
    }

    let target_unit = grayscale_target_unit(buffer, width, height, pixel_count)?;
    let level_values = multibit_level_values(levels);
    let mut levels_map = clustered_dot_initial_levels(&target_unit, width, height, levels);
    let mut quantized = levels_to_unit_values(&levels_map, &level_values);
    let hvs = dbs_hvs_filter();
    let mut filtered_error = dbs_filtered_error_map_levels(
        &target_unit,
        &quantized,
        width,
        height,
        &hvs,
        DBS_HVS_RADIUS,
    );

    for _ in 0..max_iters {
        let mut improved = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let current = usize::from(levels_map[idx]);
                let mut best_delta = 0.0_f64;
                let mut best_level = current;

                for candidate in candidate_levels(current, level_values.len()) {
                    let delta = level_values[candidate] - level_values[current];
                    if delta == 0.0 {
                        continue;
                    }

                    let candidate_delta = dbs_candidate_delta_energy(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &filtered_error,
                        &[(x, y, delta)],
                    );
                    if candidate_delta < best_delta {
                        best_delta = candidate_delta;
                        best_level = candidate;
                    }
                }

                if best_level != current {
                    let delta = level_values[best_level] - level_values[current];
                    levels_map[idx] = best_level as u16;
                    quantized[idx] = level_values[best_level];
                    dbs_apply_delta_filtered_error(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &mut filtered_error,
                        &[(x, y, delta)],
                    );
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
    }

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let domain = level_index_to_domain(i32::from(levels_map[idx]), max_value, levels);
            *value = domain_to_sample(domain, max_value);
        }
    }

    Ok(())
}

pub fn direct_pattern_control_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_iters: usize,
) -> Result<()> {
    buffer.validate()?;
    ensure_rgb_integer_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;

    let target = rgb_target_unit(buffer, width, height, pixel_count)?;
    let mut states = target
        .iter()
        .map(|&rgb| nearest_primary_index(rgb))
        .collect::<Vec<usize>>();
    let mut output = states_to_primary_unit(&states);
    let target_r = target.iter().map(|rgb| rgb[0]).collect::<Vec<f32>>();
    let target_g = target.iter().map(|rgb| rgb[1]).collect::<Vec<f32>>();
    let target_b = target.iter().map(|rgb| rgb[2]).collect::<Vec<f32>>();

    let hvs = dbs_hvs_filter();
    let mut filtered_error_r =
        dbs_filtered_error_map_levels(&target_r, &output.0, width, height, &hvs, DBS_HVS_RADIUS);
    let mut filtered_error_g =
        dbs_filtered_error_map_levels(&target_g, &output.1, width, height, &hvs, DBS_HVS_RADIUS);
    let mut filtered_error_b =
        dbs_filtered_error_map_levels(&target_b, &output.2, width, height, &hvs, DBS_HVS_RADIUS);

    for _ in 0..max_iters {
        let mut improved = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let current = states[idx];
                let current_rgb = primary_unit(current);
                let mut best_state = current;
                let mut best_delta = 0.0_f64;

                for candidate in primary_candidates(current) {
                    if candidate == current {
                        continue;
                    }

                    let next_rgb = primary_unit(candidate);
                    let dr = next_rgb[0] - current_rgb[0];
                    let dg = next_rgb[1] - current_rgb[1];
                    let db = next_rgb[2] - current_rgb[2];

                    let delta_r = dbs_candidate_delta_energy(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &filtered_error_r,
                        &[(x, y, dr)],
                    );
                    let delta_g = dbs_candidate_delta_energy(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &filtered_error_g,
                        &[(x, y, dg)],
                    );
                    let delta_b = dbs_candidate_delta_energy(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &filtered_error_b,
                        &[(x, y, db)],
                    );
                    let delta = delta_r + delta_g + delta_b;
                    if delta < best_delta {
                        best_delta = delta;
                        best_state = candidate;
                    }
                }

                if best_state != current {
                    let next_rgb = primary_unit(best_state);
                    let dr = next_rgb[0] - current_rgb[0];
                    let dg = next_rgb[1] - current_rgb[1];
                    let db = next_rgb[2] - current_rgb[2];

                    states[idx] = best_state;
                    output.0[idx] = next_rgb[0];
                    output.1[idx] = next_rgb[1];
                    output.2[idx] = next_rgb[2];

                    dbs_apply_delta_filtered_error(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &mut filtered_error_r,
                        &[(x, y, dr)],
                    );
                    dbs_apply_delta_filtered_error(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &mut filtered_error_g,
                        &[(x, y, dg)],
                    );
                    dbs_apply_delta_filtered_error(
                        width,
                        height,
                        &hvs,
                        DBS_HVS_RADIUS,
                        &mut filtered_error_b,
                        &[(x, y, db)],
                    );
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
    }

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for x in 0..width {
            let idx = y * width + x;
            let state = states[idx];
            let rgb = primary_domain(state, max_value);
            let base = x * L::CHANNELS;
            row[base] = domain_to_sample(rgb[0], max_value);
            row[base + 1] = domain_to_sample(rgb[1], max_value);
            row[base + 2] = domain_to_sample(rgb[2], max_value);
        }
    }

    Ok(())
}

pub fn hierarchical_colorant_dbs_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_iters: usize,
) -> Result<()> {
    buffer.validate()?;
    ensure_rgb_integer_opaque_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;

    let target = rgb_target_unit(buffer, width, height, pixel_count)?;
    let regions = target
        .iter()
        .copied()
        .map(mbvq_region_for_rgb)
        .collect::<Vec<MbvqRegion>>();
    let mut states = target
        .iter()
        .zip(regions.iter().copied())
        .map(|(&rgb, region)| nearest_region_state(rgb, region))
        .collect::<Vec<usize>>();
    let mut output = states_to_primary_unit(&states);
    let target_r = target.iter().map(|rgb| rgb[0]).collect::<Vec<f32>>();
    let target_g = target.iter().map(|rgb| rgb[1]).collect::<Vec<f32>>();
    let target_b = target.iter().map(|rgb| rgb[2]).collect::<Vec<f32>>();

    let hvs = dbs_hvs_filter();
    let mut filtered_error_r =
        dbs_filtered_error_map_levels(&target_r, &output.0, width, height, &hvs, DBS_HVS_RADIUS);
    let mut filtered_error_g =
        dbs_filtered_error_map_levels(&target_g, &output.1, width, height, &hvs, DBS_HVS_RADIUS);
    let mut filtered_error_b =
        dbs_filtered_error_map_levels(&target_b, &output.2, width, height, &hvs, DBS_HVS_RADIUS);

    for _ in 0..max_iters {
        let mut improved = false;

        for group_mask in HIERARCHICAL_COLORANT_GROUP_MASKS {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    let current = states[idx];
                    let current_rgb = primary_unit(current);
                    let mut best_state = current;
                    let mut best_delta = 0.0_f64;
                    let region = regions[idx];

                    for candidate in mbvq_region_states(region) {
                        if candidate == current
                            || ((current ^ candidate) & (!group_mask & 0b111)) != 0
                        {
                            continue;
                        }

                        let next_rgb = primary_unit(candidate);
                        let dr = next_rgb[0] - current_rgb[0];
                        let dg = next_rgb[1] - current_rgb[1];
                        let db = next_rgb[2] - current_rgb[2];

                        let delta_r = dbs_candidate_delta_energy(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &filtered_error_r,
                            &[(x, y, dr)],
                        );
                        let delta_g = dbs_candidate_delta_energy(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &filtered_error_g,
                            &[(x, y, dg)],
                        );
                        let delta_b = dbs_candidate_delta_energy(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &filtered_error_b,
                            &[(x, y, db)],
                        );
                        let delta = delta_r + delta_g + delta_b;
                        if delta < best_delta {
                            best_delta = delta;
                            best_state = candidate;
                        }
                    }

                    if best_state != current {
                        let next_rgb = primary_unit(best_state);
                        let dr = next_rgb[0] - current_rgb[0];
                        let dg = next_rgb[1] - current_rgb[1];
                        let db = next_rgb[2] - current_rgb[2];

                        states[idx] = best_state;
                        output.0[idx] = next_rgb[0];
                        output.1[idx] = next_rgb[1];
                        output.2[idx] = next_rgb[2];

                        dbs_apply_delta_filtered_error(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &mut filtered_error_r,
                            &[(x, y, dr)],
                        );
                        dbs_apply_delta_filtered_error(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &mut filtered_error_g,
                            &[(x, y, dg)],
                        );
                        dbs_apply_delta_filtered_error(
                            width,
                            height,
                            &hvs,
                            DBS_HVS_RADIUS,
                            &mut filtered_error_b,
                            &[(x, y, db)],
                        );
                        improved = true;
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for x in 0..width {
            let idx = y * width + x;
            let state = states[idx];
            let rgb = primary_domain(state, max_value);
            let base = x * L::CHANNELS;
            row[base] = domain_to_sample(rgb[0], max_value);
            row[base + 1] = domain_to_sample(rgb[1], max_value);
            row[base + 2] = domain_to_sample(rgb[2], max_value);
        }
    }

    Ok(())
}

pub fn lattice_boltzmann_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_steps: usize,
) -> Result<()> {
    buffer.validate()?;
    ensure_grayscale_integer_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;
    let max_value_f64 = max_value as f64;

    let mut state = Vec::with_capacity(pixel_count);
    for y in 0..height {
        let row = buffer.try_row(y)?;
        for &value in row.iter().take(width) {
            state.push(f64::from(value.to_unit_f32().clamp(0.0, 1.0)) * max_value_f64);
        }
    }

    let total_gray = state.iter().sum::<f64>();
    if total_gray > 0.0 {
        let white_count_target =
            ((total_gray / max_value_f64).round() as isize).clamp(0, pixel_count as isize) as usize;
        let normalized_sum = white_count_target as f64 * max_value_f64;
        let scale = normalized_sum / total_gray;
        for value in &mut state {
            *value *= scale;
        }
    }

    let low_threshold = 1.0 / max_value_f64;
    let convergence_limit = LBM_CONVERGENCE_EPSILON * max_value_f64 * (pixel_count as f64).sqrt();
    let mut next = vec![0.0_f64; pixel_count];
    let mut reference_state = [0.0_f64; 9];

    for _ in 0..max_steps {
        next.fill(0.0);

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let center = state[idx];
                lbm_reference_state_for_pixel(
                    &state,
                    (width, height),
                    (x, y),
                    center,
                    max_value_f64,
                    low_threshold,
                    &mut reference_state,
                );

                for d in 0..9 {
                    let contribution = reference_state[d];
                    if contribution <= 0.0 {
                        continue;
                    }

                    let (dx, dy) = LBM_DIRECTIONS[d];
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                        continue;
                    }

                    let nidx = ny as usize * width + nx as usize;
                    next[nidx] += contribution;
                }
            }
        }

        let mut l2 = 0.0_f64;
        for i in 0..pixel_count {
            let delta = next[i] - state[i];
            l2 += delta * delta;
        }
        l2 = l2.sqrt();

        state.copy_from_slice(&next);
        if l2 < convergence_limit {
            break;
        }
    }

    let total_after = state.iter().sum::<f64>();
    let white_count =
        ((total_after / max_value_f64).round() as isize).clamp(0, pixel_count as isize) as usize;

    let mut ranking = state.iter().enumerate().collect::<Vec<(usize, &f64)>>();
    ranking.sort_by(|(ia, va), (ib, vb)| vb.total_cmp(va).then_with(|| ia.cmp(ib)));

    let mut white = vec![false; pixel_count];
    for (rank, (idx, _)) in ranking.into_iter().enumerate() {
        if rank >= white_count {
            break;
        }
        white[idx] = true;
    }

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let domain = if white[idx] { max_value } else { 0 };
            *value = domain_to_sample(domain, max_value);
        }
    }

    Ok(())
}

pub fn electrostatic_halftoning_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_steps: usize,
) -> Result<()> {
    buffer.validate()?;
    ensure_grayscale_integer_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;
    let mut darkness = Vec::with_capacity(pixel_count);

    for y in 0..height {
        let row = buffer.try_row(y)?;
        for &value in row.iter().take(width) {
            darkness.push(1.0 - f64::from(value.to_unit_f32().clamp(0.0, 1.0)));
        }
    }

    let mut particle_count = darkness.iter().sum::<f64>().round() as usize;
    if particle_count > pixel_count {
        particle_count = pixel_count;
    }
    if particle_count == 0 {
        for y in 0..height {
            let row = buffer.try_row_mut(y)?;
            for value in row.iter_mut().take(width) {
                *value = domain_to_sample(max_value, max_value);
            }
        }
        return Ok(());
    }

    let forcefield = electrostatic_forcefield(&darkness, width, height);
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    let mut positions =
        electrostatic_initial_positions(&darkness, width, height, particle_count, &mut rng);
    let mut forces = vec![[0.0_f64; 2]; particle_count];

    for step in 0..max_steps {
        electrostatic_compute_forces(&positions, &forcefield, width, height, &mut forces);

        let mut max_displacement = 0.0_f64;
        let shake = if max_steps == 0 {
            0.0
        } else {
            ELECTRO_SHAKE_BASE * (1.0 - step as f64 / max_steps as f64)
        };
        let apply_shake = step % 10 == 0 && shake > 0.0;

        for i in 0..positions.len() {
            let mut next_x = positions[i][0] + ELECTRO_STEP_SIZE * forces[i][0];
            let mut next_y = positions[i][1] + ELECTRO_STEP_SIZE * forces[i][1];

            if apply_shake {
                next_x += shake * (electrostatic_rand_unit(&mut rng) * 2.0 - 1.0);
                next_y += shake * (electrostatic_rand_unit(&mut rng) * 2.0 - 1.0);
            }

            next_x = next_x.clamp(0.0, (width - 1) as f64);
            next_y = next_y.clamp(0.0, (height - 1) as f64);

            let dx = next_x - positions[i][0];
            let dy = next_y - positions[i][1];
            let displacement = (dx * dx + dy * dy).sqrt();
            if displacement > max_displacement {
                max_displacement = displacement;
            }

            positions[i][0] = next_x;
            positions[i][1] = next_y;
        }

        if max_displacement < ELECTRO_CONVERGENCE_EPSILON {
            break;
        }
    }

    let occupied = electrostatic_project_to_grid(&positions, width, height);

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let domain = if occupied[idx] { 0 } else { max_value };
            *value = domain_to_sample(domain, max_value);
        }
    }

    Ok(())
}

pub fn model_based_med_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
) -> Result<()> {
    buffer.validate()?;
    ensure_grayscale_integer_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;
    let target = grayscale_target_unit(buffer, width, height, pixel_count)?;
    let binary = model_based_med_binary(&target, width, height);

    write_binary_output(buffer, &binary, width, height, max_value)
}

pub fn least_squares_model_based_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    max_iters: usize,
) -> Result<()> {
    buffer.validate()?;
    ensure_grayscale_integer_format::<S, L>()?;

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let max_value = integer_sample_max::<S>()?;
    let target = grayscale_target_unit(buffer, width, height, pixel_count)?;
    let mut binary = model_based_med_binary(&target, width, height);
    let eye = model_based_eye_filter();
    let kernel = model_based_kernel(&eye);
    let kernel_size = MODEL_EYE_SIZE + 2;
    let kernel_radius = kernel_size / 2;
    let mut filtered_error =
        model_based_filtered_error_map(&target, &binary, width, height, &kernel, kernel_size);

    for _ in 0..max_iters {
        let mut improved = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let mut best_delta = 0.0_f64;
                let mut best_move = CandidateMove::None;

                let toggle_delta = model_based_candidate_delta_energy(
                    width,
                    height,
                    kernel_radius,
                    &kernel,
                    kernel_size,
                    &filtered_error,
                    &[(x, y, toggle_delta_value(binary[idx]))],
                );
                if toggle_delta < best_delta {
                    best_delta = toggle_delta;
                    best_move = CandidateMove::Toggle(idx);
                }

                for (dx, dy) in DBS_SWAP_NEIGHBORS {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                        continue;
                    }
                    let nidx = ny as usize * width + nx as usize;
                    if binary[idx] == binary[nidx] {
                        continue;
                    }

                    let delta_primary = swap_primary_delta(binary[idx], binary[nidx]);
                    let swap_delta = model_based_candidate_delta_energy(
                        width,
                        height,
                        kernel_radius,
                        &kernel,
                        kernel_size,
                        &filtered_error,
                        &[
                            (x, y, delta_primary),
                            (nx as usize, ny as usize, -delta_primary),
                        ],
                    );
                    if swap_delta < best_delta {
                        best_delta = swap_delta;
                        best_move = CandidateMove::Swap(idx, nidx);
                    }
                }

                match best_move {
                    CandidateMove::None => {}
                    CandidateMove::Toggle(i) => {
                        let px = i % width;
                        let py = i / width;
                        let delta = toggle_delta_value(binary[i]);
                        binary[i] ^= 1;
                        model_based_apply_delta_filtered_error(
                            width,
                            height,
                            kernel_radius,
                            &kernel,
                            kernel_size,
                            &mut filtered_error,
                            &[(px, py, delta)],
                        );
                        improved = true;
                    }
                    CandidateMove::Swap(i, j) => {
                        let ix = i % width;
                        let iy = i / width;
                        let jx = j % width;
                        let jy = j / width;
                        let delta_primary = swap_primary_delta(binary[i], binary[j]);
                        binary.swap(i, j);
                        model_based_apply_delta_filtered_error(
                            width,
                            height,
                            kernel_radius,
                            &kernel,
                            kernel_size,
                            &mut filtered_error,
                            &[(ix, iy, delta_primary), (jx, jy, -delta_primary)],
                        );
                        improved = true;
                    }
                }
            }
        }

        if !improved {
            break;
        }
    }

    write_binary_output(buffer, &binary, width, height, max_value)
}

fn grayscale_target_unit<S: Sample, L: PixelLayout>(
    buffer: &Buffer<'_, S, L>,
    width: usize,
    height: usize,
    pixel_count: usize,
) -> Result<Vec<f32>> {
    let mut target = Vec::with_capacity(pixel_count);
    for y in 0..height {
        let row = buffer.try_row(y)?;
        for &value in row.iter().take(width) {
            target.push(value.to_unit_f32().clamp(0.0, 1.0));
        }
    }
    Ok(target)
}

fn integer_sample_capacity<S: Sample>() -> Result<usize> {
    match size_of::<S>() {
        1 => Ok(256),
        2 => Ok(65_536),
        _ => Err(Error::UnsupportedFormat(
            "unsupported integer sample width for research algorithms",
        )),
    }
}

fn multibit_level_values(levels: u16) -> Vec<f32> {
    let levels_m1 = f32::from(levels.saturating_sub(1));
    (0..levels)
        .map(|level| f32::from(level) / levels_m1)
        .collect()
}

fn clustered_dot_initial_levels(
    target_unit: &[f32],
    width: usize,
    height: usize,
    levels: u16,
) -> Vec<u16> {
    let levels_m1 = usize::from(levels.saturating_sub(1));
    let mut out = Vec::with_capacity(target_unit.len());

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let scaled = target_unit[idx].clamp(0.0, 1.0) * levels_m1 as f32;
            let base = scaled.floor().clamp(0.0, levels_m1 as f32) as usize;
            let frac = scaled - base as f32;
            let rank = CLUSTER_DOT_8X8_FLAT[(y % 8) * 8 + (x % 8)] as f32;
            let threshold = (rank + 0.5) / 64.0;
            let level = if frac > threshold && base < levels_m1 {
                base + 1
            } else {
                base
            };
            out.push(level as u16);
        }
    }

    out
}

fn levels_to_unit_values(levels_map: &[u16], level_values: &[f32]) -> Vec<f32> {
    levels_map
        .iter()
        .map(|&level| level_values[usize::from(level)])
        .collect()
}

fn candidate_levels(current: usize, total: usize) -> [usize; 2] {
    let lower = current.saturating_sub(1);
    let upper = (current + 1).min(total.saturating_sub(1));
    [lower, upper]
}

fn level_index_to_domain(level: i32, max_value: i32, levels: u16) -> i32 {
    let levels_m1 = i32::from(levels.saturating_sub(1)).max(1);
    let numerator = i64::from(level) * i64::from(max_value) + i64::from(levels_m1 / 2);
    (numerator / i64::from(levels_m1)) as i32
}

fn rgb_target_unit<S: Sample, L: PixelLayout>(
    buffer: &Buffer<'_, S, L>,
    width: usize,
    height: usize,
    pixel_count: usize,
) -> Result<Vec<[f32; 3]>> {
    let mut target = Vec::with_capacity(pixel_count);
    for y in 0..height {
        let row = buffer.try_row(y)?;
        for x in 0..width {
            let base = x * L::CHANNELS;
            target.push([
                row[base].to_unit_f32().clamp(0.0, 1.0),
                row[base + 1].to_unit_f32().clamp(0.0, 1.0),
                row[base + 2].to_unit_f32().clamp(0.0, 1.0),
            ]);
        }
    }
    Ok(target)
}

fn nearest_primary_index(rgb: [f32; 3]) -> usize {
    let mut best_index = 0_usize;
    let mut best_dist = f32::INFINITY;

    for candidate in 0..8_usize {
        let p = primary_unit(candidate);
        let dr = rgb[0] - p[0];
        let dg = rgb[1] - p[1];
        let db = rgb[2] - p[2];
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best_index = candidate;
        }
    }

    best_index
}

fn states_to_primary_unit(states: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Vec::with_capacity(states.len());
    let mut g = Vec::with_capacity(states.len());
    let mut b = Vec::with_capacity(states.len());

    for &state in states {
        let rgb = primary_unit(state);
        r.push(rgb[0]);
        g.push(rgb[1]);
        b.push(rgb[2]);
    }

    (r, g, b)
}

fn primary_unit(state: usize) -> [f32; 3] {
    [
        if state & 0b001 != 0 { 1.0 } else { 0.0 },
        if state & 0b010 != 0 { 1.0 } else { 0.0 },
        if state & 0b100 != 0 { 1.0 } else { 0.0 },
    ]
}

fn primary_domain(state: usize, max_value: i32) -> [i32; 3] {
    [
        if state & 0b001 != 0 { max_value } else { 0 },
        if state & 0b010 != 0 { max_value } else { 0 },
        if state & 0b100 != 0 { max_value } else { 0 },
    ]
}

fn primary_candidates(current: usize) -> [usize; 8] {
    [
        current,
        current ^ 0b001,
        current ^ 0b010,
        current ^ 0b100,
        current ^ 0b011,
        current ^ 0b101,
        current ^ 0b110,
        current ^ 0b111,
    ]
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

fn mbvq_region_states(region: MbvqRegion) -> [usize; 4] {
    match region {
        MbvqRegion::Cmyw => [0b110, 0b101, 0b011, 0b111],
        MbvqRegion::Mygc => [0b101, 0b011, 0b010, 0b110],
        MbvqRegion::Rgmy => [0b001, 0b010, 0b101, 0b011],
        MbvqRegion::Krgb => [0b000, 0b001, 0b010, 0b100],
        MbvqRegion::Rgbm => [0b001, 0b010, 0b100, 0b101],
        MbvqRegion::Cmgb => [0b110, 0b101, 0b010, 0b100],
    }
}

fn nearest_region_state(rgb: [f32; 3], region: MbvqRegion) -> usize {
    let mut best_state = mbvq_region_states(region)[0];
    let mut best_dist = f32::INFINITY;

    for state in mbvq_region_states(region) {
        let candidate = primary_unit(state);
        let dr = rgb[0] - candidate[0];
        let dg = rgb[1] - candidate[1];
        let db = rgb[2] - candidate[2];
        let dist = dr * dr + dg * dg + db * db;
        if dist < best_dist {
            best_dist = dist;
            best_state = state;
        }
    }

    best_state
}

fn write_binary_output<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    binary: &[u8],
    width: usize,
    height: usize,
    max_value: i32,
) -> Result<()> {
    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let domain = if binary[idx] == 0 { 0 } else { max_value };
            *value = domain_to_sample(domain, max_value);
        }
    }
    Ok(())
}

fn model_based_med_binary(target: &[f32], width: usize, height: usize) -> Vec<u8> {
    let pixel_count = width * height;
    let mut binary = target
        .iter()
        .copied()
        .map(|value| if value >= 0.5 { 1_u8 } else { 0_u8 })
        .collect::<Vec<u8>>();
    let mut error = vec![0.0_f32; pixel_count];

    for y in 0..height {
        let left_to_right = y % 2 == 0;

        if left_to_right {
            for x in 0..width {
                model_based_med_apply_pixel(
                    target,
                    (width, height),
                    x,
                    y,
                    true,
                    &mut binary,
                    &mut error,
                );
            }
        } else {
            for x in (0..width).rev() {
                model_based_med_apply_pixel(
                    target,
                    (width, height),
                    x,
                    y,
                    false,
                    &mut binary,
                    &mut error,
                );
            }
        }
    }

    binary
}

fn model_based_med_apply_pixel(
    target: &[f32],
    dims: (usize, usize),
    x: usize,
    y: usize,
    left_to_right: bool,
    binary: &mut [u8],
    error: &mut [f32],
) {
    let (width, height) = dims;
    let idx = y * width + x;
    let desired = (target[idx] + error[idx]).clamp(0.0, 1.0);
    let predicted_white = model_based_local_printer_intensity(binary, width, height, x, y, 1);
    let predicted_black = model_based_local_printer_intensity(binary, width, height, x, y, 0);
    let choose_white = if (desired - predicted_white).abs() < (desired - predicted_black).abs() {
        true
    } else if (desired - predicted_white).abs() > (desired - predicted_black).abs() {
        false
    } else {
        desired >= 0.5
    };

    let selected = if choose_white { 1_u8 } else { 0_u8 };
    let predicted = if choose_white {
        predicted_white
    } else {
        predicted_black
    };
    binary[idx] = selected;
    error[idx] = 0.0;
    let quant_error = (desired - predicted).clamp(-1.0, 1.0);
    model_based_diffuse_error(error, width, height, x, y, left_to_right, quant_error);
}

fn model_based_local_printer_intensity(
    binary: &[u8],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    center_value: u8,
) -> f32 {
    let mut intensity = 0.0_f32;

    for ky in 0..3 {
        for kx in 0..3 {
            let sx = x as isize + kx as isize - 1;
            let sy = y as isize + ky as isize - 1;
            let weight = MODEL_PRINTER_KERNEL[ky * 3 + kx];

            if sx < 0 || sy < 0 || sx as usize >= width || sy as usize >= height {
                intensity += weight;
                continue;
            }

            let source_value = if sx as usize == x && sy as usize == y {
                center_value
            } else {
                binary[sy as usize * width + sx as usize]
            };
            intensity += weight * f32::from(source_value);
        }
    }

    intensity.clamp(0.0, 1.0)
}

fn model_based_diffuse_error(
    error: &mut [f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    left_to_right: bool,
    quant_error: f32,
) {
    const WEIGHTS: [(isize, isize, f32); 4] = [
        (1, 0, 7.0 / 16.0),
        (-1, 1, 3.0 / 16.0),
        (0, 1, 5.0 / 16.0),
        (1, 1, 1.0 / 16.0),
    ];

    for &(dx, dy, weight) in &WEIGHTS {
        let mx = if left_to_right { dx } else { -dx };
        let nx = x as isize + mx;
        let ny = y as isize + dy;
        if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
            continue;
        }

        let idx = ny as usize * width + nx as usize;
        let next = error[idx] + quant_error * weight;
        error[idx] = next.clamp(-MODEL_MED_ERROR_LIMIT, MODEL_MED_ERROR_LIMIT);
    }
}

fn model_based_eye_filter() -> [f32; MODEL_EYE_SIZE * MODEL_EYE_SIZE] {
    let mut kernel = [0.0_f32; MODEL_EYE_SIZE * MODEL_EYE_SIZE];
    let mut sum = 0.0_f64;

    for ky in 0..MODEL_EYE_SIZE {
        for kx in 0..MODEL_EYE_SIZE {
            let dx = kx as isize - MODEL_EYE_RADIUS as isize;
            let dy = ky as isize - MODEL_EYE_RADIUS as isize;
            let dist2 = (dx * dx + dy * dy) as f64;
            let value = (-dist2 / (2.0 * MODEL_EYE_SIGMA * MODEL_EYE_SIGMA)).exp();
            kernel[ky * MODEL_EYE_SIZE + kx] = value as f32;
            sum += value;
        }
    }

    if sum > 0.0 {
        for value in &mut kernel {
            *value = (*value as f64 / sum) as f32;
        }
    }

    kernel
}

fn model_based_kernel(eye: &[f32; MODEL_EYE_SIZE * MODEL_EYE_SIZE]) -> Vec<f32> {
    let size = MODEL_EYE_SIZE + 2;
    let mut kernel = vec![0.0_f32; size * size];

    for py in 0..3 {
        for px in 0..3 {
            let printer_weight = MODEL_PRINTER_KERNEL[py * 3 + px];
            if printer_weight == 0.0 {
                continue;
            }

            for ey in 0..MODEL_EYE_SIZE {
                for ex in 0..MODEL_EYE_SIZE {
                    let out_x = px + ex;
                    let out_y = py + ey;
                    kernel[out_y * size + out_x] += printer_weight * eye[ey * MODEL_EYE_SIZE + ex];
                }
            }
        }
    }

    let sum = kernel.iter().copied().sum::<f32>();
    if sum != 0.0 {
        for value in &mut kernel {
            *value /= sum;
        }
    }

    kernel
}

fn model_based_filtered_error_map(
    target_unit: &[f32],
    binary: &[u8],
    width: usize,
    height: usize,
    kernel: &[f32],
    kernel_size: usize,
) -> Vec<f32> {
    let radius = kernel_size / 2;
    let mut filtered = vec![0.0_f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;

            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let sx = x as isize + kx as isize - radius as isize;
                    let sy = y as isize + ky as isize - radius as isize;
                    if sx < 0 || sy < 0 || sx as usize >= width || sy as usize >= height {
                        continue;
                    }

                    let sidx = sy as usize * width + sx as usize;
                    let halftone = f32::from(binary[sidx]);
                    acc += kernel[ky * kernel_size + kx] * (halftone - target_unit[sidx]);
                }
            }

            filtered[y * width + x] = acc;
        }
    }

    filtered
}

fn model_based_candidate_delta_energy(
    width: usize,
    height: usize,
    radius: usize,
    kernel: &[f32],
    kernel_size: usize,
    filtered_error: &[f32],
    points: &[(usize, usize, f32)],
) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let (min_x, max_x, min_y, max_y) = dbs_affected_bounds(width, height, radius, points);
    let mut delta_energy = 0.0_f64;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let mut delta = 0.0_f32;
            for &(px, py, dv) in points {
                let dx = x as isize - px as isize;
                let dy = y as isize - py as isize;
                if dx.unsigned_abs() > radius || dy.unsigned_abs() > radius {
                    continue;
                }

                let fx = (dx + radius as isize) as usize;
                let fy = (dy + radius as isize) as usize;
                delta += kernel[fy * kernel_size + fx] * dv;
            }

            if delta == 0.0 {
                continue;
            }

            let idx = y * width + x;
            let current = filtered_error[idx];
            delta_energy +=
                2.0 * f64::from(current) * f64::from(delta) + f64::from(delta) * f64::from(delta);
        }
    }

    delta_energy
}

fn model_based_apply_delta_filtered_error(
    width: usize,
    height: usize,
    radius: usize,
    kernel: &[f32],
    kernel_size: usize,
    filtered_error: &mut [f32],
    points: &[(usize, usize, f32)],
) {
    if points.is_empty() {
        return;
    }

    let (min_x, max_x, min_y, max_y) = dbs_affected_bounds(width, height, radius, points);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let mut delta = 0.0_f32;
            for &(px, py, dv) in points {
                let dx = x as isize - px as isize;
                let dy = y as isize - py as isize;
                if dx.unsigned_abs() > radius || dy.unsigned_abs() > radius {
                    continue;
                }

                let fx = (dx + radius as isize) as usize;
                let fy = (dy + radius as isize) as usize;
                delta += kernel[fy * kernel_size + fx] * dv;
            }

            if delta != 0.0 {
                let idx = y * width + x;
                filtered_error[idx] += delta;
            }
        }
    }
}

fn dbs_hvs_filter() -> [f32; DBS_HVS_SIZE * DBS_HVS_SIZE] {
    let mut kernel = [0.0_f32; DBS_HVS_SIZE * DBS_HVS_SIZE];
    let mut sum = 0.0_f64;

    for ky in 0..DBS_HVS_SIZE {
        for kx in 0..DBS_HVS_SIZE {
            let dx = kx as isize - DBS_HVS_RADIUS as isize;
            let dy = ky as isize - DBS_HVS_RADIUS as isize;
            let dist2 = (dx * dx + dy * dy) as f64;
            let value = (-dist2 / (2.0 * DBS_HVS_SIGMA * DBS_HVS_SIGMA)).exp();
            kernel[ky * DBS_HVS_SIZE + kx] = value as f32;
            sum += value;
        }
    }

    if sum > 0.0 {
        for value in &mut kernel {
            *value = (*value as f64 / sum) as f32;
        }
    }

    kernel
}

fn dbs_filtered_error_map(
    target_unit: &[f32],
    binary: &[u8],
    width: usize,
    height: usize,
    kernel: &[f32; DBS_HVS_SIZE * DBS_HVS_SIZE],
    radius: usize,
) -> Vec<f32> {
    let mut filtered = vec![0.0_f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;

            for ky in 0..DBS_HVS_SIZE {
                for kx in 0..DBS_HVS_SIZE {
                    let sx = x as isize + kx as isize - radius as isize;
                    let sy = y as isize + ky as isize - radius as isize;
                    if sx < 0 || sy < 0 || sx as usize >= width || sy as usize >= height {
                        continue;
                    }

                    let sidx = sy as usize * width + sx as usize;
                    let halftone = if binary[sidx] == 0 { 0.0_f32 } else { 1.0_f32 };
                    acc += kernel[ky * DBS_HVS_SIZE + kx] * (halftone - target_unit[sidx]);
                }
            }

            filtered[y * width + x] = acc;
        }
    }

    filtered
}

fn dbs_filtered_error_map_levels(
    target_unit: &[f32],
    output_unit: &[f32],
    width: usize,
    height: usize,
    kernel: &[f32; DBS_HVS_SIZE * DBS_HVS_SIZE],
    radius: usize,
) -> Vec<f32> {
    let mut filtered = vec![0.0_f32; width * height];

    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0_f32;

            for ky in 0..DBS_HVS_SIZE {
                for kx in 0..DBS_HVS_SIZE {
                    let sx = x as isize + kx as isize - radius as isize;
                    let sy = y as isize + ky as isize - radius as isize;
                    if sx < 0 || sy < 0 || sx as usize >= width || sy as usize >= height {
                        continue;
                    }

                    let sidx = sy as usize * width + sx as usize;
                    acc += kernel[ky * DBS_HVS_SIZE + kx] * (output_unit[sidx] - target_unit[sidx]);
                }
            }

            filtered[y * width + x] = acc;
        }
    }

    filtered
}

fn dbs_candidate_delta_energy(
    width: usize,
    height: usize,
    kernel: &[f32; DBS_HVS_SIZE * DBS_HVS_SIZE],
    radius: usize,
    filtered_error: &[f32],
    points: &[(usize, usize, f32)],
) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    let (min_x, max_x, min_y, max_y) = dbs_affected_bounds(width, height, radius, points);
    let mut delta_energy = 0.0_f64;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let mut delta = 0.0_f32;
            for &(px, py, dv) in points {
                let dx = x as isize - px as isize;
                let dy = y as isize - py as isize;
                if dx.unsigned_abs() > radius || dy.unsigned_abs() > radius {
                    continue;
                }

                let fx = (dx + radius as isize) as usize;
                let fy = (dy + radius as isize) as usize;
                delta += kernel[fy * DBS_HVS_SIZE + fx] * dv;
            }

            if delta == 0.0 {
                continue;
            }

            let idx = y * width + x;
            let current = filtered_error[idx];
            delta_energy +=
                2.0 * f64::from(current) * f64::from(delta) + f64::from(delta) * f64::from(delta);
        }
    }

    delta_energy
}

fn dbs_apply_delta_filtered_error(
    width: usize,
    height: usize,
    kernel: &[f32; DBS_HVS_SIZE * DBS_HVS_SIZE],
    radius: usize,
    filtered_error: &mut [f32],
    points: &[(usize, usize, f32)],
) {
    if points.is_empty() {
        return;
    }

    let (min_x, max_x, min_y, max_y) = dbs_affected_bounds(width, height, radius, points);

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let mut delta = 0.0_f32;
            for &(px, py, dv) in points {
                let dx = x as isize - px as isize;
                let dy = y as isize - py as isize;
                if dx.unsigned_abs() > radius || dy.unsigned_abs() > radius {
                    continue;
                }

                let fx = (dx + radius as isize) as usize;
                let fy = (dy + radius as isize) as usize;
                delta += kernel[fy * DBS_HVS_SIZE + fx] * dv;
            }

            if delta != 0.0 {
                let idx = y * width + x;
                filtered_error[idx] += delta;
            }
        }
    }
}

fn dbs_affected_bounds(
    width: usize,
    height: usize,
    radius: usize,
    points: &[(usize, usize, f32)],
) -> (usize, usize, usize, usize) {
    let mut min_x = width - 1;
    let mut max_x = 0_usize;
    let mut min_y = height - 1;
    let mut max_y = 0_usize;

    for &(px, py, _) in points {
        let x0 = px.saturating_sub(radius);
        let y0 = py.saturating_sub(radius);
        let x1 = px.saturating_add(radius).min(width - 1);
        let y1 = py.saturating_add(radius).min(height - 1);
        min_x = min_x.min(x0);
        max_x = max_x.max(x1);
        min_y = min_y.min(y0);
        max_y = max_y.max(y1);
    }

    (min_x, max_x, min_y, max_y)
}

fn toggle_delta_value(current: u8) -> f32 {
    if current == 0 {
        1.0
    } else {
        -1.0
    }
}

fn swap_primary_delta(primary: u8, secondary: u8) -> f32 {
    if primary == secondary {
        0.0
    } else if secondary == 1 {
        1.0
    } else {
        -1.0
    }
}

fn lbm_reference_state_for_pixel(
    state: &[f64],
    dims: (usize, usize),
    xy: (usize, usize),
    center: f64,
    max_value: f64,
    low_threshold: f64,
    reference: &mut [f64; 9],
) {
    let (width, height) = dims;
    let (x, y) = xy;
    reference.fill(0.0);
    let mut active_sum = 0.0_f64;

    if center < low_threshold {
        for d in 1..9 {
            let (dx, dy) = LBM_DIRECTIONS[d];
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                continue;
            }

            reference[d] = LBM_T_WEIGHTS[d];
            active_sum += reference[d];
        }

        if active_sum > 0.0 {
            let scale = center / active_sum;
            for value in reference.iter_mut().skip(1) {
                *value *= scale;
            }
        } else {
            reference[0] = center;
        }

        return;
    }

    if center > max_value {
        reference[0] = max_value;
        let extra = center - max_value;
        for d in 1..9 {
            let (dx, dy) = LBM_DIRECTIONS[d];
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                continue;
            }

            reference[d] = LBM_T_WEIGHTS[d];
            active_sum += reference[d];
        }

        if active_sum > 0.0 {
            let scale = extra / active_sum;
            for value in reference.iter_mut().skip(1) {
                *value *= scale;
            }
        } else {
            reference[0] += extra;
        }

        return;
    }

    reference[0] = LBM_T_WEIGHTS[0];
    active_sum += reference[0];

    for d in 1..9 {
        let (dx, dy) = LBM_DIRECTIONS[d];
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
            continue;
        }

        let neighbor_idx = ny as usize * width + nx as usize;
        let neighbor = state[neighbor_idx];
        if neighbor > center && neighbor < max_value {
            reference[d] = LBM_T_WEIGHTS[d];
            active_sum += reference[d];
        }
    }

    if active_sum <= 0.0 {
        reference[0] = center;
        return;
    }

    let scale = center / active_sum;
    for value in reference.iter_mut() {
        *value *= scale;
    }
}

fn electrostatic_forcefield(darkness: &[f64], width: usize, height: usize) -> Vec<[f64; 2]> {
    let mut field = vec![[0.0_f64; 2]; darkness.len()];

    for py in 0..height {
        for px in 0..width {
            let pidx = py * width + px;
            let mut fx = 0.0_f64;
            let mut fy = 0.0_f64;

            for gy in 0..height {
                for gx in 0..width {
                    let gidx = gy * width + gx;
                    if gidx == pidx {
                        continue;
                    }

                    let charge = darkness[gidx].clamp(0.0, 1.0);
                    if charge == 0.0 {
                        continue;
                    }

                    let dx = gx as f64 - px as f64;
                    let dy = gy as f64 - py as f64;
                    let dist2 = dx * dx + dy * dy + ELECTRO_FORCE_EPSILON;
                    let inv_r3 = charge / (dist2 * dist2.sqrt());
                    fx += dx * inv_r3;
                    fy += dy * inv_r3;
                }
            }

            field[pidx] = [
                fx * ELECTRO_ATTRACTION_WEIGHT,
                fy * ELECTRO_ATTRACTION_WEIGHT,
            ];
        }
    }

    field
}

fn electrostatic_initial_positions(
    darkness: &[f64],
    width: usize,
    height: usize,
    particle_count: usize,
    rng: &mut u64,
) -> Vec<[f64; 2]> {
    let pixel_count = width * height;
    let mut occupied = vec![false; pixel_count];
    let mut positions = Vec::with_capacity(particle_count);
    let mut attempts = 0_usize;
    let max_attempts = pixel_count.saturating_mul(64).max(1);

    while positions.len() < particle_count && attempts < max_attempts {
        let idx = electrostatic_rand_index(rng, pixel_count);
        attempts += 1;
        if occupied[idx] {
            continue;
        }

        let prob = darkness[idx].clamp(0.0, 1.0);
        if electrostatic_rand_unit(rng) <= prob {
            occupied[idx] = true;
            positions.push([(idx % width) as f64, (idx / width) as f64]);
        }
    }

    if positions.len() < particle_count {
        let mut ranking = darkness.iter().enumerate().collect::<Vec<(usize, &f64)>>();
        ranking.sort_by(|(ia, va), (ib, vb)| vb.total_cmp(va).then_with(|| ia.cmp(ib)));

        for (idx, _) in ranking {
            if positions.len() >= particle_count {
                break;
            }
            if occupied[idx] {
                continue;
            }
            occupied[idx] = true;
            positions.push([(idx % width) as f64, (idx / width) as f64]);
        }
    }

    positions
}

fn electrostatic_compute_forces(
    positions: &[[f64; 2]],
    field: &[[f64; 2]],
    width: usize,
    height: usize,
    forces: &mut [[f64; 2]],
) {
    for i in 0..positions.len() {
        let pos = positions[i];
        let mut force = electrostatic_bilinear(field, width, height, pos);

        for (j, other) in positions.iter().enumerate() {
            if i == j {
                continue;
            }

            let dx = pos[0] - other[0];
            let dy = pos[1] - other[1];
            let dist2 = dx * dx + dy * dy + ELECTRO_FORCE_EPSILON;
            let inv_r3 = ELECTRO_REPULSION_WEIGHT / (dist2 * dist2.sqrt());
            force[0] += dx * inv_r3;
            force[1] += dy * inv_r3;
        }

        forces[i] = force;
    }
}

fn electrostatic_bilinear(
    field: &[[f64; 2]],
    width: usize,
    height: usize,
    position: [f64; 2],
) -> [f64; 2] {
    let x = position[0].clamp(0.0, (width - 1) as f64);
    let y = position[1].clamp(0.0, (height - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f64;
    let ty = y - y0 as f64;

    let f00 = field[y0 * width + x0];
    let f10 = field[y0 * width + x1];
    let f01 = field[y1 * width + x0];
    let f11 = field[y1 * width + x1];

    let fx0 = f00[0] * (1.0 - tx) + f10[0] * tx;
    let fy0 = f00[1] * (1.0 - tx) + f10[1] * tx;
    let fx1 = f01[0] * (1.0 - tx) + f11[0] * tx;
    let fy1 = f01[1] * (1.0 - tx) + f11[1] * tx;

    [fx0 * (1.0 - ty) + fx1 * ty, fy0 * (1.0 - ty) + fy1 * ty]
}

fn electrostatic_project_to_grid(positions: &[[f64; 2]], width: usize, height: usize) -> Vec<bool> {
    let pixel_count = width * height;
    let mut occupied = vec![false; pixel_count];

    let mut ordering = positions
        .iter()
        .enumerate()
        .collect::<Vec<(usize, &[f64; 2])>>();
    ordering.sort_by(|(ia, pa), (ib, pb)| {
        let da = pa[0] + pa[1] * width as f64;
        let db = pb[0] + pb[1] * width as f64;
        da.total_cmp(&db).then_with(|| ia.cmp(ib))
    });

    for (_, position) in ordering {
        let idx = electrostatic_nearest_free_pixel(position, &occupied, width, height);
        occupied[idx] = true;
    }

    occupied
}

fn electrostatic_nearest_free_pixel(
    position: &[f64; 2],
    occupied: &[bool],
    width: usize,
    height: usize,
) -> usize {
    let center_x = position[0].round().clamp(0.0, (width - 1) as f64) as usize;
    let center_y = position[1].round().clamp(0.0, (height - 1) as f64) as usize;
    let max_radius = width.max(height);

    for radius in 0..=max_radius {
        let mut best_idx = None;
        let mut best_dist2 = f64::INFINITY;

        let x_min = center_x.saturating_sub(radius);
        let y_min = center_y.saturating_sub(radius);
        let x_max = center_x.saturating_add(radius).min(width - 1);
        let y_max = center_y.saturating_add(radius).min(height - 1);

        for y in y_min..=y_max {
            for x in x_min..=x_max {
                if x.abs_diff(center_x).max(y.abs_diff(center_y)) != radius {
                    continue;
                }

                let idx = y * width + x;
                if occupied[idx] {
                    continue;
                }

                let dx = position[0] - x as f64;
                let dy = position[1] - y as f64;
                let dist2 = dx * dx + dy * dy;
                let tie_break = match best_idx {
                    Some(best) => idx < best,
                    None => true,
                };
                if dist2 < best_dist2 || (dist2 == best_dist2 && tie_break) {
                    best_dist2 = dist2;
                    best_idx = Some(idx);
                }
            }
        }

        if let Some(idx) = best_idx {
            return idx;
        }
    }

    0
}

fn electrostatic_rand_index(seed: &mut u64, len: usize) -> usize {
    if len == 0 {
        0
    } else {
        (electrostatic_next_u64(seed) as usize) % len
    }
}

fn electrostatic_rand_unit(seed: &mut u64) -> f64 {
    let value = electrostatic_next_u64(seed);
    (value as f64) / (u64::MAX as f64)
}

fn electrostatic_next_u64(seed: &mut u64) -> u64 {
    let mut x = *seed;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *seed = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateMove {
    None,
    Toggle(usize),
    Swap(usize, usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MbvqRegion {
    Cmyw,
    Mygc,
    Rgmy,
    Krgb,
    Rgbm,
    Cmgb,
}

fn ensure_grayscale_integer_format<S: Sample, L: PixelLayout>() -> Result<()> {
    if L::CHANNELS == 1 && !L::HAS_ALPHA && !S::IS_FLOAT {
        Ok(())
    } else {
        Err(Error::UnsupportedFormat(
            "research algorithms support Gray8 and Gray16 only",
        ))
    }
}

fn ensure_rgb_integer_format<S: Sample, L: PixelLayout>() -> Result<()> {
    if (L::CHANNELS == 3 || L::CHANNELS == 4) && L::COLOR_CHANNELS == 3 && !S::IS_FLOAT {
        Ok(())
    } else {
        Err(Error::UnsupportedFormat(
            "research algorithms support Rgb8/Rgba8 and Rgb16/Rgba16 only",
        ))
    }
}

fn ensure_rgb_integer_opaque_format<S: Sample, L: PixelLayout>() -> Result<()> {
    if L::CHANNELS == 3 && L::COLOR_CHANNELS == 3 && !L::HAS_ALPHA && !S::IS_FLOAT {
        Ok(())
    } else {
        Err(Error::UnsupportedFormat(
            "research algorithms support Rgb8 and Rgb16 only",
        ))
    }
}

fn integer_sample_max<S: Sample>() -> Result<i32> {
    if S::IS_FLOAT {
        return Err(Error::UnsupportedFormat(
            "research algorithms support integer sample types only",
        ));
    }

    match size_of::<S>() {
        1 => Ok(255),
        2 => Ok(65_535),
        _ => Err(Error::UnsupportedFormat(
            "unsupported integer sample width for research algorithms",
        )),
    }
}

fn domain_to_sample<S: Sample>(value: i32, max_value: i32) -> S {
    S::from_unit_f32(value.clamp(0, max_value) as f32 / max_value as f32)
}
