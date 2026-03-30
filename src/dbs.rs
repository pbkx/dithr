use crate::{
    core::{PixelLayout, Sample},
    math::fixed::mul_div_i32,
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
const LBM_SCALE: i32 = 16_384;
const LBM_HALF_SCALE: i32 = LBM_SCALE / 2;
const LBM_WEIGHTS: [i32; 9] = [7_284, 1_820, 1_820, 1_820, 1_820, 455, 455, 455, 455];
const LBM_DIRECTIONS: [(isize, isize); 9] = [
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
];
const LBM_OPPOSITE: [usize; 9] = [0, 3, 4, 1, 2, 7, 8, 5, 6];
const LBM_OMEGA_NUM: i32 = 6;
const LBM_OMEGA_DEN: i32 = 5;
const LBM_FORCING_NUM: i32 = 1;
const LBM_FORCING_DEN: i32 = 4;
const ELECTRO_ATTRACT_WEIGHT: i64 = 16;
const ELECTRO_REPEL_WEIGHT: i64 = 12;
const ELECTRO_REPEL_SCALE: i64 = 256;
const ELECTRO_NEIGHBORS: [(isize, isize); 9] = [
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (1, 1),
];

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

    let mut target_unit = Vec::with_capacity(pixel_count);
    for y in 0..height {
        let row = buffer.try_row(y)?;
        for &value in row.iter().take(width) {
            target_unit.push(value.to_unit_f32().clamp(0.0, 1.0));
        }
    }

    let mut distributions = vec![[0_i32; 9]; pixel_count];
    let mut post_collision = vec![[0_i32; 9]; pixel_count];
    let mut streamed = vec![[0_i32; 9]; pixel_count];

    for i in 0..pixel_count {
        let target_scaled = unit_to_lbm(target_unit[i]);
        for d in 0..9 {
            distributions[i][d] = mul_div_i32(target_scaled, LBM_WEIGHTS[d], LBM_SCALE)?;
        }
    }

    for _ in 0..max_steps {
        lattice_state_step_unit(
            &mut distributions,
            &mut post_collision,
            &mut streamed,
            &target_unit,
            width,
            height,
        )?;
    }

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y * width + x;
            let rho = distributions[idx].iter().sum::<i32>();
            let domain = if rho >= LBM_HALF_SCALE { max_value } else { 0 };
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
    let mut darkness_unit = Vec::with_capacity(pixel_count);

    for y in 0..height {
        let row = buffer.try_row(y)?;
        for &value in row.iter().take(width) {
            darkness_unit.push(1.0 - value.to_unit_f32().clamp(0.0, 1.0));
        }
    }

    let darkness_rank = darkness_unit
        .iter()
        .map(|&value| unit_to_domain(value, 255).clamp(0, 255) as u8)
        .collect::<Vec<_>>();
    let darkness_sum: usize = darkness_rank.iter().map(|&value| usize::from(value)).sum();
    let mut particle_count = (darkness_sum + 127) / 255;
    if particle_count > pixel_count {
        particle_count = pixel_count;
    }

    let (mut occupied, mut particles) = initial_particles(&darkness_rank, particle_count);

    for _ in 0..max_steps {
        let mut moved = false;

        for particle_index in 0..particles.len() {
            let current = particles[particle_index];
            let mut best = current;
            let mut best_energy =
                electrostatic_energy_unit(current, current, &particles, &darkness_unit, width);

            let cx = current % width;
            let cy = current / width;

            for (dx, dy) in ELECTRO_NEIGHBORS {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                    continue;
                }

                let candidate = ny as usize * width + nx as usize;
                if candidate != current && occupied[candidate] {
                    continue;
                }

                let energy = electrostatic_energy_unit(
                    candidate,
                    current,
                    &particles,
                    &darkness_unit,
                    width,
                );
                if energy > best_energy {
                    best_energy = energy;
                    best = candidate;
                }
            }

            if best != current {
                occupied[current] = false;
                occupied[best] = true;
                particles[particle_index] = best;
                moved = true;
            }
        }

        if !moved {
            break;
        }
    }

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

fn lattice_state_step_unit(
    distributions: &mut Vec<[i32; 9]>,
    post_collision: &mut [[i32; 9]],
    streamed: &mut Vec<[i32; 9]>,
    target_unit: &[f32],
    width: usize,
    height: usize,
) -> Result<()> {
    for idx in 0..distributions.len() {
        let rho = distributions[idx].iter().sum::<i32>();
        let target_scaled = unit_to_lbm(target_unit[idx]);
        let rho_for_eq = rho + mul_div_i32(target_scaled - rho, LBM_FORCING_NUM, LBM_FORCING_DEN)?;

        for d in 0..9 {
            let feq = mul_div_i32(rho_for_eq, LBM_WEIGHTS[d], LBM_SCALE)?;
            let delta = feq - distributions[idx][d];
            post_collision[idx][d] =
                distributions[idx][d] + mul_div_i32(delta, LBM_OMEGA_NUM, LBM_OMEGA_DEN)?;
        }
    }

    streamed.fill([0_i32; 9]);

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;

            for d in 0..9 {
                let (dx, dy) = LBM_DIRECTIONS[d];
                let nx = x as isize + dx;
                let ny = y as isize + dy;

                if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                    let nidx = ny as usize * width + nx as usize;
                    streamed[nidx][d] += post_collision[idx][d];
                } else {
                    let opp = LBM_OPPOSITE[d];
                    streamed[idx][opp] += post_collision[idx][d];
                }
            }
        }
    }

    std::mem::swap(distributions, streamed);
    Ok(())
}

fn electrostatic_energy_unit(
    candidate: usize,
    self_position: usize,
    particles: &[usize],
    darkness_unit: &[f32],
    width: usize,
) -> i64 {
    let cx = (candidate % width) as i64;
    let cy = (candidate / width) as i64;
    let mut repulsion = 0_i64;

    for &other in particles {
        if other == self_position {
            continue;
        }

        let ox = (other % width) as i64;
        let oy = (other / width) as i64;
        let dx = cx - ox;
        let dy = cy - oy;
        let dist2 = i128::from(dx) * i128::from(dx) + i128::from(dy) * i128::from(dy);
        if dist2 == 0 {
            continue;
        }
        repulsion += (i128::from(ELECTRO_REPEL_SCALE) / dist2) as i64;
    }

    let attraction = i64::from(unit_to_domain(
        darkness_unit[candidate].clamp(0.0, 1.0),
        255,
    )) * ELECTRO_ATTRACT_WEIGHT;
    attraction - (repulsion * ELECTRO_REPEL_WEIGHT)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandidateMove {
    None,
    Toggle(usize),
    Swap(usize, usize),
}

fn initial_particles(darkness: &[u8], particle_count: usize) -> (Vec<bool>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..darkness.len()).collect();
    indices.sort_by(|a, b| darkness[*b].cmp(&darkness[*a]).then(a.cmp(b)));

    let mut occupied = vec![false; darkness.len()];
    let mut particles = indices.into_iter().take(particle_count).collect::<Vec<_>>();
    particles.sort_unstable();

    for &idx in &particles {
        occupied[idx] = true;
    }

    (occupied, particles)
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

fn unit_to_domain(unit: f32, max_value: i32) -> i32 {
    (unit.clamp(0.0, 1.0) * max_value as f32).round() as i32
}

fn domain_to_sample<S: Sample>(value: i32, max_value: i32) -> S {
    S::from_unit_f32(value.clamp(0, max_value) as f32 / max_value as f32)
}

fn unit_to_lbm(unit: f32) -> i32 {
    (unit.clamp(0.0, 1.0) * LBM_SCALE as f32).round() as i32
}
