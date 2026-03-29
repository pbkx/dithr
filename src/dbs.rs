use crate::{
    core::{PixelLayout, Sample},
    math::fixed::mul_div_i32,
    Buffer, Error, Result,
};
use std::mem::size_of;

const DBS_KERNEL: [[u32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
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
            binary.push(if unit >= 0.5 { max_value } else { 0 });
        }
    }

    let mut current_objective =
        dbs_objective_unit(&target_unit, &binary, width, height, max_value)?;

    for _ in 0..max_iters {
        let mut improved = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let mut best = current_objective;
                let mut best_move = CandidateMove::None;

                if let Some(flip_objective) = candidate_flip_improves(
                    &target_unit,
                    &mut binary,
                    width,
                    height,
                    max_value,
                    idx,
                    current_objective,
                )? {
                    best = flip_objective;
                    best_move = CandidateMove::Flip(idx);
                }

                if x + 1 < width {
                    let right = idx + 1;
                    if binary[idx] != binary[right] {
                        binary.swap(idx, right);
                        let swap_right_objective =
                            dbs_objective_unit(&target_unit, &binary, width, height, max_value)?;
                        if swap_right_objective < best {
                            best = swap_right_objective;
                            best_move = CandidateMove::Swap(idx, right);
                        }
                        binary.swap(idx, right);
                    }
                }

                if y + 1 < height {
                    let below = idx + width;
                    if binary[idx] != binary[below] {
                        binary.swap(idx, below);
                        let swap_down_objective =
                            dbs_objective_unit(&target_unit, &binary, width, height, max_value)?;
                        if swap_down_objective < best {
                            best = swap_down_objective;
                            best_move = CandidateMove::Swap(idx, below);
                        }
                        binary.swap(idx, below);
                    }
                }

                match best_move {
                    CandidateMove::None => {}
                    CandidateMove::Flip(i) => {
                        binary[i] = max_value - binary[i];
                        current_objective = best;
                        improved = true;
                    }
                    CandidateMove::Swap(i, j) => {
                        binary.swap(i, j);
                        current_objective = best;
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
            *value = domain_to_sample(binary[start + x], max_value);
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

fn dbs_objective_unit(
    target_unit: &[f32],
    binary: &[i32],
    width: usize,
    height: usize,
    max_value: i32,
) -> Result<u64> {
    let mut total = 0_u64;

    for y in 0..height {
        for x in 0..width {
            let mut weighted_sum = 0_i64;
            let mut weight_total = 0_i64;

            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let ky = ny + 1 - y;
                    let kx = nx + 1 - x;
                    let weight = i64::from(DBS_KERNEL[ky][kx]);
                    let idx = ny * width + nx;

                    weighted_sum = weighted_sum
                        .checked_add(i64::from(binary[idx]) * weight)
                        .ok_or(Error::InvalidArgument("objective weighted sum overflow"))?;
                    weight_total = weight_total
                        .checked_add(weight)
                        .ok_or(Error::InvalidArgument("objective weight overflow"))?;
                }
            }

            let filtered = ((weighted_sum + (weight_total / 2)) / weight_total) as i32;
            let idx = y * width + x;
            let target = unit_to_domain(target_unit[idx], max_value);
            let diff = target - filtered;
            let sq = (i64::from(diff) * i64::from(diff)) as u64;
            total = total
                .checked_add(sq)
                .ok_or(Error::InvalidArgument("objective overflow"))?;
        }
    }

    Ok(total)
}

fn candidate_flip_improves(
    target_unit: &[f32],
    binary: &mut [i32],
    width: usize,
    height: usize,
    max_value: i32,
    idx: usize,
    current_objective: u64,
) -> Result<Option<u64>> {
    let original = binary[idx];
    binary[idx] = max_value - original;
    let flip_objective = dbs_objective_unit(target_unit, binary, width, height, max_value)?;
    binary[idx] = original;

    if flip_objective < current_objective {
        Ok(Some(flip_objective))
    } else {
        Ok(None)
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
    Flip(usize),
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
