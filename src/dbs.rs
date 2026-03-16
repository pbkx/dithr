use crate::{math::fixed::mul_div_i32, Buffer, PixelFormat};

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

pub fn direct_binary_search_in_place(buffer: &mut Buffer<'_>, max_iters: usize) {
    buffer
        .validate()
        .expect("buffer must be valid for direct binary search");
    assert_eq!(
        buffer.format,
        PixelFormat::Gray8,
        "direct binary search v1 supports Gray8 only"
    );

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width.checked_mul(height).expect("image size overflow");

    let mut target = Vec::with_capacity(pixel_count);
    let mut binary = Vec::with_capacity(pixel_count);

    for y in 0..height {
        let row = buffer.row(y);
        for &value in row.iter().take(width) {
            target.push(value);
            binary.push(if value >= 128 { 255 } else { 0 });
        }
    }

    let mut current_objective = dbs_objective(&target, &binary, width, height);

    for _ in 0..max_iters {
        let mut improved = false;

        for y in 0..height {
            for x in 0..width {
                let idx = y
                    .checked_mul(width)
                    .and_then(|base| base.checked_add(x))
                    .expect("pixel index overflow");
                let mut best = current_objective;
                let mut best_move = CandidateMove::None;

                let original = binary[idx];
                binary[idx] = 255_u8.wrapping_sub(original);
                let flip_objective = dbs_objective(&target, &binary, width, height);
                if flip_objective < best {
                    best = flip_objective;
                    best_move = CandidateMove::Flip(idx);
                }
                binary[idx] = original;

                if x + 1 < width {
                    let right = idx + 1;
                    if binary[idx] != binary[right] {
                        binary.swap(idx, right);
                        let swap_right_objective = dbs_objective(&target, &binary, width, height);
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
                        let swap_down_objective = dbs_objective(&target, &binary, width, height);
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
                        binary[i] = 255_u8.wrapping_sub(binary[i]);
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
        let start = y.checked_mul(width).expect("binary row start overflow");
        let end = start.checked_add(width).expect("binary row end overflow");
        let row = buffer.row_mut(y);
        row[..width].copy_from_slice(&binary[start..end]);
    }
}

pub fn lattice_boltzmann_in_place(buffer: &mut Buffer<'_>, max_steps: usize) {
    buffer
        .validate()
        .expect("buffer must be valid for lattice-boltzmann dithering");
    assert_eq!(
        buffer.format,
        PixelFormat::Gray8,
        "lattice-boltzmann v1 supports Gray8 only"
    );

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width.checked_mul(height).expect("image size overflow");
    let mut target = Vec::with_capacity(pixel_count);

    for y in 0..height {
        let row = buffer.row(y);
        for &value in row.iter().take(width) {
            target.push(mul_div_i32(i32::from(value), LBM_SCALE, 255));
        }
    }

    let mut distributions = vec![[0_i32; 9]; pixel_count];
    let mut post_collision = vec![[0_i32; 9]; pixel_count];
    let mut streamed = vec![[0_i32; 9]; pixel_count];

    for i in 0..pixel_count {
        for d in 0..9 {
            distributions[i][d] = mul_div_i32(target[i], LBM_WEIGHTS[d], LBM_SCALE);
        }
    }

    for _ in 0..max_steps {
        for idx in 0..pixel_count {
            let rho = distributions[idx].iter().sum::<i32>();
            let rho_for_eq = rho + mul_div_i32(target[idx] - rho, LBM_FORCING_NUM, LBM_FORCING_DEN);

            for d in 0..9 {
                let feq = mul_div_i32(rho_for_eq, LBM_WEIGHTS[d], LBM_SCALE);
                let delta = feq - distributions[idx][d];
                post_collision[idx][d] =
                    distributions[idx][d] + mul_div_i32(delta, LBM_OMEGA_NUM, LBM_OMEGA_DEN);
            }
        }

        streamed.fill([0_i32; 9]);

        for y in 0..height {
            for x in 0..width {
                let idx = y
                    .checked_mul(width)
                    .and_then(|base| base.checked_add(x))
                    .expect("stream index overflow");

                for d in 0..9 {
                    let (dx, dy) = LBM_DIRECTIONS[d];
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;

                    if nx >= 0 && ny >= 0 && (nx as usize) < width && (ny as usize) < height {
                        let nidx = (ny as usize)
                            .checked_mul(width)
                            .and_then(|base| base.checked_add(nx as usize))
                            .expect("stream target overflow");
                        streamed[nidx][d] += post_collision[idx][d];
                    } else {
                        let opp = LBM_OPPOSITE[d];
                        streamed[idx][opp] += post_collision[idx][d];
                    }
                }
            }
        }

        std::mem::swap(&mut distributions, &mut streamed);
    }

    for y in 0..height {
        let row = buffer.row_mut(y);
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y
                .checked_mul(width)
                .and_then(|base| base.checked_add(x))
                .expect("output index overflow");
            let rho = distributions[idx].iter().sum::<i32>();
            *value = if rho >= LBM_HALF_SCALE { 255 } else { 0 };
        }
    }
}

pub fn electrostatic_halftoning_in_place(buffer: &mut Buffer<'_>, max_steps: usize) {
    buffer
        .validate()
        .expect("buffer must be valid for electrostatic halftoning");
    assert_eq!(
        buffer.format,
        PixelFormat::Gray8,
        "electrostatic halftoning v1 supports Gray8 only"
    );

    let width = buffer.width;
    let height = buffer.height;
    let pixel_count = width.checked_mul(height).expect("image size overflow");
    let mut darkness = Vec::with_capacity(pixel_count);

    for y in 0..height {
        let row = buffer.row(y);
        for &value in row.iter().take(width) {
            darkness.push(255_u8.wrapping_sub(value));
        }
    }

    let darkness_sum: usize = darkness.iter().map(|&value| usize::from(value)).sum();
    let mut particle_count = (darkness_sum + 127) / 255;
    if particle_count > pixel_count {
        particle_count = pixel_count;
    }

    let (mut occupied, mut particles) = initial_particles(&darkness, particle_count);

    for _ in 0..max_steps {
        let mut moved = false;

        for particle_index in 0..particles.len() {
            let current = particles[particle_index];
            let mut best = current;
            let mut best_utility =
                electrostatic_utility(current, current, &particles, &darkness, width);

            let cx = current % width;
            let cy = current / width;

            for (dx, dy) in ELECTRO_NEIGHBORS {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                    continue;
                }

                let candidate = (ny as usize)
                    .checked_mul(width)
                    .and_then(|base| base.checked_add(nx as usize))
                    .expect("candidate index overflow");
                if candidate != current && occupied[candidate] {
                    continue;
                }

                let utility =
                    electrostatic_utility(candidate, current, &particles, &darkness, width);
                if utility > best_utility {
                    best_utility = utility;
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
        let row = buffer.row_mut(y);
        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y
                .checked_mul(width)
                .and_then(|base| base.checked_add(x))
                .expect("output index overflow");
            *value = if occupied[idx] { 0 } else { 255 };
        }
    }
}

fn dbs_objective(target: &[u8], binary: &[u8], width: usize, height: usize) -> u64 {
    let mut total = 0_u64;

    for y in 0..height {
        for x in 0..width {
            let mut weighted_sum = 0_u32;
            let mut weight_total = 0_u32;

            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let ky = ny + 1 - y;
                    let kx = nx + 1 - x;
                    let weight = DBS_KERNEL[ky][kx];
                    let idx = ny
                        .checked_mul(width)
                        .and_then(|base| base.checked_add(nx))
                        .expect("neighbor index overflow");

                    weighted_sum = weighted_sum
                        .checked_add(
                            u32::from(binary[idx])
                                .checked_mul(weight)
                                .expect("weighted sum overflow"),
                        )
                        .expect("weighted accumulation overflow");
                    weight_total = weight_total
                        .checked_add(weight)
                        .expect("weight accumulation overflow");
                }
            }

            let filtered = ((weighted_sum + (weight_total / 2)) / weight_total) as i32;
            let idx = y
                .checked_mul(width)
                .and_then(|base| base.checked_add(x))
                .expect("target index overflow");
            let diff = i32::from(target[idx]) - filtered;
            let sq = i64::from(diff) * i64::from(diff);
            total = total.checked_add(sq as u64).expect("objective overflow");
        }
    }

    total
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

fn electrostatic_utility(
    candidate: usize,
    self_position: usize,
    particles: &[usize],
    darkness: &[u8],
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
        let dist2 = dx
            .checked_mul(dx)
            .and_then(|v| v.checked_add(dy * dy))
            .expect("distance overflow");
        if dist2 == 0 {
            continue;
        }
        repulsion += ELECTRO_REPEL_SCALE / dist2;
    }

    let attraction = i64::from(darkness[candidate]) * ELECTRO_ATTRACT_WEIGHT;
    attraction - (repulsion * ELECTRO_REPEL_WEIGHT)
}
