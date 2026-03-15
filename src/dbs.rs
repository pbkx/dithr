use crate::{Buffer, PixelFormat};

const DBS_KERNEL: [[u32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];

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
