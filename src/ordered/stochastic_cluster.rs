use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{PixelLayout, Sample},
    Buffer, QuantizeMode, Result,
};
use core::cmp::Ordering;
use std::sync::OnceLock;

const STOCHASTIC_CLUSTER_SIDE: usize = 64;
const STOCHASTIC_CLUSTER_LEN: usize = STOCHASTIC_CLUSTER_SIDE * STOCHASTIC_CLUSTER_SIDE;
const DOT_RADIUS: i32 = 4;

static STOCHASTIC_CLUSTER_64X64_FLAT: OnceLock<[u16; STOCHASTIC_CLUSTER_LEN]> = OnceLock::new();

pub fn stochastic_clustered_dot_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(
        buffer,
        mode,
        stochastic_cluster_64x64_flat(),
        STOCHASTIC_CLUSTER_SIDE,
        STOCHASTIC_CLUSTER_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn stochastic_cluster_64x64_flat() -> &'static [u16; STOCHASTIC_CLUSTER_LEN] {
    STOCHASTIC_CLUSTER_64X64_FLAT.get_or_init(generate_stochastic_cluster_64x64_flat)
}

fn generate_stochastic_cluster_64x64_flat() -> [u16; STOCHASTIC_CLUSTER_LEN] {
    let centers = place_stochastic_centers();
    let preliminary = build_preliminary_thresholds(&centers);
    renumber_thresholds(&preliminary)
}

fn place_stochastic_centers() -> Vec<usize> {
    let mut occupied = [false; STOCHASTIC_CLUSTER_LEN];
    let mut order = [0_usize; STOCHASTIC_CLUSTER_LEN];
    for (idx, slot) in order.iter_mut().enumerate() {
        *slot = idx;
    }
    order.sort_by_key(|&idx| visit_hash(idx));

    let mut centers = Vec::with_capacity(STOCHASTIC_CLUSTER_LEN / 24);
    for idx in order {
        if occupied[idx] {
            continue;
        }
        centers.push(idx);
        mark_disk_occupied(idx, &mut occupied);
    }

    if centers.is_empty() {
        centers.push(0);
    }

    centers
}

fn mark_disk_occupied(center_idx: usize, occupied: &mut [bool; STOCHASTIC_CLUSTER_LEN]) {
    let cx = (center_idx % STOCHASTIC_CLUSTER_SIDE) as i32;
    let cy = (center_idx / STOCHASTIC_CLUSTER_SIDE) as i32;
    let radius_sq = DOT_RADIUS * DOT_RADIUS;

    for dy in -DOT_RADIUS..=DOT_RADIUS {
        for dx in -DOT_RADIUS..=DOT_RADIUS {
            if dx * dx + dy * dy > radius_sq {
                continue;
            }

            let x = wrap_axis(cx + dx);
            let y = wrap_axis(cy + dy);
            occupied[y * STOCHASTIC_CLUSTER_SIDE + x] = true;
        }
    }
}

fn wrap_axis(value: i32) -> usize {
    let side = STOCHASTIC_CLUSTER_SIDE as i32;
    let wrapped = value.rem_euclid(side);
    wrapped as usize
}

fn build_preliminary_thresholds(centers: &[usize]) -> [f32; STOCHASTIC_CLUSTER_LEN] {
    let mut out = [0.0_f32; STOCHASTIC_CLUSTER_LEN];

    for (idx, slot) in out.iter_mut().enumerate() {
        let mut nearest = u32::MAX;
        let mut second = u32::MAX;
        let mut nearest_center = 0_usize;

        for &center in centers {
            let dist = toroidal_distance_sq(idx, center);
            if dist < nearest {
                second = nearest;
                nearest = dist;
                nearest_center = center;
            } else if dist < second {
                second = dist;
            }
        }

        if second == u32::MAX {
            second = nearest.max(1);
        }

        let d1 = (nearest as f32).sqrt();
        let d2 = (second as f32).sqrt();
        let base = if d1 + d2 > 0.0 { d1 / (d1 + d2) } else { 0.0 };
        let jitter = tiny_jitter(idx, nearest_center);
        *slot = (base + jitter).clamp(0.0, 1.0);
    }

    out
}

fn toroidal_distance_sq(a_idx: usize, b_idx: usize) -> u32 {
    let ax = a_idx % STOCHASTIC_CLUSTER_SIDE;
    let ay = a_idx / STOCHASTIC_CLUSTER_SIDE;
    let bx = b_idx % STOCHASTIC_CLUSTER_SIDE;
    let by = b_idx / STOCHASTIC_CLUSTER_SIDE;

    let dx = toroidal_axis_delta(ax, bx);
    let dy = toroidal_axis_delta(ay, by);

    (dx * dx + dy * dy) as u32
}

fn toroidal_axis_delta(a: usize, b: usize) -> usize {
    let delta = a.abs_diff(b);
    delta.min(STOCHASTIC_CLUSTER_SIDE - delta)
}

fn tiny_jitter(index: usize, center: usize) -> f32 {
    let seed = mix_u64(
        (index as u64)
            .wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
            .wrapping_add((center as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64)),
    );
    (seed & 0xffff_u64) as f32 / (65535.0_f32 * 65535.0_f32)
}

fn renumber_thresholds(
    preliminary: &[f32; STOCHASTIC_CLUSTER_LEN],
) -> [u16; STOCHASTIC_CLUSTER_LEN] {
    let mut order = [0_usize; STOCHASTIC_CLUSTER_LEN];
    for (idx, slot) in order.iter_mut().enumerate() {
        *slot = idx;
    }
    order.sort_by(|&lhs, &rhs| compare_cells(preliminary, lhs, rhs));

    let mut out = [0_u16; STOCHASTIC_CLUSTER_LEN];
    for (rank, &idx) in order.iter().enumerate() {
        out[idx] = rank as u16;
    }

    out
}

fn compare_cells(preliminary: &[f32; STOCHASTIC_CLUSTER_LEN], lhs: usize, rhs: usize) -> Ordering {
    preliminary[lhs]
        .partial_cmp(&preliminary[rhs])
        .unwrap_or(Ordering::Equal)
        .then_with(|| cell_hash(lhs).cmp(&cell_hash(rhs)))
        .then_with(|| lhs.cmp(&rhs))
}

fn visit_hash(idx: usize) -> u64 {
    let x = (idx % STOCHASTIC_CLUSTER_SIDE) as u64;
    let y = (idx / STOCHASTIC_CLUSTER_SIDE) as u64;
    mix_u64(
        x.wrapping_mul(0x632b_e59b_d9b4_e019_u64)
            .wrapping_add(y.wrapping_mul(0x8cb9_2baa_33a5_049f_u64))
            .wrapping_add(0x4f1b_bcdc_b0a7_13f5_u64),
    )
}

fn cell_hash(idx: usize) -> u64 {
    let x = (idx % STOCHASTIC_CLUSTER_SIDE) as u64;
    let y = (idx / STOCHASTIC_CLUSTER_SIDE) as u64;
    mix_u64(
        x.wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
            .wrapping_add(y.wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64))
            .wrapping_add(0x1656_67b1_9e37_79f9_u64),
    )
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd_u64);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53_u64);
    value ^ (value >> 33)
}

#[cfg(test)]
mod tests {
    use super::generate_stochastic_cluster_64x64_flat;

    #[test]
    fn generated_map_has_full_rank_coverage() {
        let map = generate_stochastic_cluster_64x64_flat();
        let mut seen = vec![false; map.len()];

        for &rank in &map {
            let idx = usize::from(rank);
            assert!(idx < map.len());
            assert!(!seen[idx]);
            seen[idx] = true;
        }

        assert!(seen.into_iter().all(|entry| entry));
    }
}
