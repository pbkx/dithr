use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{layout::validate_layout_invariants, PixelLayout, Sample},
    data::{maps::generate_void_and_cluster_64x64_flat, CLUSTER_DOT_8X8_FLAT},
    Buffer, Error, QuantizeMode, Result,
};
use core::cmp::Ordering;
use std::sync::OnceLock;

const BLUE_NOISE_MULTITONE_SIDE: usize = 64;
const BLUE_NOISE_MULTITONE_LEN: usize = BLUE_NOISE_MULTITONE_SIDE * BLUE_NOISE_MULTITONE_SIDE;
const BLUE_NOISE_MULTITONE_SEED: u64 = 0x626c_7565_6e6f_6973;
const BLUE_NOISE_MULTITONE_BASE_WEIGHT: f32 = 0.86;
const BLUE_NOISE_MULTITONE_CLUSTER_WEIGHT: f32 = 0.09;
const BLUE_NOISE_MULTITONE_RING_WEIGHT: f32 = 0.05;

static BLUE_NOISE_MULTITONE_64X64_FLAT: OnceLock<[u16; BLUE_NOISE_MULTITONE_LEN]> = OnceLock::new();

pub fn blue_noise_multitone_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;

    if !L::IS_GRAY {
        return Err(Error::UnsupportedFormat(
            "blue-noise multitone dithering supports Gray layouts only",
        ));
    }

    ordered_dither_in_place(
        buffer,
        mode,
        blue_noise_multitone_64x64_flat(),
        BLUE_NOISE_MULTITONE_SIDE,
        BLUE_NOISE_MULTITONE_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn blue_noise_multitone_64x64_flat() -> &'static [u16; BLUE_NOISE_MULTITONE_LEN] {
    BLUE_NOISE_MULTITONE_64X64_FLAT.get_or_init(generate_blue_noise_multitone_64x64_flat)
}

fn generate_blue_noise_multitone_64x64_flat() -> [u16; BLUE_NOISE_MULTITONE_LEN] {
    let base = generate_void_and_cluster_64x64_flat();
    let mut scores = [0.0_f32; BLUE_NOISE_MULTITONE_LEN];

    for (idx, slot) in scores.iter_mut().enumerate() {
        let x = idx % BLUE_NOISE_MULTITONE_SIDE;
        let y = idx / BLUE_NOISE_MULTITONE_SIDE;
        let base_norm = f32::from(base[idx]) / (BLUE_NOISE_MULTITONE_LEN as f32 - 1.0);
        let clustered_norm = f32::from(CLUSTER_DOT_8X8_FLAT[(y % 8) * 8 + (x % 8)]) / 63.0;
        let ring_norm = micro_ring_value(x, y);
        let jitter = hash01(x, y, BLUE_NOISE_MULTITONE_SEED ^ 0x9c4f_2a3e_5d17_801b) * 1.0e-6;

        *slot = (base_norm * BLUE_NOISE_MULTITONE_BASE_WEIGHT
            + clustered_norm * BLUE_NOISE_MULTITONE_CLUSTER_WEIGHT
            + ring_norm * BLUE_NOISE_MULTITONE_RING_WEIGHT
            + jitter)
            .clamp(0.0, 1.0);
    }

    rank_scores(&scores)
}

fn micro_ring_value(x: usize, y: usize) -> f32 {
    let tx = (x % 4) as f32 - 1.5;
    let ty = (y % 4) as f32 - 1.5;
    ((tx * tx + ty * ty).sqrt() * 0.471_404_52).clamp(0.0, 1.0)
}

fn rank_scores(scores: &[f32; BLUE_NOISE_MULTITONE_LEN]) -> [u16; BLUE_NOISE_MULTITONE_LEN] {
    let mut order = [0_usize; BLUE_NOISE_MULTITONE_LEN];
    for (idx, slot) in order.iter_mut().enumerate() {
        *slot = idx;
    }

    order.sort_by(|&lhs, &rhs| compare_cells(scores, lhs, rhs));

    let mut ranks = [0_u16; BLUE_NOISE_MULTITONE_LEN];
    for (rank, &idx) in order.iter().enumerate() {
        ranks[idx] = rank as u16;
    }

    ranks
}

fn compare_cells(scores: &[f32; BLUE_NOISE_MULTITONE_LEN], lhs: usize, rhs: usize) -> Ordering {
    scores[lhs]
        .partial_cmp(&scores[rhs])
        .unwrap_or(Ordering::Equal)
        .then_with(|| cell_hash(lhs).cmp(&cell_hash(rhs)))
        .then_with(|| lhs.cmp(&rhs))
}

fn cell_hash(idx: usize) -> u64 {
    let x = (idx % BLUE_NOISE_MULTITONE_SIDE) as u64;
    let y = (idx / BLUE_NOISE_MULTITONE_SIDE) as u64;
    mix_u64(
        x.wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
            .wrapping_add(y.wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64))
            .wrapping_add(BLUE_NOISE_MULTITONE_SEED),
    )
}

fn hash01(x: usize, y: usize, seed: u64) -> f32 {
    let mixed = mix_u64(
        (x as u64)
            .wrapping_mul(0x632b_e59b_d9b4_e019_u64)
            .wrapping_add((y as u64).wrapping_mul(0x8cb9_2baa_33a5_049f_u64))
            .wrapping_add(seed),
    );
    (mixed as f64 / u64::MAX as f64) as f32
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
    use super::generate_blue_noise_multitone_64x64_flat;

    #[test]
    fn generated_map_has_full_rank_coverage() {
        let map = generate_blue_noise_multitone_64x64_flat();
        let mut seen = vec![false; map.len()];

        for &rank in &map {
            let idx = usize::from(rank);
            assert!(idx < map.len());
            assert!(!seen[idx]);
            seen[idx] = true;
        }

        assert!(seen.into_iter().all(|entry| entry));
    }

    #[test]
    fn generated_map_is_deterministic() {
        assert_eq!(
            generate_blue_noise_multitone_64x64_flat(),
            generate_blue_noise_multitone_64x64_flat()
        );
    }
}
