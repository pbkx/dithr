use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{layout::validate_layout_invariants, PixelLayout, Sample},
    data::maps::generate_void_and_cluster_64x64_flat,
    data::CLUSTER_DOT_8X8_FLAT,
    Buffer, Error, QuantizeMode, Result,
};
use core::cmp::Ordering;
use std::sync::OnceLock;

const AM_FM_SIDE: usize = 64;
const AM_FM_LEN: usize = AM_FM_SIDE * AM_FM_SIDE;
const AM_FM_DENOM: f32 = (AM_FM_LEN - 1) as f32;

const AM_FM_HYBRID_SEED: u64 = 0x6f4d_6861_6c66_746f;
const CLUSTERED_AM_FM_SEED: u64 = 0x636c_7573_7465_7264;

static VOID_CLUSTER_64X64_FLAT: OnceLock<[u16; AM_FM_LEN]> = OnceLock::new();
static AM_FM_HYBRID_64X64_FLAT: OnceLock<[u16; AM_FM_LEN]> = OnceLock::new();
static CLUSTERED_AM_FM_64X64_FLAT: OnceLock<[u16; AM_FM_LEN]> = OnceLock::new();

pub fn am_fm_hybrid_halftoning_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;

    if !L::IS_GRAY {
        return Err(Error::UnsupportedFormat(
            "AM/FM hybrid halftoning supports Gray layouts only",
        ));
    }

    ordered_dither_in_place(
        buffer,
        mode,
        am_fm_hybrid_64x64_flat(),
        AM_FM_SIDE,
        AM_FM_SIDE,
        DEFAULT_STRENGTH,
    )
}

pub fn clustered_am_fm_halftoning_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;

    if !L::IS_GRAY {
        return Err(Error::UnsupportedFormat(
            "clustered AM/FM halftoning supports Gray layouts only",
        ));
    }

    ordered_dither_in_place(
        buffer,
        mode,
        clustered_am_fm_64x64_flat(),
        AM_FM_SIDE,
        AM_FM_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn am_fm_hybrid_64x64_flat() -> &'static [u16; AM_FM_LEN] {
    AM_FM_HYBRID_64X64_FLAT.get_or_init(|| generate_am_fm_map(AM_FM_HYBRID_SEED, AM_FM_HYBRID))
}

fn clustered_am_fm_64x64_flat() -> &'static [u16; AM_FM_LEN] {
    CLUSTERED_AM_FM_64X64_FLAT
        .get_or_init(|| generate_am_fm_map(CLUSTERED_AM_FM_SEED, CLUSTERED_AM_FM))
}

fn void_cluster_64x64_flat() -> &'static [u16; AM_FM_LEN] {
    VOID_CLUSTER_64X64_FLAT.get_or_init(generate_void_and_cluster_64x64_flat)
}

fn generate_am_fm_map(seed: u64, weights: AmFmWeights) -> [u16; AM_FM_LEN] {
    let mut scores = [0.0_f32; AM_FM_LEN];
    let fm = void_cluster_64x64_flat();

    for (idx, slot) in scores.iter_mut().enumerate() {
        let x = idx % AM_FM_SIDE;
        let y = idx / AM_FM_SIDE;
        let fm_norm = f32::from(fm[idx]) / AM_FM_DENOM;
        let am_norm = f32::from(CLUSTER_DOT_8X8_FLAT[(y % 8) * 8 + (x % 8)]) / 63.0;
        let cluster = cluster_profile(x, y);
        let macro_tone = macro_screen_modulation(x, y, seed);
        let jitter = hash01(x, y, seed ^ 0x5bf0_3635_45fa_9f59) * 1.0e-6;

        *slot = (weights.fm * fm_norm
            + weights.am * am_norm
            + weights.cluster * cluster
            + weights.macro_tone * macro_tone
            + jitter)
            .clamp(0.0, 1.0);
    }

    rank_scores(&scores, seed)
}

fn cluster_profile(x: usize, y: usize) -> f32 {
    let local_x = (x % 4) as f32 - 1.5;
    let local_y = (y % 4) as f32 - 1.5;
    let distance = (local_x * local_x + local_y * local_y).sqrt() * 0.471_404_52;
    (1.0 - distance).clamp(0.0, 1.0)
}

fn macro_screen_modulation(x: usize, y: usize, seed: u64) -> f32 {
    hash01(x / 8, y / 8, seed ^ 0xa24b_4fbe_7b73_8d41)
}

fn rank_scores(scores: &[f32; AM_FM_LEN], seed: u64) -> [u16; AM_FM_LEN] {
    let mut order = [0_usize; AM_FM_LEN];
    for (idx, slot) in order.iter_mut().enumerate() {
        *slot = idx;
    }

    order.sort_by(|&lhs, &rhs| compare_cells(scores, lhs, rhs, seed));

    let mut ranks = [0_u16; AM_FM_LEN];
    for (rank, &idx) in order.iter().enumerate() {
        ranks[idx] = rank as u16;
    }

    ranks
}

fn compare_cells(scores: &[f32; AM_FM_LEN], lhs: usize, rhs: usize, seed: u64) -> Ordering {
    scores[lhs]
        .partial_cmp(&scores[rhs])
        .unwrap_or(Ordering::Equal)
        .then_with(|| cell_hash(lhs, seed).cmp(&cell_hash(rhs, seed)))
        .then_with(|| lhs.cmp(&rhs))
}

fn cell_hash(idx: usize, seed: u64) -> u64 {
    let x = (idx % AM_FM_SIDE) as u64;
    let y = (idx / AM_FM_SIDE) as u64;
    mix_u64(
        x.wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
            .wrapping_add(y.wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64))
            .wrapping_add(seed),
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

#[derive(Clone, Copy)]
struct AmFmWeights {
    fm: f32,
    am: f32,
    cluster: f32,
    macro_tone: f32,
}

const AM_FM_HYBRID: AmFmWeights = AmFmWeights {
    fm: 0.55,
    am: 0.27,
    cluster: 0.08,
    macro_tone: 0.10,
};

const CLUSTERED_AM_FM: AmFmWeights = AmFmWeights {
    fm: 0.26,
    am: 0.48,
    cluster: 0.18,
    macro_tone: 0.08,
};

#[cfg(test)]
mod tests {
    use super::{
        generate_am_fm_map, AM_FM_HYBRID, AM_FM_HYBRID_SEED, CLUSTERED_AM_FM, CLUSTERED_AM_FM_SEED,
    };

    #[test]
    fn generated_am_fm_maps_have_full_rank_coverage() {
        let hybrid = generate_am_fm_map(AM_FM_HYBRID_SEED, AM_FM_HYBRID);
        let clustered = generate_am_fm_map(CLUSTERED_AM_FM_SEED, CLUSTERED_AM_FM);

        let mut seen_hybrid = vec![false; hybrid.len()];
        for &rank in &hybrid {
            let idx = usize::from(rank);
            assert!(idx < hybrid.len());
            assert!(!seen_hybrid[idx]);
            seen_hybrid[idx] = true;
        }
        assert!(seen_hybrid.into_iter().all(|entry| entry));

        let mut seen_clustered = vec![false; clustered.len()];
        for &rank in &clustered {
            let idx = usize::from(rank);
            assert!(idx < clustered.len());
            assert!(!seen_clustered[idx]);
            seen_clustered[idx] = true;
        }
        assert!(seen_clustered.into_iter().all(|entry| entry));
    }

    #[test]
    fn generated_am_fm_maps_are_deterministic() {
        assert_eq!(
            generate_am_fm_map(AM_FM_HYBRID_SEED, AM_FM_HYBRID),
            generate_am_fm_map(AM_FM_HYBRID_SEED, AM_FM_HYBRID)
        );
        assert_eq!(
            generate_am_fm_map(CLUSTERED_AM_FM_SEED, CLUSTERED_AM_FM),
            generate_am_fm_map(CLUSTERED_AM_FM_SEED, CLUSTERED_AM_FM)
        );
    }
}
