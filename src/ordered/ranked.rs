use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{PixelLayout, Sample},
    Buffer, QuantizeMode, Result,
};
use std::sync::OnceLock;

const RANKED_SIDE: usize = 16;
const RANKED_LEN: usize = RANKED_SIDE * RANKED_SIDE;

static RANKED_16X16_FLAT: OnceLock<[u16; RANKED_LEN]> = OnceLock::new();

pub fn ranked_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(
        buffer,
        mode,
        ranked_16x16_flat(),
        RANKED_SIDE,
        RANKED_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn ranked_16x16_flat() -> &'static [u16; RANKED_LEN] {
    RANKED_16X16_FLAT.get_or_init(generate_ranked_16x16_flat)
}

fn generate_ranked_16x16_flat() -> [u16; RANKED_LEN] {
    let mut ranks = [u16::MAX; RANKED_LEN];
    let mut selected = Vec::with_capacity(RANKED_LEN);

    for rank in 0..RANKED_LEN {
        let candidate = select_rank_candidate(&ranks, &selected);
        let Some(idx) = candidate else {
            break;
        };
        ranks[idx] = rank as u16;
        selected.push(idx);
    }

    let mut next = 0_u16;
    for entry in &mut ranks {
        if *entry == u16::MAX {
            *entry = next;
            next = next.saturating_add(1);
        }
    }

    ranks
}

fn select_rank_candidate(ranks: &[u16; RANKED_LEN], selected: &[usize]) -> Option<usize> {
    let mut best_idx = None;
    let mut best_score = 0_u16;
    let mut best_hash = u64::MAX;

    for (idx, &rank) in ranks.iter().enumerate() {
        if rank != u16::MAX {
            continue;
        }

        let score = if selected.is_empty() {
            u16::MAX
        } else {
            min_toroidal_distance_sq(idx, selected)
        };
        let hash = ranked_hash(idx);

        let better = match best_idx {
            None => true,
            Some(current) => {
                score > best_score
                    || (score == best_score
                        && (hash < best_hash || (hash == best_hash && idx < current)))
            }
        };

        if better {
            best_idx = Some(idx);
            best_score = score;
            best_hash = hash;
        }
    }

    best_idx
}

fn min_toroidal_distance_sq(idx: usize, selected: &[usize]) -> u16 {
    let mut best = u16::MAX;
    for &other in selected {
        let distance = toroidal_distance_sq(idx, other);
        if distance < best {
            best = distance;
            if best == 0 {
                break;
            }
        }
    }
    best
}

fn toroidal_distance_sq(a_idx: usize, b_idx: usize) -> u16 {
    let ax = a_idx % RANKED_SIDE;
    let ay = a_idx / RANKED_SIDE;
    let bx = b_idx % RANKED_SIDE;
    let by = b_idx / RANKED_SIDE;

    let dx = toroidal_axis_delta(ax, bx, RANKED_SIDE);
    let dy = toroidal_axis_delta(ay, by, RANKED_SIDE);

    (dx * dx + dy * dy) as u16
}

fn toroidal_axis_delta(a: usize, b: usize, side: usize) -> usize {
    let delta = a.abs_diff(b);
    delta.min(side - delta)
}

fn ranked_hash(idx: usize) -> u64 {
    let x = (idx % RANKED_SIDE) as u64;
    let y = (idx / RANKED_SIDE) as u64;
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
