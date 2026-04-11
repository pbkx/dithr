use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{layout::validate_layout_invariants, read_unit_pixel, PixelLayout, Sample},
    Buffer, BufferError, Error, QuantizeMode, Result,
};
use core::cmp::Ordering;

const IMAGE_BASED_SIDE: usize = 16;
const IMAGE_BASED_LEN: usize = IMAGE_BASED_SIDE * IMAGE_BASED_SIDE;
const TILE_SIDE: usize = 4;
const TILE_COUNT: usize = IMAGE_BASED_SIDE / TILE_SIDE;
const TILE_TOTAL: usize = TILE_COUNT * TILE_COUNT;
const HIST_BINS: usize = 64;
const CLIP_LIMIT: u16 = 2;
const TILE_AREA_F32: f32 = (TILE_SIDE * TILE_SIDE) as f32;

pub fn image_based_dither_screen_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;
    if !ordered_layout_supported::<L>() {
        return Err(Error::UnsupportedFormat(
            "image-based dither screens support Gray, Rgb, and Rgba formats only",
        ));
    }

    let screen = build_image_based_screen_16x16::<S, L>(buffer)?;
    ordered_dither_in_place(
        buffer,
        mode,
        &screen,
        IMAGE_BASED_SIDE,
        IMAGE_BASED_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn build_image_based_screen_16x16<S: Sample, L: PixelLayout>(
    buffer: &Buffer<'_, S, L>,
) -> Result<[u16; IMAGE_BASED_LEN]> {
    let sampled = sample_luma_grid_16x16::<S, L>(buffer)?;
    let equalized = local_equalized_values(&sampled);
    Ok(ranked_screen(&equalized))
}

fn sample_luma_grid_16x16<S: Sample, L: PixelLayout>(
    buffer: &Buffer<'_, S, L>,
) -> Result<[f32; IMAGE_BASED_LEN]> {
    let width = buffer.width();
    let height = buffer.height();
    let mut out = [0.0_f32; IMAGE_BASED_LEN];

    for sy in 0..IMAGE_BASED_SIDE {
        let y0 = sy * height / IMAGE_BASED_SIDE;
        let mut y1 = (sy + 1) * height / IMAGE_BASED_SIDE;
        if y1 <= y0 {
            y1 = (y0 + 1).min(height);
        }

        for sx in 0..IMAGE_BASED_SIDE {
            let x0 = sx * width / IMAGE_BASED_SIDE;
            let mut x1 = (sx + 1) * width / IMAGE_BASED_SIDE;
            if x1 <= x0 {
                x1 = (x0 + 1).min(width);
            }

            let mut sum = 0.0_f32;
            let mut count = 0_usize;
            for y in y0..y1 {
                let row = buffer.try_row(y)?;
                for x in x0..x1 {
                    let offset = x.checked_mul(L::CHANNELS).ok_or(BufferError::OutOfBounds)?;
                    let end = offset
                        .checked_add(L::CHANNELS)
                        .ok_or(BufferError::OutOfBounds)?;
                    let pixel = row.get(offset..end).ok_or(BufferError::OutOfBounds)?;
                    let rgba = read_unit_pixel::<S, L>(pixel)?;
                    let luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
                    sum += luma.clamp(0.0, 1.0);
                    count = count.saturating_add(1);
                }
            }

            let idx = sy * IMAGE_BASED_SIDE + sx;
            if count == 0 {
                out[idx] = 0.0;
            } else {
                out[idx] = (sum / count as f32).clamp(0.0, 1.0);
            }
        }
    }

    Ok(out)
}

fn local_equalized_values(values: &[f32; IMAGE_BASED_LEN]) -> [f32; IMAGE_BASED_LEN] {
    let mut cdfs = [[0.0_f32; HIST_BINS]; TILE_TOTAL];

    for tile_y in 0..TILE_COUNT {
        for tile_x in 0..TILE_COUNT {
            let mut hist = [0_u16; HIST_BINS];

            let y_start = tile_y * TILE_SIDE;
            let x_start = tile_x * TILE_SIDE;
            for y in y_start..(y_start + TILE_SIDE) {
                for x in x_start..(x_start + TILE_SIDE) {
                    let idx = y * IMAGE_BASED_SIDE + x;
                    let bin = value_bin(values[idx]);
                    hist[bin] = hist[bin].saturating_add(1);
                }
            }

            clip_histogram(&mut hist);
            let mut cumulative = 0_u32;
            let cdf_slot = tile_index(tile_x, tile_y);
            for (bin, &count) in hist.iter().enumerate() {
                cumulative = cumulative.saturating_add(u32::from(count));
                cdfs[cdf_slot][bin] = (cumulative as f32 / TILE_AREA_F32).clamp(0.0, 1.0);
            }
        }
    }

    let mut out = [0.0_f32; IMAGE_BASED_LEN];
    for y in 0..IMAGE_BASED_SIDE {
        for x in 0..IMAGE_BASED_SIDE {
            let tx0 = x / TILE_SIDE;
            let ty0 = y / TILE_SIDE;
            let tx1 = (tx0 + 1).min(TILE_COUNT - 1);
            let ty1 = (ty0 + 1).min(TILE_COUNT - 1);

            let fx = if tx0 == tx1 {
                0.0
            } else {
                (x % TILE_SIDE) as f32 / TILE_SIDE as f32
            };
            let fy = if ty0 == ty1 {
                0.0
            } else {
                (y % TILE_SIDE) as f32 / TILE_SIDE as f32
            };

            let idx = y * IMAGE_BASED_SIDE + x;
            let bin = value_bin(values[idx]);

            let c00 = cdfs[tile_index(tx0, ty0)][bin];
            let c10 = cdfs[tile_index(tx1, ty0)][bin];
            let c01 = cdfs[tile_index(tx0, ty1)][bin];
            let c11 = cdfs[tile_index(tx1, ty1)][bin];

            let top = lerp(c00, c10, fx);
            let bottom = lerp(c01, c11, fx);
            out[idx] = lerp(top, bottom, fy).clamp(0.0, 1.0);
        }
    }

    out
}

fn ranked_screen(values: &[f32; IMAGE_BASED_LEN]) -> [u16; IMAGE_BASED_LEN] {
    let mut order = [0_usize; IMAGE_BASED_LEN];
    for (index, slot) in order.iter_mut().enumerate() {
        *slot = index;
    }

    order.sort_by(|&lhs, &rhs| compare_screen_cells(values, lhs, rhs));

    let mut ranks = [0_u16; IMAGE_BASED_LEN];
    for (rank, &index) in order.iter().enumerate() {
        ranks[index] = rank as u16;
    }

    ranks
}

fn compare_screen_cells(values: &[f32; IMAGE_BASED_LEN], lhs: usize, rhs: usize) -> Ordering {
    values[lhs]
        .partial_cmp(&values[rhs])
        .unwrap_or(Ordering::Equal)
        .then_with(|| screen_hash(lhs).cmp(&screen_hash(rhs)))
        .then_with(|| lhs.cmp(&rhs))
}

fn clip_histogram(hist: &mut [u16; HIST_BINS]) {
    let mut excess = 0_u32;
    for count in hist.iter_mut() {
        if *count > CLIP_LIMIT {
            excess = excess.saturating_add(u32::from(*count - CLIP_LIMIT));
            *count = CLIP_LIMIT;
        }
    }

    if excess == 0 {
        return;
    }

    let bins_u32 = HIST_BINS as u32;
    let increment = (excess / bins_u32) as u16;
    let mut remainder = (excess % bins_u32) as usize;

    if increment > 0 {
        for count in hist.iter_mut() {
            *count = count.saturating_add(increment);
        }
    }

    let mut idx = 0_usize;
    while remainder > 0 {
        hist[idx] = hist[idx].saturating_add(1);
        idx += 1;
        if idx == HIST_BINS {
            idx = 0;
        }
        remainder -= 1;
    }
}

fn value_bin(value: f32) -> usize {
    let scaled = (value.clamp(0.0, 1.0) * (HIST_BINS - 1) as f32).round();
    let clamped = scaled.clamp(0.0, (HIST_BINS - 1) as f32);
    clamped as usize
}

fn tile_index(x: usize, y: usize) -> usize {
    y * TILE_COUNT + x
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn screen_hash(idx: usize) -> u64 {
    let x = (idx % IMAGE_BASED_SIDE) as u64;
    let y = (idx / IMAGE_BASED_SIDE) as u64;
    mix_u64(
        x.wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
            .wrapping_add(y.wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64))
            .wrapping_add(0x4f1b_bcdc_b0a7_13f5_u64),
    )
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd_u64);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53_u64);
    value ^ (value >> 33)
}

const fn ordered_layout_supported<L: PixelLayout>() -> bool {
    (L::HAS_ALPHA && L::CHANNELS == 4) || (!L::HAS_ALPHA && (L::CHANNELS == 1 || L::CHANNELS == 3))
}
