use super::core::{ordered_threshold_for_xy, ordered_threshold_unit};
use crate::{
    core::{alpha_index, layout::validate_layout_invariants, read_unit_pixel, PixelLayout, Sample},
    data::{
        generate_bayer_16x16_flat, BAYER_2X2_FLAT, BAYER_4X4_FLAT, BAYER_8X8_FLAT,
        CLUSTER_DOT_4X4_FLAT, CLUSTER_DOT_8X8_FLAT,
    },
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};
use std::sync::OnceLock;

const ADAPTIVE_STRENGTH: f32 = 64.0 / 255.0;

static BAYER_16X16_FLAT: OnceLock<[u16; 256]> = OnceLock::new();

struct ThresholdMap<'a> {
    data: &'a [u16],
    width: usize,
    height: usize,
    denom: u16,
}

pub fn adaptive_ordered_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;
    if !ordered_layout_supported::<L>() {
        return Err(Error::UnsupportedFormat(
            "adaptive ordered dithering supports Gray, Rgb, and Rgba formats only",
        ));
    }

    let width = buffer.width();
    let height = buffer.height();
    let stride = buffer.stride();
    let source = buffer.data().to_vec();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for x in 0..width {
            let activity = local_activity::<S, L>(&source, width, height, stride, x, y)?;
            let map = select_threshold_map(activity);
            let threshold = ordered_threshold_for_xy(x, y, map.data, map.width, map.height)?;
            let offset = x.checked_mul(L::CHANNELS).ok_or(BufferError::OutOfBounds)?;
            let end = offset
                .checked_add(L::CHANNELS)
                .ok_or(BufferError::OutOfBounds)?;
            let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
            adaptive_apply_pixel::<S, L>(pixel, threshold, map.denom, mode)?;
        }
    }

    Ok(())
}

fn select_threshold_map(activity: f32) -> ThresholdMap<'static> {
    if activity < 0.08 {
        return ThresholdMap {
            data: &CLUSTER_DOT_8X8_FLAT,
            width: 8,
            height: 8,
            denom: 64,
        };
    }

    if activity < 0.16 {
        return ThresholdMap {
            data: &CLUSTER_DOT_4X4_FLAT,
            width: 4,
            height: 4,
            denom: 16,
        };
    }

    if activity < 0.26 {
        return ThresholdMap {
            data: bayer_16x16_flat(),
            width: 16,
            height: 16,
            denom: 256,
        };
    }

    if activity < 0.38 {
        return ThresholdMap {
            data: &BAYER_8X8_FLAT,
            width: 8,
            height: 8,
            denom: 64,
        };
    }

    if activity < 0.52 {
        return ThresholdMap {
            data: &BAYER_4X4_FLAT,
            width: 4,
            height: 4,
            denom: 16,
        };
    }

    ThresholdMap {
        data: &BAYER_2X2_FLAT,
        width: 2,
        height: 2,
        denom: 4,
    }
}

fn local_activity<S: Sample, L: PixelLayout>(
    source: &[S],
    width: usize,
    height: usize,
    stride: usize,
    x: usize,
    y: usize,
) -> Result<f32> {
    let x_prev = x.saturating_sub(1);
    let y_prev = y.saturating_sub(1);
    let x_next = (x + 1).min(width - 1);
    let y_next = (y + 1).min(height - 1);

    let center = luma_at::<S, L>(source, stride, x, y)?;
    let left = luma_at::<S, L>(source, stride, x_prev, y)?;
    let right = luma_at::<S, L>(source, stride, x_next, y)?;
    let up = luma_at::<S, L>(source, stride, x, y_prev)?;
    let down = luma_at::<S, L>(source, stride, x, y_next)?;

    let gx = (right - left).abs();
    let gy = (down - up).abs();
    let lap = (4.0 * center - left - right - up - down).abs();

    Ok((gx + gy + lap * 0.5).clamp(0.0, 1.0))
}

fn luma_at<S: Sample, L: PixelLayout>(
    source: &[S],
    stride: usize,
    x: usize,
    y: usize,
) -> Result<f32> {
    let row_start = y.checked_mul(stride).ok_or(BufferError::OutOfBounds)?;
    let offset = x
        .checked_mul(L::CHANNELS)
        .and_then(|in_row| row_start.checked_add(in_row))
        .ok_or(BufferError::OutOfBounds)?;
    let end = offset
        .checked_add(L::CHANNELS)
        .ok_or(BufferError::OutOfBounds)?;
    let pixel = source.get(offset..end).ok_or(BufferError::OutOfBounds)?;
    let rgba = read_unit_pixel::<S, L>(pixel)?;
    Ok((0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]).clamp(0.0, 1.0))
}

fn adaptive_apply_pixel<S: Sample, L: PixelLayout>(
    pixel: &mut [S],
    threshold_rank: u16,
    threshold_den: u16,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    if pixel.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }

    let preserved_alpha = alpha_index::<L>().and_then(|idx| pixel.get(idx).copied());
    let mut rgba = read_unit_pixel::<S, L>(pixel)?;
    let threshold = ordered_threshold_unit(threshold_rank, threshold_den, ADAPTIVE_STRENGTH);
    for channel in rgba.iter_mut().take(3) {
        *channel = (*channel + threshold).clamp(0.0, 1.0);
    }

    let biased = [
        S::from_unit_f32(rgba[0]),
        S::from_unit_f32(rgba[1]),
        S::from_unit_f32(rgba[2]),
        S::from_unit_f32(rgba[3]),
    ];
    let quantized = quantize_pixel::<S, L>(&biased[..L::CHANNELS], mode)?;
    pixel[..L::COLOR_CHANNELS].copy_from_slice(&quantized[..L::COLOR_CHANNELS]);
    if let Some(alpha_lane) = alpha_index::<L>() {
        pixel[alpha_lane] = preserved_alpha.ok_or(BufferError::OutOfBounds)?;
    }

    Ok(())
}

fn bayer_16x16_flat() -> &'static [u16; 256] {
    BAYER_16X16_FLAT.get_or_init(generate_bayer_16x16_flat)
}

const fn ordered_layout_supported<L: PixelLayout>() -> bool {
    (L::HAS_ALPHA && L::CHANNELS == 4) || (!L::HAS_ALPHA && (L::CHANNELS == 1 || L::CHANNELS == 3))
}
