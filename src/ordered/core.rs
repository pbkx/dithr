use crate::{
    core::{alpha_index, read_unit_pixel, PixelLayout, Sample},
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(crate) fn ordered_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u16],
    map_w: usize,
    map_h: usize,
    strength: f32,
) -> Result<()> {
    buffer.validate()?;
    let (map_min, threshold_den) = validate_map(map, map_w, map_h)?;

    let width = buffer.width;
    let height = buffer.height;
    let ctx = OrderedDitherCtx {
        map,
        map_w,
        map_h,
        map_min,
        threshold_den,
        strength,
    };

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        ordered_dither_row::<S, L>(row, y, width, &ctx, mode)?;
    }

    Ok(())
}

#[cfg(feature = "rayon")]
pub(crate) fn ordered_dither_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u16],
    map_w: usize,
    map_h: usize,
    strength: f32,
) -> Result<()> {
    buffer.validate()?;
    let (map_min, threshold_den) = validate_map(map, map_w, map_h)?;

    let width = buffer.width;
    let height = buffer.height;
    let stride = buffer.stride;
    let ctx = OrderedDitherCtx {
        map,
        map_w,
        map_h,
        map_min,
        threshold_den,
        strength,
    };

    buffer
        .data
        .par_chunks_mut(stride)
        .take(height)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<()> {
            ordered_dither_row::<S, L>(row, y, width, &ctx, mode)
        })?;

    Ok(())
}

struct OrderedDitherCtx<'a> {
    map: &'a [u16],
    map_w: usize,
    map_h: usize,
    map_min: u16,
    threshold_den: u16,
    strength: f32,
}

fn ordered_dither_row<S: Sample, L: PixelLayout>(
    row: &mut [S],
    y: usize,
    width: usize,
    ctx: &OrderedDitherCtx<'_>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    for x in 0..width {
        let threshold = ordered_threshold_for_xy(x, y, ctx.map, ctx.map_w, ctx.map_h)?;
        let threshold_rank = threshold.saturating_sub(ctx.map_min);
        let offset = x.checked_mul(L::CHANNELS).ok_or(BufferError::OutOfBounds)?;
        let end = offset
            .checked_add(L::CHANNELS)
            .ok_or(BufferError::OutOfBounds)?;
        let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
        if ctx.strength == 1.0 {
            ordered_apply_pixel::<S, L>(pixel, threshold_rank, ctx.threshold_den, mode)?;
        } else {
            ordered_apply_pixel_with_strength::<S, L>(
                pixel,
                threshold_rank,
                ctx.threshold_den,
                ctx.strength,
                mode,
            )?;
        }
    }

    Ok(())
}

pub(crate) fn ordered_apply_pixel<S: Sample, L: PixelLayout>(
    pixel: &mut [S],
    threshold_rank: u16,
    threshold_den: u16,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_apply_pixel_with_strength::<S, L>(pixel, threshold_rank, threshold_den, 1.0, mode)
}

fn ordered_apply_pixel_with_strength<S: Sample, L: PixelLayout>(
    pixel: &mut [S],
    threshold_rank: u16,
    threshold_den: u16,
    strength: f32,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    if !ordered_layout_supported::<L>() {
        return Err(Error::UnsupportedFormat(
            "ordered dithering supports Gray, Rgb, and Rgba formats only",
        ));
    }

    if pixel.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }

    let preserved_alpha = alpha_index::<L>().and_then(|idx| pixel.get(idx).copied());
    let mut rgba = read_unit_pixel::<S, L>(pixel)?;
    let threshold = ordered_threshold_unit(threshold_rank, threshold_den, strength);
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

pub(crate) fn ordered_threshold_unit(rank: u16, denom: u16, strength: f32) -> f32 {
    if denom <= 1 || strength == 0.0 {
        return 0.0;
    }

    let range = f32::from(denom - 1);
    let centered = f32::from(rank) * 2.0 - range;
    let scaled_steps = (centered * (strength * 255.0) / range).trunc();
    scaled_steps / 255.0
}

pub(crate) fn ordered_threshold_for_xy(
    x: usize,
    y: usize,
    map: &[u16],
    map_w: usize,
    map_h: usize,
) -> Result<u16> {
    if map_w == 0 || map_h == 0 {
        return Err(Error::InvalidArgument(
            "ordered map dimensions must be positive",
        ));
    }
    let expected_len = map_w
        .checked_mul(map_h)
        .ok_or(Error::InvalidArgument("ordered map dimensions overflow"))?;
    if map.len() != expected_len {
        return Err(Error::InvalidArgument(
            "ordered map length must match dimensions",
        ));
    }
    let map_x = x % map_w;
    let map_y = y % map_h;
    let idx = map_y
        .checked_mul(map_w)
        .and_then(|base| base.checked_add(map_x))
        .ok_or(Error::InvalidArgument("ordered map indexing overflow"))?;
    map.get(idx)
        .copied()
        .ok_or(Error::InvalidArgument("ordered map index out of bounds"))
}

fn validate_map(map: &[u16], map_w: usize, map_h: usize) -> Result<(u16, u16)> {
    if map_w == 0 || map_h == 0 {
        return Err(Error::InvalidArgument(
            "ordered map dimensions must be positive",
        ));
    }

    let expected_len = map_w
        .checked_mul(map_h)
        .ok_or(Error::InvalidArgument("ordered map dimensions overflow"))?;
    if map.len() != expected_len {
        return Err(Error::InvalidArgument(
            "ordered map length must match dimensions",
        ));
    }

    let map_min = *map
        .iter()
        .min()
        .ok_or(Error::InvalidArgument("ordered map must not be empty"))?;
    let map_max = *map
        .iter()
        .max()
        .ok_or(Error::InvalidArgument("ordered map must not be empty"))?;
    let threshold_den = map_max.saturating_sub(map_min).saturating_add(1);

    Ok((map_min, threshold_den))
}

const fn ordered_layout_supported<L: PixelLayout>() -> bool {
    (L::HAS_ALPHA && L::CHANNELS == 4) || (!L::HAS_ALPHA && (L::CHANNELS == 1 || L::CHANNELS == 3))
}

#[cfg(test)]
mod tests {
    use super::ordered_threshold_for_xy;
    use crate::Error;

    #[test]
    fn ordered_threshold_rejects_zero_dimensions() {
        let map = [0_u16, 1, 2, 3];
        assert_eq!(
            ordered_threshold_for_xy(0, 0, &map, 0, 2),
            Err(Error::InvalidArgument(
                "ordered map dimensions must be positive"
            ))
        );
    }

    #[test]
    fn ordered_threshold_rejects_mismatched_map_length() {
        let map = [0_u16, 1, 2];
        assert_eq!(
            ordered_threshold_for_xy(0, 0, &map, 2, 2),
            Err(Error::InvalidArgument(
                "ordered map length must match dimensions"
            ))
        );
    }
}
