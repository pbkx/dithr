mod adaptive;
mod am_fm;
mod bayer;
mod cluster;
pub(crate) mod core;
mod image_based_screen;
mod multitone;
mod polyomino;
mod ranked;
mod space_filling;
mod stochastic_cluster;
mod void_cluster;
mod yliluoma;

use crate::{
    core::{PixelLayout, Sample},
    Buffer, Palette, QuantizeMode, Result,
};
pub(crate) const DEFAULT_STRENGTH: f32 = 64.0 / 255.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedError {
    EmptyMap,
    InvalidDimensions,
    ValueOutOfRange,
}

impl std::fmt::Display for OrderedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyMap => f.write_str("ordered threshold map must not be empty"),
            Self::InvalidDimensions => f.write_str("ordered threshold map dimensions are invalid"),
            Self::ValueOutOfRange => {
                f.write_str("ordered threshold map contains values out of range")
            }
        }
    }
}

impl std::error::Error for OrderedError {}

pub use adaptive::adaptive_ordered_dither_in_place;
pub use am_fm::{am_fm_hybrid_halftoning_in_place, clustered_am_fm_halftoning_in_place};
pub use bayer::{bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place};
#[cfg(feature = "rayon")]
pub use bayer::{
    bayer_16x16_in_place_par, bayer_2x2_in_place_par, bayer_4x4_in_place_par,
    bayer_8x8_in_place_par,
};
pub use cluster::{cluster_dot_4x4_in_place, cluster_dot_8x8_in_place};
#[cfg(feature = "rayon")]
pub use cluster::{cluster_dot_4x4_in_place_par, cluster_dot_8x8_in_place_par};
pub use image_based_screen::image_based_dither_screen_in_place;
pub use multitone::blue_noise_multitone_dither_in_place;
pub use polyomino::polyomino_ordered_dither_in_place;
pub use ranked::ranked_dither_in_place;
pub use space_filling::space_filling_curve_ordered_dither_in_place;
pub use stochastic_cluster::stochastic_clustered_dot_in_place;
pub use void_cluster::void_and_cluster_in_place;

#[doc(hidden)]
pub(crate) fn ordered_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u16],
    map_w: usize,
    map_h: usize,
    strength: f32,
) -> Result<()> {
    core::ordered_dither_in_place(buffer, mode, map, map_w, map_h, strength)
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
    core::ordered_dither_in_place_par(buffer, mode, map, map_w, map_h, strength)
}

pub fn custom_ordered_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> Result<()> {
    validate_custom_map(map, map_w, map_h)?;
    let ranked_map: Vec<u16> = map.iter().map(|&value| u16::from(value)).collect();

    ordered_dither_in_place(
        buffer,
        mode,
        &ranked_map,
        map_w,
        map_h,
        f32::from(strength) / 255.0,
    )
}

#[cfg(feature = "rayon")]
pub fn custom_ordered_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> Result<()> {
    validate_custom_map(map, map_w, map_h)?;
    let ranked_map: Vec<u16> = map.iter().map(|&value| u16::from(value)).collect();

    ordered_dither_in_place_par(
        buffer,
        mode,
        &ranked_map,
        map_w,
        map_h,
        f32::from(strength) / 255.0,
    )
}

pub fn yliluoma_1_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> Result<()> {
    yliluoma::yliluoma_1_in_place(buffer, palette)
}

#[cfg(feature = "rayon")]
pub fn yliluoma_1_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> Result<()> {
    yliluoma::yliluoma_1_in_place_par(buffer, palette)
}

pub fn yliluoma_2_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> Result<()> {
    yliluoma::yliluoma_2_in_place(buffer, palette)
}

#[cfg(feature = "rayon")]
pub fn yliluoma_2_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> Result<()> {
    yliluoma::yliluoma_2_in_place_par(buffer, palette)
}

pub fn yliluoma_3_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> Result<()> {
    yliluoma::yliluoma_3_in_place(buffer, palette)
}

#[cfg(feature = "rayon")]
pub fn yliluoma_3_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> Result<()> {
    yliluoma::yliluoma_3_in_place_par(buffer, palette)
}

fn validate_custom_map(
    map: &[u8],
    map_w: usize,
    map_h: usize,
) -> std::result::Result<(), OrderedError> {
    if map.is_empty() {
        return Err(OrderedError::EmptyMap);
    }

    let expected_len = map_w
        .checked_mul(map_h)
        .ok_or(OrderedError::InvalidDimensions)?;
    if expected_len == 0 || expected_len != map.len() {
        return Err(OrderedError::InvalidDimensions);
    }

    if map.iter().any(|&value| usize::from(value) >= expected_len) {
        return Err(OrderedError::ValueOutOfRange);
    }

    Ok(())
}
