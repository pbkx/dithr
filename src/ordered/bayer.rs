#[cfg(feature = "rayon")]
use super::ordered_dither_in_place_par;
use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{PixelLayout, Sample},
    data::{generate_bayer_16x16_flat, BAYER_2X2_FLAT, BAYER_4X4_FLAT, BAYER_8X8_FLAT},
    Buffer, QuantizeMode, Result,
};
use std::sync::OnceLock;

static BAYER_16X16_FLAT: OnceLock<[u16; 256]> = OnceLock::new();

pub fn bayer_2x2_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(buffer, mode, &BAYER_2X2_FLAT, 2, 2, DEFAULT_STRENGTH)
}

#[cfg(feature = "rayon")]
pub fn bayer_2x2_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place_par(buffer, mode, &BAYER_2X2_FLAT, 2, 2, DEFAULT_STRENGTH)
}

pub fn bayer_4x4_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(buffer, mode, &BAYER_4X4_FLAT, 4, 4, DEFAULT_STRENGTH)
}

#[cfg(feature = "rayon")]
pub fn bayer_4x4_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place_par(buffer, mode, &BAYER_4X4_FLAT, 4, 4, DEFAULT_STRENGTH)
}

pub fn bayer_8x8_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(buffer, mode, &BAYER_8X8_FLAT, 8, 8, DEFAULT_STRENGTH)
}

#[cfg(feature = "rayon")]
pub fn bayer_8x8_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place_par(buffer, mode, &BAYER_8X8_FLAT, 8, 8, DEFAULT_STRENGTH)
}

pub fn bayer_16x16_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(buffer, mode, bayer_16x16_flat(), 16, 16, DEFAULT_STRENGTH)
}

#[cfg(feature = "rayon")]
pub fn bayer_16x16_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place_par(buffer, mode, bayer_16x16_flat(), 16, 16, DEFAULT_STRENGTH)
}

fn bayer_16x16_flat() -> &'static [u16; 256] {
    BAYER_16X16_FLAT.get_or_init(generate_bayer_16x16_flat)
}
