#[cfg(feature = "rayon")]
use super::ordered_dither_in_place_par;
use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{PixelLayout, Sample},
    data::{CLUSTER_DOT_4X4_FLAT, CLUSTER_DOT_8X8_FLAT},
    Buffer, QuantizeMode, Result,
};

pub fn cluster_dot_4x4_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(buffer, mode, &CLUSTER_DOT_4X4_FLAT, 4, 4, DEFAULT_STRENGTH)
}

#[cfg(feature = "rayon")]
pub fn cluster_dot_4x4_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place_par(buffer, mode, &CLUSTER_DOT_4X4_FLAT, 4, 4, DEFAULT_STRENGTH)
}

pub fn cluster_dot_8x8_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(buffer, mode, &CLUSTER_DOT_8X8_FLAT, 8, 8, DEFAULT_STRENGTH)
}

#[cfg(feature = "rayon")]
pub fn cluster_dot_8x8_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place_par(buffer, mode, &CLUSTER_DOT_8X8_FLAT, 8, 8, DEFAULT_STRENGTH)
}
