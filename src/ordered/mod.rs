pub(crate) mod core;
mod yliluoma;

use crate::{
    data::{
        generate_bayer_16x16_flat, BAYER_2X2_FLAT, BAYER_4X4_FLAT, BAYER_8X8_FLAT,
        CLUSTER_DOT_4X4_FLAT, CLUSTER_DOT_8X8_FLAT,
    },
    Buffer, DithrResult, Palette, QuantizeMode,
};
use std::sync::OnceLock;

static BAYER_16X16_FLAT: OnceLock<[u8; 256]> = OnceLock::new();
const DEFAULT_STRENGTH: i16 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedError {
    EmptyMap,
    InvalidDimensions,
    ValueOutOfRange,
}

#[doc(hidden)]
pub(crate) fn ordered_dither_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> DithrResult<()> {
    core::ordered_dither_in_place(buffer, mode, map, map_w, map_h, strength)
}

pub fn bayer_2x2_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    ordered_dither_in_place(buffer, mode, &BAYER_2X2_FLAT, 2, 2, DEFAULT_STRENGTH)
}

pub fn bayer_4x4_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    ordered_dither_in_place(buffer, mode, &BAYER_4X4_FLAT, 4, 4, DEFAULT_STRENGTH)
}

pub fn bayer_8x8_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    ordered_dither_in_place(buffer, mode, &BAYER_8X8_FLAT, 8, 8, DEFAULT_STRENGTH)
}

pub fn bayer_16x16_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    ordered_dither_in_place(buffer, mode, bayer_16x16_flat(), 16, 16, DEFAULT_STRENGTH)
}

pub fn cluster_dot_4x4_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
) -> DithrResult<()> {
    ordered_dither_in_place(buffer, mode, &CLUSTER_DOT_4X4_FLAT, 4, 4, DEFAULT_STRENGTH)
}

pub fn cluster_dot_8x8_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
) -> DithrResult<()> {
    ordered_dither_in_place(buffer, mode, &CLUSTER_DOT_8X8_FLAT, 8, 8, DEFAULT_STRENGTH)
}

pub fn custom_ordered_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> DithrResult<()> {
    if map.is_empty() {
        return Err(OrderedError::EmptyMap.into());
    }

    let expected_len = map_w
        .checked_mul(map_h)
        .ok_or(OrderedError::InvalidDimensions)?;
    if expected_len == 0 || expected_len != map.len() {
        return Err(OrderedError::InvalidDimensions.into());
    }

    if map.iter().any(|&value| usize::from(value) >= map.len()) {
        return Err(OrderedError::ValueOutOfRange.into());
    }

    ordered_dither_in_place(buffer, mode, map, map_w, map_h, strength)
}

pub fn yliluoma_1_in_place(buffer: &mut Buffer<'_>, palette: &Palette) {
    yliluoma::yliluoma_1_in_place(buffer, palette);
}

pub fn yliluoma_2_in_place(buffer: &mut Buffer<'_>, palette: &Palette) {
    yliluoma::yliluoma_2_in_place(buffer, palette);
}

pub fn yliluoma_3_in_place(buffer: &mut Buffer<'_>, palette: &Palette) {
    yliluoma::yliluoma_3_in_place(buffer, palette);
}

fn bayer_16x16_flat() -> &'static [u8; 256] {
    BAYER_16X16_FLAT.get_or_init(generate_bayer_16x16_flat)
}
