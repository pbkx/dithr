pub(crate) mod core;

use crate::{
    data::{
        generate_bayer_16x16, BAYER_2X2, BAYER_4X4, BAYER_8X8, CLUSTER_DOT_4X4, CLUSTER_DOT_8X8,
    },
    Buffer, QuantizeMode,
};
use std::sync::OnceLock;

const BAYER_2X2_FLAT: [u8; 4] = [
    BAYER_2X2[0][0],
    BAYER_2X2[0][1],
    BAYER_2X2[1][0],
    BAYER_2X2[1][1],
];
const BAYER_4X4_FLAT: [u8; 16] = [
    BAYER_4X4[0][0],
    BAYER_4X4[0][1],
    BAYER_4X4[0][2],
    BAYER_4X4[0][3],
    BAYER_4X4[1][0],
    BAYER_4X4[1][1],
    BAYER_4X4[1][2],
    BAYER_4X4[1][3],
    BAYER_4X4[2][0],
    BAYER_4X4[2][1],
    BAYER_4X4[2][2],
    BAYER_4X4[2][3],
    BAYER_4X4[3][0],
    BAYER_4X4[3][1],
    BAYER_4X4[3][2],
    BAYER_4X4[3][3],
];
const BAYER_8X8_FLAT: [u8; 64] = [
    BAYER_8X8[0][0],
    BAYER_8X8[0][1],
    BAYER_8X8[0][2],
    BAYER_8X8[0][3],
    BAYER_8X8[0][4],
    BAYER_8X8[0][5],
    BAYER_8X8[0][6],
    BAYER_8X8[0][7],
    BAYER_8X8[1][0],
    BAYER_8X8[1][1],
    BAYER_8X8[1][2],
    BAYER_8X8[1][3],
    BAYER_8X8[1][4],
    BAYER_8X8[1][5],
    BAYER_8X8[1][6],
    BAYER_8X8[1][7],
    BAYER_8X8[2][0],
    BAYER_8X8[2][1],
    BAYER_8X8[2][2],
    BAYER_8X8[2][3],
    BAYER_8X8[2][4],
    BAYER_8X8[2][5],
    BAYER_8X8[2][6],
    BAYER_8X8[2][7],
    BAYER_8X8[3][0],
    BAYER_8X8[3][1],
    BAYER_8X8[3][2],
    BAYER_8X8[3][3],
    BAYER_8X8[3][4],
    BAYER_8X8[3][5],
    BAYER_8X8[3][6],
    BAYER_8X8[3][7],
    BAYER_8X8[4][0],
    BAYER_8X8[4][1],
    BAYER_8X8[4][2],
    BAYER_8X8[4][3],
    BAYER_8X8[4][4],
    BAYER_8X8[4][5],
    BAYER_8X8[4][6],
    BAYER_8X8[4][7],
    BAYER_8X8[5][0],
    BAYER_8X8[5][1],
    BAYER_8X8[5][2],
    BAYER_8X8[5][3],
    BAYER_8X8[5][4],
    BAYER_8X8[5][5],
    BAYER_8X8[5][6],
    BAYER_8X8[5][7],
    BAYER_8X8[6][0],
    BAYER_8X8[6][1],
    BAYER_8X8[6][2],
    BAYER_8X8[6][3],
    BAYER_8X8[6][4],
    BAYER_8X8[6][5],
    BAYER_8X8[6][6],
    BAYER_8X8[6][7],
    BAYER_8X8[7][0],
    BAYER_8X8[7][1],
    BAYER_8X8[7][2],
    BAYER_8X8[7][3],
    BAYER_8X8[7][4],
    BAYER_8X8[7][5],
    BAYER_8X8[7][6],
    BAYER_8X8[7][7],
];
const CLUSTER_DOT_4X4_FLAT: [u8; 16] = [
    CLUSTER_DOT_4X4[0][0],
    CLUSTER_DOT_4X4[0][1],
    CLUSTER_DOT_4X4[0][2],
    CLUSTER_DOT_4X4[0][3],
    CLUSTER_DOT_4X4[1][0],
    CLUSTER_DOT_4X4[1][1],
    CLUSTER_DOT_4X4[1][2],
    CLUSTER_DOT_4X4[1][3],
    CLUSTER_DOT_4X4[2][0],
    CLUSTER_DOT_4X4[2][1],
    CLUSTER_DOT_4X4[2][2],
    CLUSTER_DOT_4X4[2][3],
    CLUSTER_DOT_4X4[3][0],
    CLUSTER_DOT_4X4[3][1],
    CLUSTER_DOT_4X4[3][2],
    CLUSTER_DOT_4X4[3][3],
];
const CLUSTER_DOT_8X8_FLAT: [u8; 64] = [
    CLUSTER_DOT_8X8[0][0],
    CLUSTER_DOT_8X8[0][1],
    CLUSTER_DOT_8X8[0][2],
    CLUSTER_DOT_8X8[0][3],
    CLUSTER_DOT_8X8[0][4],
    CLUSTER_DOT_8X8[0][5],
    CLUSTER_DOT_8X8[0][6],
    CLUSTER_DOT_8X8[0][7],
    CLUSTER_DOT_8X8[1][0],
    CLUSTER_DOT_8X8[1][1],
    CLUSTER_DOT_8X8[1][2],
    CLUSTER_DOT_8X8[1][3],
    CLUSTER_DOT_8X8[1][4],
    CLUSTER_DOT_8X8[1][5],
    CLUSTER_DOT_8X8[1][6],
    CLUSTER_DOT_8X8[1][7],
    CLUSTER_DOT_8X8[2][0],
    CLUSTER_DOT_8X8[2][1],
    CLUSTER_DOT_8X8[2][2],
    CLUSTER_DOT_8X8[2][3],
    CLUSTER_DOT_8X8[2][4],
    CLUSTER_DOT_8X8[2][5],
    CLUSTER_DOT_8X8[2][6],
    CLUSTER_DOT_8X8[2][7],
    CLUSTER_DOT_8X8[3][0],
    CLUSTER_DOT_8X8[3][1],
    CLUSTER_DOT_8X8[3][2],
    CLUSTER_DOT_8X8[3][3],
    CLUSTER_DOT_8X8[3][4],
    CLUSTER_DOT_8X8[3][5],
    CLUSTER_DOT_8X8[3][6],
    CLUSTER_DOT_8X8[3][7],
    CLUSTER_DOT_8X8[4][0],
    CLUSTER_DOT_8X8[4][1],
    CLUSTER_DOT_8X8[4][2],
    CLUSTER_DOT_8X8[4][3],
    CLUSTER_DOT_8X8[4][4],
    CLUSTER_DOT_8X8[4][5],
    CLUSTER_DOT_8X8[4][6],
    CLUSTER_DOT_8X8[4][7],
    CLUSTER_DOT_8X8[5][0],
    CLUSTER_DOT_8X8[5][1],
    CLUSTER_DOT_8X8[5][2],
    CLUSTER_DOT_8X8[5][3],
    CLUSTER_DOT_8X8[5][4],
    CLUSTER_DOT_8X8[5][5],
    CLUSTER_DOT_8X8[5][6],
    CLUSTER_DOT_8X8[5][7],
    CLUSTER_DOT_8X8[6][0],
    CLUSTER_DOT_8X8[6][1],
    CLUSTER_DOT_8X8[6][2],
    CLUSTER_DOT_8X8[6][3],
    CLUSTER_DOT_8X8[6][4],
    CLUSTER_DOT_8X8[6][5],
    CLUSTER_DOT_8X8[6][6],
    CLUSTER_DOT_8X8[6][7],
    CLUSTER_DOT_8X8[7][0],
    CLUSTER_DOT_8X8[7][1],
    CLUSTER_DOT_8X8[7][2],
    CLUSTER_DOT_8X8[7][3],
    CLUSTER_DOT_8X8[7][4],
    CLUSTER_DOT_8X8[7][5],
    CLUSTER_DOT_8X8[7][6],
    CLUSTER_DOT_8X8[7][7],
];
static BAYER_16X16_FLAT: OnceLock<[u8; 256]> = OnceLock::new();
const DEFAULT_STRENGTH: i16 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderedError {
    EmptyMap,
    InvalidDimensions,
    ValueOutOfRange,
}

#[doc(hidden)]
pub fn ordered_dither_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) {
    core::ordered_dither_in_place(buffer, mode, map, map_w, map_h, strength);
}

#[must_use]
#[doc(hidden)]
pub fn ordered_threshold_for_xy(x: usize, y: usize, map: &[u8], map_w: usize, map_h: usize) -> u8 {
    core::ordered_threshold_for_xy(x, y, map, map_w, map_h)
}

pub fn bayer_2x2_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    ordered_dither_in_place(buffer, mode, &BAYER_2X2_FLAT, 2, 2, DEFAULT_STRENGTH);
}

pub fn bayer_4x4_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    ordered_dither_in_place(buffer, mode, &BAYER_4X4_FLAT, 4, 4, DEFAULT_STRENGTH);
}

pub fn bayer_8x8_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    ordered_dither_in_place(buffer, mode, &BAYER_8X8_FLAT, 8, 8, DEFAULT_STRENGTH);
}

pub fn bayer_16x16_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    ordered_dither_in_place(buffer, mode, bayer_16x16_flat(), 16, 16, DEFAULT_STRENGTH);
}

pub fn cluster_dot_4x4_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    ordered_dither_in_place(buffer, mode, &CLUSTER_DOT_4X4_FLAT, 4, 4, DEFAULT_STRENGTH);
}

pub fn cluster_dot_8x8_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    ordered_dither_in_place(buffer, mode, &CLUSTER_DOT_8X8_FLAT, 8, 8, DEFAULT_STRENGTH);
}

pub fn custom_ordered_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    map: &[u8],
    map_w: usize,
    map_h: usize,
    strength: i16,
) -> Result<(), OrderedError> {
    if map.is_empty() {
        return Err(OrderedError::EmptyMap);
    }

    let expected_len = map_w
        .checked_mul(map_h)
        .ok_or(OrderedError::InvalidDimensions)?;
    if expected_len == 0 || expected_len != map.len() {
        return Err(OrderedError::InvalidDimensions);
    }

    if map.iter().any(|&value| usize::from(value) >= map.len()) {
        return Err(OrderedError::ValueOutOfRange);
    }

    ordered_dither_in_place(buffer, mode, map, map_w, map_h, strength);
    Ok(())
}

fn bayer_16x16_flat() -> &'static [u8; 256] {
    BAYER_16X16_FLAT.get_or_init(|| {
        let map = generate_bayer_16x16();
        let mut flat = [0_u8; 256];

        for y in 0..16 {
            for x in 0..16 {
                flat[y * 16 + x] = map[y][x];
            }
        }

        flat
    })
}
