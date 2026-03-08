pub(crate) mod core;

use crate::{
    data::{BAYER_2X2, BAYER_4X4},
    Buffer, QuantizeMode,
};

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
const DEFAULT_STRENGTH: i16 = 64;

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
