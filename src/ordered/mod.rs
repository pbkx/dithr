pub(crate) mod core;

use crate::{data::BAYER_2X2, Buffer, QuantizeMode};

const BAYER_2X2_FLAT: [u8; 4] = [
    BAYER_2X2[0][0],
    BAYER_2X2[0][1],
    BAYER_2X2[1][0],
    BAYER_2X2[1][1],
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
