pub(crate) mod core;

use crate::{Buffer, QuantizeMode};

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
