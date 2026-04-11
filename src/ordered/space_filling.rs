use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{PixelLayout, Sample},
    Buffer, QuantizeMode, Result,
};
use std::sync::OnceLock;

const SPACE_FILLING_SIDE: usize = 16;
const SPACE_FILLING_LEN: usize = SPACE_FILLING_SIDE * SPACE_FILLING_SIDE;

static SPACE_FILLING_16X16_FLAT: OnceLock<[u16; SPACE_FILLING_LEN]> = OnceLock::new();

pub fn space_filling_curve_ordered_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(
        buffer,
        mode,
        space_filling_16x16_flat(),
        SPACE_FILLING_SIDE,
        SPACE_FILLING_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn space_filling_16x16_flat() -> &'static [u16; SPACE_FILLING_LEN] {
    SPACE_FILLING_16X16_FLAT.get_or_init(generate_space_filling_16x16_flat)
}

fn generate_space_filling_16x16_flat() -> [u16; SPACE_FILLING_LEN] {
    let mut out = [0_u16; SPACE_FILLING_LEN];

    for rank in 0..SPACE_FILLING_LEN {
        let (x, y) = hilbert_d2xy(SPACE_FILLING_SIDE, rank);
        out[y * SPACE_FILLING_SIDE + x] = rank as u16;
    }

    out
}

fn hilbert_d2xy(size: usize, mut distance: usize) -> (usize, usize) {
    let mut x = 0_usize;
    let mut y = 0_usize;
    let mut step = 1_usize;

    while step < size {
        let rx = (distance / 2) & 1;
        let ry = (distance ^ rx) & 1;
        hilbert_rotate(step, &mut x, &mut y, rx, ry);
        x += step * rx;
        y += step * ry;
        distance /= 4;
        step *= 2;
    }

    (x, y)
}

fn hilbert_rotate(step: usize, x: &mut usize, y: &mut usize, rx: usize, ry: usize) {
    if ry == 0 {
        if rx == 1 {
            *x = step - 1 - *x;
            *y = step - 1 - *y;
        }
        std::mem::swap(x, y);
    }
}
