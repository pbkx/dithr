use crate::{
    math::{
        color::luma_u8,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, Error, PixelFormat, QuantizeMode, Result,
};

const HISTORY_LEN: usize = 16;
const HISTORY_WEIGHTS: [i32; HISTORY_LEN] = [1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 23, 27];
const HISTORY_WEIGHT_SUM: i32 = 153;

pub fn riemersma_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    buffer.validate()?;

    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(Error::InvalidArgument("image dimensions overflow"))?;
    let side = hilbert_side(width.max(height))?;
    let total_steps = side
        .checked_mul(side)
        .ok_or(Error::InvalidArgument("hilbert size overflow"))?;

    let mut history = [[0_i32; 3]; HISTORY_LEN];
    let mut head = 0_usize;
    let mut filled = 0_usize;
    let mut visited = 0_usize;

    for d in 0..total_steps {
        if visited == pixel_count {
            break;
        }

        let (x, y) = hilbert_d2xy(side, d)?;
        if x >= width || y >= height {
            continue;
        }

        let offset = buffer.try_pixel_offset(x, y)?;
        let weighted = weighted_error(&history, head, filled);
        let new_error = match format {
            PixelFormat::Gray8 => {
                let adjusted = clamp_u8(i32::from(buffer.data[offset]) + weighted[0]);
                let quantized = quantize_pixel(PixelFormat::Gray8, &[adjusted], mode)?;
                let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
                buffer.data[offset] = quantized_gray;
                [
                    i32::from(clamp_i16(i32::from(adjusted) - i32::from(quantized_gray))),
                    0,
                    0,
                ]
            }
            PixelFormat::Rgb8 => {
                let adjusted = [
                    clamp_u8(i32::from(buffer.data[offset]) + weighted[0]),
                    clamp_u8(i32::from(buffer.data[offset + 1]) + weighted[1]),
                    clamp_u8(i32::from(buffer.data[offset + 2]) + weighted[2]),
                ];
                let quantized = quantize_pixel(PixelFormat::Rgb8, &adjusted, mode)?;
                buffer.data[offset] = quantized[0];
                buffer.data[offset + 1] = quantized[1];
                buffer.data[offset + 2] = quantized[2];
                [
                    i32::from(clamp_i16(i32::from(adjusted[0]) - i32::from(quantized[0]))),
                    i32::from(clamp_i16(i32::from(adjusted[1]) - i32::from(quantized[1]))),
                    i32::from(clamp_i16(i32::from(adjusted[2]) - i32::from(quantized[2]))),
                ]
            }
            PixelFormat::Rgba8 => {
                let alpha = buffer.data[offset + 3];
                let adjusted = [
                    clamp_u8(i32::from(buffer.data[offset]) + weighted[0]),
                    clamp_u8(i32::from(buffer.data[offset + 1]) + weighted[1]),
                    clamp_u8(i32::from(buffer.data[offset + 2]) + weighted[2]),
                    alpha,
                ];
                let quantized = quantize_pixel(PixelFormat::Rgba8, &adjusted, mode)?;
                buffer.data[offset] = quantized[0];
                buffer.data[offset + 1] = quantized[1];
                buffer.data[offset + 2] = quantized[2];
                buffer.data[offset + 3] = alpha;
                [
                    i32::from(clamp_i16(i32::from(adjusted[0]) - i32::from(quantized[0]))),
                    i32::from(clamp_i16(i32::from(adjusted[1]) - i32::from(quantized[1]))),
                    i32::from(clamp_i16(i32::from(adjusted[2]) - i32::from(quantized[2]))),
                ]
            }
        };

        push_error(&mut history, &mut head, &mut filled, new_error);
        visited += 1;
    }

    Ok(())
}

fn weighted_error(history: &[[i32; 3]; HISTORY_LEN], head: usize, filled: usize) -> [i32; 3] {
    if filled == 0 {
        return [0, 0, 0];
    }

    let start_weight = HISTORY_LEN - filled;
    let mut accum = [0_i32; 3];

    for i in 0..filled {
        let slot = if filled == HISTORY_LEN {
            (head + i) % HISTORY_LEN
        } else {
            i
        };
        let weight = HISTORY_WEIGHTS[start_weight + i];
        accum[0] += history[slot][0] * weight;
        accum[1] += history[slot][1] * weight;
        accum[2] += history[slot][2] * weight;
    }

    [
        accum[0] / HISTORY_WEIGHT_SUM,
        accum[1] / HISTORY_WEIGHT_SUM,
        accum[2] / HISTORY_WEIGHT_SUM,
    ]
}

fn push_error(
    history: &mut [[i32; 3]; HISTORY_LEN],
    head: &mut usize,
    filled: &mut usize,
    value: [i32; 3],
) {
    if *filled < HISTORY_LEN {
        history[*filled] = value;
        *filled += 1;
    } else {
        history[*head] = value;
        *head = (*head + 1) % HISTORY_LEN;
    }
}

fn hilbert_side(max_dimension: usize) -> Result<usize> {
    let mut side = 1_usize;
    while side < max_dimension {
        side = side
            .checked_shl(1)
            .ok_or(Error::InvalidArgument("hilbert side overflow"))?;
    }
    Ok(side)
}

fn hilbert_d2xy(side: usize, distance: usize) -> Result<(usize, usize)> {
    let mut d = distance;
    let mut x = 0_usize;
    let mut y = 0_usize;
    let mut s = 1_usize;

    while s < side {
        let rx = (d / 2) & 1;
        let ry = (d ^ rx) & 1;
        let (nx, ny) = hilbert_rotate(s, x, y, rx, ry);
        x = nx + s * rx;
        y = ny + s * ry;
        d /= 4;
        s = s
            .checked_shl(1)
            .ok_or(Error::InvalidArgument("hilbert step overflow"))?;
    }

    Ok((x, y))
}

fn hilbert_rotate(side: usize, x: usize, y: usize, rx: usize, ry: usize) -> (usize, usize) {
    if ry == 0 {
        if rx == 1 {
            let nx = side - 1 - x;
            let ny = side - 1 - y;
            return (ny, nx);
        }
        return (y, x);
    }

    (x, y)
}
