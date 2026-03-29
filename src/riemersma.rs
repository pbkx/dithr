use crate::{
    core::{PixelLayout, Sample},
    quantize_pixel, Buffer, Error, PixelFormat, QuantizeMode, Result,
};
use std::mem::size_of;

const HISTORY_LEN: usize = 16;
const HISTORY_WEIGHTS_I32: [i32; HISTORY_LEN] =
    [1, 1, 2, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 23, 27];
const HISTORY_WEIGHTS: [f32; HISTORY_LEN] = [
    1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 19.0, 23.0, 27.0,
];
const HISTORY_WEIGHT_SUM_I32: i32 = 153;
const HISTORY_WEIGHT_SUM: f32 = 153.0;

pub fn riemersma_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
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

    let mut history = [[0.0_f32; 3]; HISTORY_LEN];
    let mut head = 0_usize;
    let mut filled = 0_usize;
    let mut visited = 0_usize;
    let max_value = if S::IS_FLOAT {
        0
    } else {
        integer_sample_max::<S>()?
    };

    for d in 0..total_steps {
        if visited == pixel_count {
            break;
        }

        let (x, y) = hilbert_d2xy(side, d)?;
        if x >= width || y >= height {
            continue;
        }

        let offset = buffer.try_pixel_offset(x, y)?;
        let new_error = if S::IS_FLOAT {
            let weighted = weighted_error_float(&history, head, filled);
            match format {
                PixelFormat::Gray8 | PixelFormat::Gray16 => {
                    let adjusted =
                        (buffer.data[offset].to_unit_f32() + weighted[0]).clamp(0.0, 1.0);
                    let quantized = quantize_pixel::<S, crate::core::Gray>(
                        &[S::from_unit_f32(adjusted)],
                        mode,
                    )?;
                    let quantized_gray = quantized[0];
                    buffer.data[offset] = quantized_gray;
                    [adjusted - quantized_gray.to_unit_f32(), 0.0, 0.0]
                }
                PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::Rgb32F => {
                    let adjusted = [
                        (buffer.data[offset].to_unit_f32() + weighted[0]).clamp(0.0, 1.0),
                        (buffer.data[offset + 1].to_unit_f32() + weighted[1]).clamp(0.0, 1.0),
                        (buffer.data[offset + 2].to_unit_f32() + weighted[2]).clamp(0.0, 1.0),
                    ];
                    let pixel = [
                        S::from_unit_f32(adjusted[0]),
                        S::from_unit_f32(adjusted[1]),
                        S::from_unit_f32(adjusted[2]),
                    ];
                    let quantized = quantize_pixel::<S, crate::core::Rgb>(&pixel, mode)?;
                    buffer.data[offset] = quantized[0];
                    buffer.data[offset + 1] = quantized[1];
                    buffer.data[offset + 2] = quantized[2];
                    [
                        adjusted[0] - quantized[0].to_unit_f32(),
                        adjusted[1] - quantized[1].to_unit_f32(),
                        adjusted[2] - quantized[2].to_unit_f32(),
                    ]
                }
                PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::Rgba32F => {
                    let alpha = buffer.data[offset + 3];
                    let adjusted = [
                        (buffer.data[offset].to_unit_f32() + weighted[0]).clamp(0.0, 1.0),
                        (buffer.data[offset + 1].to_unit_f32() + weighted[1]).clamp(0.0, 1.0),
                        (buffer.data[offset + 2].to_unit_f32() + weighted[2]).clamp(0.0, 1.0),
                    ];
                    let pixel = [
                        S::from_unit_f32(adjusted[0]),
                        S::from_unit_f32(adjusted[1]),
                        S::from_unit_f32(adjusted[2]),
                        alpha,
                    ];
                    let quantized = quantize_pixel::<S, crate::core::Rgba>(&pixel, mode)?;
                    buffer.data[offset] = quantized[0];
                    buffer.data[offset + 1] = quantized[1];
                    buffer.data[offset + 2] = quantized[2];
                    buffer.data[offset + 3] = alpha;
                    [
                        adjusted[0] - quantized[0].to_unit_f32(),
                        adjusted[1] - quantized[1].to_unit_f32(),
                        adjusted[2] - quantized[2].to_unit_f32(),
                    ]
                }
                _ => {
                    return Err(Error::UnsupportedFormat(
                        "riemersma supports Gray, Rgb, and Rgba formats only",
                    ));
                }
            }
        } else {
            let weighted = weighted_error_int(&history, head, filled, max_value);
            match format {
                PixelFormat::Gray8 | PixelFormat::Gray16 => {
                    let adjusted = (sample_to_domain(buffer.data[offset], max_value) + weighted[0])
                        .clamp(0, max_value);
                    let quantized = quantize_pixel::<S, crate::core::Gray>(
                        &[domain_to_sample::<S>(adjusted, max_value)],
                        mode,
                    )?;
                    let quantized_gray = quantized[0];
                    buffer.data[offset] = quantized_gray;
                    [
                        domain_to_unit(
                            adjusted - sample_to_domain(quantized_gray, max_value),
                            max_value,
                        ),
                        0.0,
                        0.0,
                    ]
                }
                PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::Rgb32F => {
                    let adjusted = [
                        (sample_to_domain(buffer.data[offset], max_value) + weighted[0])
                            .clamp(0, max_value),
                        (sample_to_domain(buffer.data[offset + 1], max_value) + weighted[1])
                            .clamp(0, max_value),
                        (sample_to_domain(buffer.data[offset + 2], max_value) + weighted[2])
                            .clamp(0, max_value),
                    ];
                    let pixel = [
                        domain_to_sample::<S>(adjusted[0], max_value),
                        domain_to_sample::<S>(adjusted[1], max_value),
                        domain_to_sample::<S>(adjusted[2], max_value),
                    ];
                    let quantized = quantize_pixel::<S, crate::core::Rgb>(&pixel, mode)?;
                    buffer.data[offset] = quantized[0];
                    buffer.data[offset + 1] = quantized[1];
                    buffer.data[offset + 2] = quantized[2];
                    [
                        domain_to_unit(
                            adjusted[0] - sample_to_domain(quantized[0], max_value),
                            max_value,
                        ),
                        domain_to_unit(
                            adjusted[1] - sample_to_domain(quantized[1], max_value),
                            max_value,
                        ),
                        domain_to_unit(
                            adjusted[2] - sample_to_domain(quantized[2], max_value),
                            max_value,
                        ),
                    ]
                }
                PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::Rgba32F => {
                    let alpha = buffer.data[offset + 3];
                    let adjusted = [
                        (sample_to_domain(buffer.data[offset], max_value) + weighted[0])
                            .clamp(0, max_value),
                        (sample_to_domain(buffer.data[offset + 1], max_value) + weighted[1])
                            .clamp(0, max_value),
                        (sample_to_domain(buffer.data[offset + 2], max_value) + weighted[2])
                            .clamp(0, max_value),
                    ];
                    let pixel = [
                        domain_to_sample::<S>(adjusted[0], max_value),
                        domain_to_sample::<S>(adjusted[1], max_value),
                        domain_to_sample::<S>(adjusted[2], max_value),
                        alpha,
                    ];
                    let quantized = quantize_pixel::<S, crate::core::Rgba>(&pixel, mode)?;
                    buffer.data[offset] = quantized[0];
                    buffer.data[offset + 1] = quantized[1];
                    buffer.data[offset + 2] = quantized[2];
                    buffer.data[offset + 3] = alpha;
                    [
                        domain_to_unit(
                            adjusted[0] - sample_to_domain(quantized[0], max_value),
                            max_value,
                        ),
                        domain_to_unit(
                            adjusted[1] - sample_to_domain(quantized[1], max_value),
                            max_value,
                        ),
                        domain_to_unit(
                            adjusted[2] - sample_to_domain(quantized[2], max_value),
                            max_value,
                        ),
                    ]
                }
                _ => {
                    return Err(Error::UnsupportedFormat(
                        "riemersma supports Gray, Rgb, and Rgba formats only",
                    ));
                }
            }
        };

        push_error(&mut history, &mut head, &mut filled, new_error);
        visited += 1;
    }

    Ok(())
}

fn weighted_error_float(history: &[[f32; 3]; HISTORY_LEN], head: usize, filled: usize) -> [f32; 3] {
    if filled == 0 {
        return [0.0, 0.0, 0.0];
    }

    let start_weight = HISTORY_LEN - filled;
    let mut accum = [0.0_f32; 3];

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

fn weighted_error_int(
    history: &[[f32; 3]; HISTORY_LEN],
    head: usize,
    filled: usize,
    max_value: i32,
) -> [i32; 3] {
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
        let weight = HISTORY_WEIGHTS_I32[start_weight + i];
        accum[0] += unit_to_domain(history[slot][0], max_value) * weight;
        accum[1] += unit_to_domain(history[slot][1], max_value) * weight;
        accum[2] += unit_to_domain(history[slot][2], max_value) * weight;
    }

    [
        accum[0] / HISTORY_WEIGHT_SUM_I32,
        accum[1] / HISTORY_WEIGHT_SUM_I32,
        accum[2] / HISTORY_WEIGHT_SUM_I32,
    ]
}

fn push_error(
    history: &mut [[f32; 3]; HISTORY_LEN],
    head: &mut usize,
    filled: &mut usize,
    value: [f32; 3],
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

fn integer_sample_max<S: Sample>() -> Result<i32> {
    if S::IS_FLOAT {
        return Err(Error::UnsupportedFormat(
            "riemersma integer path does not support floating-point samples",
        ));
    }

    match size_of::<S>() {
        1 => Ok(255),
        2 => Ok(65_535),
        _ => Err(Error::UnsupportedFormat(
            "unsupported integer sample width for riemersma",
        )),
    }
}

fn sample_to_domain<S: Sample>(value: S, max_value: i32) -> i32 {
    (value.to_unit_f32().clamp(0.0, 1.0) * max_value as f32).round() as i32
}

fn domain_to_sample<S: Sample>(value: i32, max_value: i32) -> S {
    S::from_unit_f32(value.clamp(0, max_value) as f32 / max_value as f32)
}

fn unit_to_domain(unit: f32, max_value: i32) -> i32 {
    (unit * max_value as f32).round() as i32
}

fn domain_to_unit(value: i32, max_value: i32) -> f32 {
    value as f32 / max_value as f32
}
