use crate::{
    data::BAYER_8X8,
    math::color::{luma_u8, rgb_distance_sq},
    Buffer, Palette, PixelFormat,
};
use std::collections::HashMap;

const DITHER_LEVELS: u8 = 64;

#[derive(Clone, Copy)]
struct MixingPlan {
    first_index: usize,
    second_index: usize,
    ratio: u8,
}

pub(crate) fn yliluoma_1_in_place(buffer: &mut Buffer<'_>, palette: &Palette) {
    buffer
        .validate()
        .expect("buffer must be valid for yliluoma algorithm 1");

    let colors = palette.as_slice();
    assert!(!colors.is_empty(), "palette must not be empty");

    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let mut cache = HashMap::<u32, MixingPlan>::new();

    for y in 0..height {
        let row = buffer.row_mut(y);

        for x in 0..width {
            let offset = x.checked_mul(bpp).expect("pixel offset overflow in row");
            let source_rgb = match format {
                PixelFormat::Gray8 => {
                    let g = row[offset];
                    [g, g, g]
                }
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    [row[offset], row[offset + 1], row[offset + 2]]
                }
            };
            let key = pack_rgb(source_rgb);
            let plan = if let Some(&cached) = cache.get(&key) {
                cached
            } else {
                let computed = devise_best_mixing_plan(source_rgb, colors);
                cache.insert(key, computed);
                computed
            };
            let threshold = BAYER_8X8[y % 8][x % 8];
            let chosen = if threshold < plan.ratio {
                colors[plan.second_index]
            } else {
                colors[plan.first_index]
            };

            match format {
                PixelFormat::Gray8 => {
                    row[offset] = luma_u8(chosen);
                }
                PixelFormat::Rgb8 => {
                    row[offset] = chosen[0];
                    row[offset + 1] = chosen[1];
                    row[offset + 2] = chosen[2];
                }
                PixelFormat::Rgba8 => {
                    let alpha = row[offset + 3];
                    row[offset] = chosen[0];
                    row[offset + 1] = chosen[1];
                    row[offset + 2] = chosen[2];
                    row[offset + 3] = alpha;
                }
            }
        }
    }
}

fn devise_best_mixing_plan(target: [u8; 3], palette: &[[u8; 3]]) -> MixingPlan {
    let mut best_plan = MixingPlan {
        first_index: 0,
        second_index: 0,
        ratio: 0,
    };
    let mut best_error = u32::MAX;

    for (first_index, &first) in palette.iter().enumerate() {
        for (second_index, &second) in palette.iter().enumerate() {
            for ratio in 0..DITHER_LEVELS {
                let mixed = interpolate_rgb(first, second, ratio, DITHER_LEVELS);
                let error = rgb_distance_sq(target, mixed);
                if error < best_error {
                    best_error = error;
                    best_plan = MixingPlan {
                        first_index,
                        second_index,
                        ratio,
                    };
                }
            }
        }
    }

    best_plan
}

fn interpolate_rgb(a: [u8; 3], b: [u8; 3], ratio: u8, levels: u8) -> [u8; 3] {
    [
        interpolate_channel(a[0], b[0], ratio, levels),
        interpolate_channel(a[1], b[1], ratio, levels),
        interpolate_channel(a[2], b[2], ratio, levels),
    ]
}

fn interpolate_channel(a: u8, b: u8, ratio: u8, levels: u8) -> u8 {
    let a_i = i32::from(a);
    let delta = i32::from(b) - a_i;
    let mixed = a_i + (i32::from(ratio) * delta) / i32::from(levels);

    mixed as u8
}

fn pack_rgb(rgb: [u8; 3]) -> u32 {
    (u32::from(rgb[0]) << 16) | (u32::from(rgb[1]) << 8) | u32::from(rgb[2])
}
