use crate::{
    data::BAYER_8X8,
    math::color::{luma_u8, rgb_distance_sq},
    Buffer, BufferError, Palette, PixelFormat, Result,
};
use std::collections::HashMap;

const DITHER_LEVELS: u8 = 64;
const GAMMA: f64 = 2.2;
const INV_GAMMA: f64 = 1.0 / GAMMA;

#[derive(Clone, Copy)]
struct MixingPlan {
    first_index: usize,
    second_index: usize,
    ratio: u8,
}

struct PaletteView<'a> {
    colors: &'a [[u8; 3]],
    luma: Vec<u32>,
    packed: Vec<u32>,
}

impl<'a> PaletteView<'a> {
    fn new(palette: &'a Palette) -> Self {
        let colors = palette.as_slice();
        let mut luma = Vec::with_capacity(colors.len());
        let mut packed = Vec::with_capacity(colors.len());

        for &color in colors {
            luma.push(
                u32::from(color[0]) * 299 + u32::from(color[1]) * 587 + u32::from(color[2]) * 114,
            );
            packed.push(pack_rgb(color));
        }

        Self {
            colors,
            luma,
            packed,
        }
    }

    fn gamma_table(&self) -> Vec<[f64; 3]> {
        self.colors
            .iter()
            .map(|color| {
                [
                    gamma_correct_channel(color[0]),
                    gamma_correct_channel(color[1]),
                    gamma_correct_channel(color[2]),
                ]
            })
            .collect()
    }
}

pub(crate) fn yliluoma_1_in_place(buffer: &mut Buffer<'_>, palette: &Palette) -> Result<()> {
    buffer.validate()?;

    let view = PaletteView::new(palette);
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let mut cache = HashMap::<u32, MixingPlan>::new();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let offset = x.checked_mul(bpp).ok_or(BufferError::OutOfBounds)?;
            let source_rgb = match format {
                PixelFormat::Gray8 => {
                    let g = row[offset];
                    [g, g, g]
                }
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    [row[offset], row[offset + 1], row[offset + 2]]
                }
                _ => {
                    return Err(crate::Error::UnsupportedFormat(
                        "yliluoma ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            };
            let key = pack_rgb(source_rgb);
            let plan = if let Some(&cached) = cache.get(&key) {
                cached
            } else {
                let computed = devise_best_mixing_plan(source_rgb, view.colors);
                cache.insert(key, computed);
                computed
            };
            let threshold = BAYER_8X8[y % 8][x % 8];
            let chosen = if threshold < plan.ratio {
                view.colors[plan.second_index]
            } else {
                view.colors[plan.first_index]
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
                _ => {
                    return Err(crate::Error::UnsupportedFormat(
                        "yliluoma ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn yliluoma_2_in_place(buffer: &mut Buffer<'_>, palette: &Palette) -> Result<()> {
    buffer.validate()?;

    let view = PaletteView::new(palette);
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let mut cache = HashMap::<u32, Vec<usize>>::new();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let offset = x.checked_mul(bpp).ok_or(BufferError::OutOfBounds)?;
            let source_rgb = match format {
                PixelFormat::Gray8 => {
                    let g = row[offset];
                    [g, g, g]
                }
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    [row[offset], row[offset + 1], row[offset + 2]]
                }
                _ => {
                    return Err(crate::Error::UnsupportedFormat(
                        "yliluoma ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            };
            let key = pack_rgb(source_rgb);
            let selected = {
                let plan = cache
                    .entry(key)
                    .or_insert_with(|| devise_color_sequence(source_rgb, view.colors, &view.luma));
                let map_value = usize::from(BAYER_8X8[y % 8][x % 8]);
                let plan_index = map_value
                    .checked_mul(plan.len())
                    .ok_or(BufferError::OutOfBounds)?
                    / usize::from(DITHER_LEVELS);

                plan[plan_index]
            };
            let chosen = view.colors[selected];

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
                _ => {
                    return Err(crate::Error::UnsupportedFormat(
                        "yliluoma ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            }
        }
    }

    Ok(())
}

pub(crate) fn yliluoma_3_in_place(buffer: &mut Buffer<'_>, palette: &Palette) -> Result<()> {
    buffer.validate()?;

    let view = PaletteView::new(palette);
    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let mut cache = HashMap::<u32, Vec<usize>>::new();
    let palette_gamma = view.gamma_table();

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;

        for x in 0..width {
            let offset = x.checked_mul(bpp).ok_or(BufferError::OutOfBounds)?;
            let source_rgb = match format {
                PixelFormat::Gray8 => {
                    let g = row[offset];
                    [g, g, g]
                }
                PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
                    [row[offset], row[offset + 1], row[offset + 2]]
                }
                _ => {
                    return Err(crate::Error::UnsupportedFormat(
                        "yliluoma ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            };
            let key = pack_rgb(source_rgb);
            let selected = {
                let plan = cache.entry(key).or_insert_with(|| {
                    devise_color_sequence_algorithm3(source_rgb, &view, &palette_gamma)
                });
                let map_value = usize::from(BAYER_8X8[y % 8][x % 8]);
                let plan_index = map_value
                    .checked_mul(plan.len())
                    .ok_or(BufferError::OutOfBounds)?
                    / usize::from(DITHER_LEVELS);

                plan[plan_index]
            };
            let chosen = view.colors[selected];

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
                _ => {
                    return Err(crate::Error::UnsupportedFormat(
                        "yliluoma ordered dithering supports Gray8, Rgb8, and Rgba8 only",
                    ));
                }
            }
        }
    }

    Ok(())
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

fn devise_color_sequence(target: [u8; 3], palette: &[[u8; 3]], palette_luma: &[u32]) -> Vec<usize> {
    let sequence_len = usize::from(DITHER_LEVELS);
    let mut result = Vec::with_capacity(sequence_len);
    let mut proportion_total = 0_usize;
    let mut so_far = [0_u32; 3];

    while proportion_total < sequence_len {
        let mut chosen_amount = 1_usize;
        let mut chosen_index = 0_usize;
        let max_test_count = proportion_total.max(1);
        let mut least_penalty = f64::INFINITY;

        for (index, &color) in palette.iter().enumerate() {
            let mut sum = so_far;
            let mut add = [
                u32::from(color[0]),
                u32::from(color[1]),
                u32::from(color[2]),
            ];
            let mut amount = 1_usize;

            while amount <= max_test_count {
                for channel in 0..3 {
                    sum[channel] = sum[channel].saturating_add(add[channel]);
                    add[channel] = add[channel].saturating_add(add[channel]);
                }

                let total = proportion_total.saturating_add(amount);
                let total_u32 = total as u32;
                let tested = [
                    (sum[0] / total_u32) as u8,
                    (sum[1] / total_u32) as u8,
                    (sum[2] / total_u32) as u8,
                ];
                let penalty = color_compare_rgb_luma(target, tested);

                if penalty < least_penalty {
                    least_penalty = penalty;
                    chosen_index = index;
                    chosen_amount = amount;
                }

                amount = amount.saturating_mul(2);
            }
        }

        for _ in 0..chosen_amount {
            if proportion_total >= sequence_len {
                break;
            }

            result.push(chosen_index);
            proportion_total += 1;
        }

        let chosen = palette[chosen_index];
        let chosen_amount_u32 = chosen_amount as u32;
        so_far[0] = so_far[0].saturating_add(u32::from(chosen[0]) * chosen_amount_u32);
        so_far[1] = so_far[1].saturating_add(u32::from(chosen[1]) * chosen_amount_u32);
        so_far[2] = so_far[2].saturating_add(u32::from(chosen[2]) * chosen_amount_u32);
    }

    result.sort_by_key(|&index| palette_luma[index]);
    result
}

fn color_compare_rgb_luma(a: [u8; 3], b: [u8; 3]) -> f64 {
    let luma_a = (f64::from(a[0]) * 299.0 + f64::from(a[1]) * 587.0 + f64::from(a[2]) * 114.0)
        / (255.0 * 1000.0);
    let luma_b = (f64::from(b[0]) * 299.0 + f64::from(b[1]) * 587.0 + f64::from(b[2]) * 114.0)
        / (255.0 * 1000.0);
    let luma_diff = luma_a - luma_b;
    let diff_r = (f64::from(a[0]) - f64::from(b[0])) / 255.0;
    let diff_g = (f64::from(a[1]) - f64::from(b[1])) / 255.0;
    let diff_b = (f64::from(a[2]) - f64::from(b[2])) / 255.0;

    (diff_r * diff_r * 0.299 + diff_g * diff_g * 0.587 + diff_b * diff_b * 0.114) * 0.75
        + luma_diff * luma_diff
}

fn devise_color_sequence_algorithm3(
    target: [u8; 3],
    view: &PaletteView<'_>,
    palette_gamma: &[[f64; 3]],
) -> Vec<usize> {
    let palette_len = view.colors.len();
    let mut counts = vec![0_usize; palette_len];
    let initial = nearest_palette_index(target, view);
    counts[initial] = usize::from(DITHER_LEVELS);
    let mut current_penalty = evaluate_penalty_with_counts(target, &counts, palette_gamma);

    loop {
        let mut best_penalty = current_penalty;
        let mut best_split: Option<(usize, usize, usize, usize, usize)> = None;

        for split_from in 0..palette_len {
            let split_count = counts[split_from];
            if split_count == 0 {
                continue;
            }

            let portion1 = split_count / 2;
            let portion2 = split_count - portion1;
            let mut base = [0.0_f64; 3];

            for (index, &count) in counts.iter().enumerate() {
                if index == split_from || count == 0 {
                    continue;
                }

                let weight = count as f64 / f64::from(DITHER_LEVELS);
                base[0] += palette_gamma[index][0] * weight;
                base[1] += palette_gamma[index][1] * weight;
                base[2] += palette_gamma[index][2] * weight;
            }

            let weight1 = portion1 as f64 / f64::from(DITHER_LEVELS);
            let weight2 = portion2 as f64 / f64::from(DITHER_LEVELS);

            for c1 in 0..palette_len {
                let c2_start = if portion1 == portion2 { c1 + 1 } else { 0 };
                for c2 in c2_start..palette_len {
                    if c1 == c2 {
                        continue;
                    }

                    let mixed_gamma = [
                        base[0] + palette_gamma[c1][0] * weight1 + palette_gamma[c2][0] * weight2,
                        base[1] + palette_gamma[c1][1] * weight1 + palette_gamma[c2][1] * weight2,
                        base[2] + palette_gamma[c1][2] * weight1 + palette_gamma[c2][2] * weight2,
                    ];
                    let tested = [
                        gamma_uncorrect_channel(mixed_gamma[0]),
                        gamma_uncorrect_channel(mixed_gamma[1]),
                        gamma_uncorrect_channel(mixed_gamma[2]),
                    ];
                    let penalty = color_compare_rgb_luma(target, tested);

                    if penalty < best_penalty {
                        best_penalty = penalty;
                        best_split = Some((split_from, split_count, c1, c2, portion1));
                    }
                }
            }
        }

        let Some((split_from, split_count, c1, c2, portion1)) = best_split else {
            break;
        };

        if best_penalty >= current_penalty {
            break;
        }

        let portion2 = split_count - portion1;
        counts[split_from] = 0;
        counts[c1] = counts[c1].saturating_add(portion1);
        counts[c2] = counts[c2].saturating_add(portion2);
        current_penalty = best_penalty;
    }

    let mut result = Vec::with_capacity(usize::from(DITHER_LEVELS));
    for (index, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            result.push(index);
        }
    }

    if result.is_empty() {
        result.push(initial);
    }

    result.sort_by_key(|&index| (view.luma[index], index));
    result
}

fn evaluate_penalty_with_counts(
    target: [u8; 3],
    counts: &[usize],
    palette_gamma: &[[f64; 3]],
) -> f64 {
    let mut mixed_gamma = [0.0_f64; 3];

    for (index, &count) in counts.iter().enumerate() {
        if count == 0 {
            continue;
        }

        let weight = count as f64 / f64::from(DITHER_LEVELS);
        mixed_gamma[0] += palette_gamma[index][0] * weight;
        mixed_gamma[1] += palette_gamma[index][1] * weight;
        mixed_gamma[2] += palette_gamma[index][2] * weight;
    }

    let mixed = [
        gamma_uncorrect_channel(mixed_gamma[0]),
        gamma_uncorrect_channel(mixed_gamma[1]),
        gamma_uncorrect_channel(mixed_gamma[2]),
    ];

    color_compare_rgb_luma(target, mixed)
}

fn nearest_palette_index(target: [u8; 3], view: &PaletteView<'_>) -> usize {
    let packed = pack_rgb(target);
    if let Some(index) = view.packed.iter().position(|&entry| entry == packed) {
        return index;
    }

    let mut best_index = 0_usize;
    let mut best_error = rgb_distance_sq(target, view.colors[0]);

    for (index, &color) in view.colors.iter().enumerate().skip(1) {
        let error = rgb_distance_sq(target, color);
        if error < best_error {
            best_error = error;
            best_index = index;
        }
    }

    best_index
}

fn gamma_correct_channel(value: u8) -> f64 {
    (f64::from(value) / 255.0).powf(GAMMA)
}

fn gamma_uncorrect_channel(value: f64) -> u8 {
    (value.clamp(0.0, 1.0).powf(INV_GAMMA) * 255.0).round() as u8
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
