mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_8x8, gray_ramp_8x8_u16, rgb_cube_strip,
    rgb_gradient_8x8, rgb_gradient_8x8_f32, rgb_gradient_8x8_u16,
};
use dithr::data::{
    generate_bayer_16x16, BAYER_2X2, BAYER_4X4, BAYER_8X8, CLUSTER_DOT_4X4, CLUSTER_DOT_8X8,
};
use dithr::{
    bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place,
    cluster_dot_4x4_in_place, cluster_dot_8x8_in_place, custom_ordered_in_place,
    yliluoma_1_in_place, yliluoma_2_in_place, yliluoma_3_in_place, Error, IndexedImage,
    IndexedImage16, OrderedError, Palette, QuantizeMode,
};

const BAYER_2X2_FLAT: [u8; 4] = [0, 2, 3, 1];
const BAYER_4X4_FLAT: [u8; 16] = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

#[test]
fn bayer_2x2_contains_unique_values() {
    assert_unique_square_coverage_2(BAYER_2X2);
}

#[test]
fn bayer_4x4_contains_unique_values() {
    assert_unique_square_coverage_4(BAYER_4X4);
}

#[test]
fn bayer_8x8_contains_unique_values() {
    assert_unique_square_coverage_8(BAYER_8X8);
}

#[test]
fn generated_bayer_16x16_contains_0_to_255_once() {
    let map = generate_bayer_16x16();
    let mut seen = [false; 256];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 256);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}

#[test]
fn cluster_maps_have_expected_dimensions() {
    assert_eq!(CLUSTER_DOT_4X4.len(), 4);
    for row in CLUSTER_DOT_4X4 {
        assert_eq!(row.len(), 4);
    }

    assert_eq!(CLUSTER_DOT_8X8.len(), 8);
    for row in CLUSTER_DOT_8X8 {
        assert_eq!(row.len(), 8);
    }
}

#[test]
fn bayer_2x2_quantizes_only_to_allowed_values() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    bayer_2x2_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("bayer 2x2 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn bayer_4x4_periodicity_matches_4() {
    let mut data = vec![127_u8; 64];
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    bayer_4x4_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("bayer 4x4 should succeed");

    for y in 0..8 {
        for x in 0..4 {
            assert_eq!(data[y * 8 + x], data[y * 8 + x + 4]);
        }
    }

    for y in 0..4 {
        for x in 0..8 {
            assert_eq!(data[y * 8 + x], data[(y + 4) * 8 + x]);
        }
    }
}

#[test]
fn bayer_8x8_periodicity_matches_8() {
    let mut data = vec![127_u8; 256];
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("bayer 8x8 should succeed");

    for y in 0..16 {
        for x in 0..8 {
            assert_eq!(data[y * 16 + x], data[y * 16 + x + 8]);
        }
    }

    for y in 0..8 {
        for x in 0..16 {
            assert_eq!(data[y * 16 + x], data[(y + 8) * 16 + x]);
        }
    }
}

#[test]
fn bayer_16x16_runs_on_16x16_without_panic() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_16x16_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("bayer 16x16 should succeed");

    assert_eq!(data.len(), 256);
    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn cluster_dot_4x4_runs_and_quantizes() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    cluster_dot_4x4_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("cluster-dot 4x4 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn cluster_dot_8x8_runs_and_quantizes() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    cluster_dot_8x8_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("cluster-dot 8x8 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn custom_ordered_rejects_empty_map() {
    let mut data = vec![0_u8; 4];
    let mut buffer = dithr::gray_u8(&mut data, 2, 2, 2).expect("valid buffer should construct");

    let result = custom_ordered_in_place(&mut buffer, QuantizeMode::gray_bits(1), &[], 0, 0, 64);

    assert_eq!(result, Err(Error::Ordered(OrderedError::EmptyMap)));
}

#[test]
fn custom_ordered_rejects_bad_dimensions() {
    let mut data = vec![0_u8; 4];
    let mut buffer = dithr::gray_u8(&mut data, 2, 2, 2).expect("valid buffer should construct");

    let map = [0_u8, 1, 2];
    let result = custom_ordered_in_place(&mut buffer, QuantizeMode::gray_bits(1), &map, 2, 2, 64);

    assert_eq!(result, Err(Error::Ordered(OrderedError::InvalidDimensions)));
}

#[test]
fn custom_ordered_rejects_out_of_range_map_values() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");
    let normalized_map = [0_u8, 85, 170, 255];

    let result = custom_ordered_in_place(
        &mut buffer,
        QuantizeMode::gray_bits(1),
        &normalized_map,
        2,
        2,
        64,
    );
    assert_eq!(result, Err(Error::Ordered(OrderedError::ValueOutOfRange)));
}

#[test]
fn custom_ordered_2x2_matches_manual_small_case() {
    let mut data_a: Vec<u8> = (0_u8..16).map(|value| value.saturating_mul(16)).collect();
    let mut data_b = data_a.clone();

    let mut buffer_a = dithr::gray_u8(&mut data_a, 4, 4, 4).expect("valid buffer should construct");
    let mut buffer_b = dithr::gray_u8(&mut data_b, 4, 4, 4).expect("valid buffer should construct");

    custom_ordered_in_place(
        &mut buffer_a,
        QuantizeMode::gray_bits(1),
        &BAYER_2X2_FLAT,
        2,
        2,
        64,
    )
    .expect("custom ordered dither should succeed");
    bayer_2x2_in_place(&mut buffer_b, QuantizeMode::gray_bits(1))
        .expect("bayer 2x2 should succeed");

    assert_eq!(data_a, data_b);
}

#[test]
fn yliluoma_1_output_is_always_palette_member() {
    let mut data = rgb_gradient_8x8();
    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = dithr::rgb_u8(&mut data, 8, 8, 24).expect("valid buffer should construct");

    yliluoma_1_in_place(&mut buffer, &palette).expect("yliluoma 1 should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.as_slice().contains(&rgb));
    }
}

#[test]
fn yliluoma_1_deterministic_rgb_fixture() {
    let mut data_a = rgb_gradient_8x8();
    let mut data_b = rgb_gradient_8x8();
    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer_a = dithr::rgb_u8(&mut data_a, 8, 8, 24).expect("valid buffer should construct");
    let mut buffer_b = dithr::rgb_u8(&mut data_b, 8, 8, 24).expect("valid buffer should construct");

    yliluoma_1_in_place(&mut buffer_a, &palette).expect("yliluoma 1 should succeed");
    yliluoma_1_in_place(&mut buffer_b, &palette).expect("yliluoma 1 should succeed");

    assert_eq!(data_a, data_b);
}

#[test]
fn yliluoma_2_output_is_always_palette_member() {
    let mut data = rgb_gradient_8x8();
    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = dithr::rgb_u8(&mut data, 8, 8, 24).expect("valid buffer should construct");

    yliluoma_2_in_place(&mut buffer, &palette).expect("yliluoma 2 should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.as_slice().contains(&rgb));
    }
}

#[test]
fn yliluoma_3_output_is_always_palette_member() {
    let mut data = rgb_gradient_8x8();
    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [0, 255, 255],
        [255, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = dithr::rgb_u8(&mut data, 8, 8, 24).expect("valid buffer should construct");

    yliluoma_3_in_place(&mut buffer, &palette).expect("yliluoma 3 should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.as_slice().contains(&rgb));
    }
}

#[test]
fn yliluoma_u16_palette_member_invariant() {
    let width = 8_usize;
    let height = 8_usize;
    let mut data = vec![0_u16; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = ((x * 65_535) / (width - 1)) as u16;
            data[offset + 1] = ((y * 65_535) / (height - 1)) as u16;
            data[offset + 2] = (((x + y) * 65_535) / (width + height - 2)) as u16;
        }
    }
    let palette = Palette::<u16>::new(vec![
        [0, 0, 0],
        [65_535, 65_535, 65_535],
        [65_535, 0, 0],
        [0, 65_535, 0],
        [0, 0, 65_535],
        [65_535, 65_535, 0],
        [0, 65_535, 65_535],
        [65_535, 0, 65_535],
    ])
    .expect("palette should be valid");
    let mut buffer =
        dithr::rgb_u16(&mut data, width, height, width * 3).expect("valid buffer should construct");

    yliluoma_1_in_place(&mut buffer, &palette).expect("yliluoma 1 should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.contains(rgb));
    }
}

#[test]
fn yliluoma_f32_palette_member_invariant() {
    let width = 8_usize;
    let height = 8_usize;
    let mut data = vec![0.0_f32; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = x as f32 / (width - 1) as f32;
            data[offset + 1] = y as f32 / (height - 1) as f32;
            data[offset + 2] = (x + y) as f32 / (width + height - 2) as f32;
        }
    }
    let palette = Palette::<f32>::new(vec![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    .expect("palette should be valid");
    let mut buffer =
        dithr::rgb_32f(&mut data, width, height, width * 3).expect("valid buffer should construct");

    yliluoma_1_in_place(&mut buffer, &palette).expect("yliluoma 1 should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.contains(rgb));
    }
}

#[test]
fn yliluoma_rgba_alpha_preserved_u16() {
    let width = 8_usize;
    let height = 8_usize;
    let mut data: Vec<u16> = (0_u32..(width * height * 4) as u32)
        .map(|v| ((v * 977) % 65_536) as u16)
        .collect();
    let before_alpha: Vec<u16> = data.iter().skip(3).step_by(4).copied().collect();
    let palette = Palette::<u16>::new(vec![
        [0, 0, 0],
        [65_535, 65_535, 65_535],
        [65_535, 0, 0],
        [0, 65_535, 0],
        [0, 0, 65_535],
    ])
    .expect("palette should be valid");
    let mut buffer = dithr::rgba_u16(&mut data, width, height, width * 4)
        .expect("valid buffer should construct");

    yliluoma_2_in_place(&mut buffer, &palette).expect("yliluoma 2 should succeed");

    let after_alpha: Vec<u16> = data.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_alpha, after_alpha);
}

#[test]
fn yliluoma_to_indexed_u16_palette_member_consistency() {
    let width = 8_usize;
    let height = 8_usize;
    let mut data = vec![0_u16; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = ((x * 65_535) / (width - 1)) as u16;
            data[offset + 1] = ((y * 65_535) / (height - 1)) as u16;
            data[offset + 2] = (((x + y) * 65_535) / (width + height - 2)) as u16;
        }
    }

    let palette = Palette::<u16>::new(vec![
        [0, 0, 0],
        [65_535, 65_535, 65_535],
        [65_535, 0, 0],
        [0, 65_535, 0],
        [0, 0, 65_535],
        [65_535, 65_535, 0],
        [0, 65_535, 65_535],
        [65_535, 0, 65_535],
    ])
    .expect("palette should be valid");

    let mut buffer =
        dithr::rgb_u16(&mut data, width, height, width * 3).expect("valid buffer should construct");
    yliluoma_1_in_place(&mut buffer, &palette).expect("yliluoma 1 should succeed");

    let mut indices = Vec::with_capacity(width * height);
    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.contains(rgb));
        let idx = palette.nearest_rgb_index(rgb) as u8;
        assert_eq!(palette.get(usize::from(idx)), Some(rgb));
        indices.push(idx);
    }

    let indexed: IndexedImage16 = IndexedImage {
        indices,
        width,
        height,
        palette,
    };

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            let rgb = [data[offset], data[offset + 1], data[offset + 2]];
            assert_eq!(indexed.color_at(x, y), Some(rgb));
        }
    }
}

#[test]
fn ordered_engine_gray_uses_only_quantized_values() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    custom_ordered_in_place(
        &mut buffer,
        QuantizeMode::gray_bits(1),
        &BAYER_2X2_FLAT,
        2,
        2,
        64,
    )
    .expect("custom ordered dither should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ordered_engine_rgb_palette_output_is_palette_member() {
    let mut data = vec![
        16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 24, 40, 56, 72, 88, 104, 120, 136,
        152, 168, 184, 200, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 40, 56, 72, 88,
        104, 120, 136, 152, 168, 184, 200, 216,
    ];
    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = dithr::rgb_u8(&mut data, 4, 4, 12).expect("valid buffer should construct");

    custom_ordered_in_place(
        &mut buffer,
        QuantizeMode::Palette(&palette),
        &BAYER_4X4_FLAT,
        4,
        4,
        64,
    )
    .expect("custom ordered dither should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.as_slice().contains(&rgb));
    }
}

#[test]
fn ordered_engine_is_deterministic() {
    let seed_data: Vec<u8> = (0_u16..120)
        .map(|value| ((value * 13 + 7) % 256) as u8)
        .collect();
    let mut a = seed_data.clone();
    let mut b = seed_data;

    let mut buffer_a = dithr::rgb_u8(&mut a, 8, 5, 24).expect("valid buffer should construct");
    let mut buffer_b = dithr::rgb_u8(&mut b, 8, 5, 24).expect("valid buffer should construct");

    custom_ordered_in_place(
        &mut buffer_a,
        QuantizeMode::rgb_bits(3),
        &BAYER_4X4_FLAT,
        4,
        4,
        40,
    )
    .expect("custom ordered dither should succeed");
    custom_ordered_in_place(
        &mut buffer_b,
        QuantizeMode::rgb_bits(3),
        &BAYER_4X4_FLAT,
        4,
        4,
        40,
    )
    .expect("custom ordered dither should succeed");

    assert_eq!(a, b);
}

#[test]
fn ordered_engine_preserves_alpha_channel() {
    let mut data: Vec<u8> = (0_u16..80)
        .map(|value| ((value * 19 + 11) % 256) as u8)
        .collect();
    let before_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    let mut buffer = dithr::rgba_u8(&mut data, 4, 5, 16).expect("valid buffer should construct");

    custom_ordered_in_place(
        &mut buffer,
        QuantizeMode::rgb_bits(2),
        &BAYER_2X2_FLAT,
        2,
        2,
        48,
    )
    .expect("custom ordered dither should succeed");

    let after_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_alpha, after_alpha);
}

#[test]
fn bayer_8x8_u16_runs_and_quantizes() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("bayer 8x8 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn bayer_8x8_f32_runs_and_quantizes() {
    let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
        .map(|value| (value % 256) as f32 / 255.0)
        .collect();
    let mut buffer = dithr::rgb_32f(&mut data, 16, 16, 48).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2)).expect("bayer 8x8 should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

#[test]
fn ordered_bayer_u16_runs_with_same_invariants() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("bayer 8x8 should succeed");

    assert_eq!(data.len(), 256);
    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn ordered_bayer_f32_runs_with_same_invariants() {
    let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
        .map(|value| (value % 256) as f32 / 255.0)
        .collect();
    let mut buffer = dithr::rgb_32f(&mut data, 16, 16, 48).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2)).expect("bayer 8x8 should succeed");

    assert_eq!(data.len(), 16 * 16 * 3);
    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

#[test]
fn ordered_engine_preserves_alpha_across_u8_u16_f32() {
    let map = [0_u8, 2, 3, 1];

    let mut rgba_u8: Vec<u8> = (0_u16..64).map(|v| (v * 3) as u8).collect();
    let before_u8: Vec<u8> = rgba_u8.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_u8 =
        dithr::rgba_u8(&mut rgba_u8, 4, 4, 16).expect("valid buffer should construct");
    custom_ordered_in_place(&mut buffer_u8, QuantizeMode::rgb_bits(2), &map, 2, 2, 48)
        .expect("u8 ordered dithering should succeed");
    let after_u8: Vec<u8> = rgba_u8.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_u8, after_u8);

    let mut rgba_u16: Vec<u16> = (0_u32..64).map(|v| ((v * 1009) % 65_536) as u16).collect();
    let before_u16: Vec<u16> = rgba_u16.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_u16 =
        dithr::rgba_u16(&mut rgba_u16, 4, 4, 16).expect("valid buffer should construct");
    custom_ordered_in_place(&mut buffer_u16, QuantizeMode::RgbLevels(4), &map, 2, 2, 48)
        .expect("u16 ordered dithering should succeed");
    let after_u16: Vec<u16> = rgba_u16.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_u16, after_u16);

    let mut rgba_f32: Vec<f32> = (0_u32..64).map(|v| (v as f32) / 63.0).collect();
    let before_f32: Vec<f32> = rgba_f32.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_f32 =
        dithr::rgba_32f(&mut rgba_f32, 4, 4, 16).expect("valid buffer should construct");
    custom_ordered_in_place(&mut buffer_f32, QuantizeMode::RgbLevels(4), &map, 2, 2, 48)
        .expect("f32 ordered dithering should succeed");
    let after_f32: Vec<f32> = rgba_f32.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_f32, after_f32);
}

#[test]
fn ordered_alpha_preserved_all_sample_types() {
    let map = [0_u8, 2, 3, 1];

    let mut rgba_u8: Vec<u8> = (0_u16..64).map(|v| (v * 3) as u8).collect();
    let before_u8: Vec<u8> = rgba_u8.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_u8 =
        dithr::rgba_u8(&mut rgba_u8, 4, 4, 16).expect("valid buffer should construct");
    custom_ordered_in_place(&mut buffer_u8, QuantizeMode::GrayLevels(2), &map, 2, 2, 48)
        .expect("u8 ordered dithering should succeed");
    let after_u8: Vec<u8> = rgba_u8.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_u8, after_u8);

    let mut rgba_u16: Vec<u16> = (0_u32..64).map(|v| ((v * 1009) % 65_536) as u16).collect();
    let before_u16: Vec<u16> = rgba_u16.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_u16 =
        dithr::rgba_u16(&mut rgba_u16, 4, 4, 16).expect("valid buffer should construct");
    custom_ordered_in_place(&mut buffer_u16, QuantizeMode::GrayLevels(2), &map, 2, 2, 48)
        .expect("u16 ordered dithering should succeed");
    let after_u16: Vec<u16> = rgba_u16.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_u16, after_u16);

    let mut rgba_f32: Vec<f32> = (0_u32..64).map(|v| (v as f32) / 63.0).collect();
    let before_f32: Vec<f32> = rgba_f32.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_f32 =
        dithr::rgba_32f(&mut rgba_f32, 4, 4, 16).expect("valid buffer should construct");
    custom_ordered_in_place(&mut buffer_f32, QuantizeMode::GrayLevels(2), &map, 2, 2, 48)
        .expect("f32 ordered dithering should succeed");
    let after_f32: Vec<f32> = rgba_f32.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_f32, after_f32);
}

#[test]
fn ordered_core_no_duplicate_integer_float_paths_smoke() {
    let mut gray_u8 = gray_ramp_16x16();
    let mut gray_f32: Vec<f32> = gray_u8.iter().map(|&v| f32::from(v) / 255.0).collect();

    let mut buffer_u8 =
        dithr::gray_u8(&mut gray_u8, 16, 16, 16).expect("valid buffer should construct");
    let mut buffer_f32 =
        dithr::gray_32f(&mut gray_f32, 16, 16, 16).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer_u8, QuantizeMode::GrayLevels(2)).expect("u8 run should succeed");
    bayer_8x8_in_place(&mut buffer_f32, QuantizeMode::GrayLevels(2))
        .expect("f32 run should succeed");

    let mask_u8: Vec<u8> = gray_u8.iter().map(|&v| u8::from(v > 127)).collect();
    let mask_f32: Vec<u8> = gray_f32.iter().map(|&v| u8::from(v > 0.5)).collect();
    assert_eq!(mask_u8, mask_f32);
}

#[test]
fn ordered_threshold_tile_logic_matches_u8_u16_f32() {
    let map = [0_u8, 2, 3, 1];
    let width = 4_usize;
    let height = 4_usize;

    let mut data_u8 = vec![127_u8; width * height];
    let mut data_u16 = vec![32_767_u16; width * height];
    let mut data_f32 = vec![0.5_f32; width * height];

    let mut buffer_u8 =
        dithr::gray_u8(&mut data_u8, width, height, width).expect("valid buffer should construct");
    let mut buffer_u16 = dithr::gray_u16(&mut data_u16, width, height, width)
        .expect("valid buffer should construct");
    let mut buffer_f32 = dithr::gray_32f(&mut data_f32, width, height, width)
        .expect("valid buffer should construct");

    custom_ordered_in_place(&mut buffer_u8, QuantizeMode::GrayLevels(2), &map, 2, 2, 64)
        .expect("u8 run should succeed");
    custom_ordered_in_place(&mut buffer_u16, QuantizeMode::GrayLevels(2), &map, 2, 2, 64)
        .expect("u16 run should succeed");
    custom_ordered_in_place(&mut buffer_f32, QuantizeMode::GrayLevels(2), &map, 2, 2, 64)
        .expect("f32 run should succeed");

    let mask_u8: Vec<u8> = data_u8.iter().map(|&v| u8::from(v > 127)).collect();
    let mask_u16: Vec<u8> = data_u16.iter().map(|&v| u8::from(v > 32_767)).collect();
    let mask_f32: Vec<u8> = data_f32.iter().map(|&v| u8::from(v > 0.5)).collect();

    assert_eq!(mask_u8, mask_u16);
    assert_eq!(mask_u8, mask_f32);
}

#[test]
fn custom_ordered_map_tiling_is_consistent() {
    let mut data = vec![127_u8; 64];
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    custom_ordered_in_place(
        &mut buffer,
        QuantizeMode::gray_bits(1),
        &BAYER_2X2_FLAT,
        2,
        2,
        64,
    )
    .expect("custom ordered dither should succeed");

    for y in 0..8 {
        for x in 0..6 {
            assert_eq!(data[y * 8 + x], data[y * 8 + x + 2]);
        }
    }

    for y in 0..6 {
        for x in 0..8 {
            assert_eq!(data[y * 8 + x], data[(y + 2) * 8 + x]);
        }
    }
}

#[test]
fn fixture_builders_are_deterministic() {
    let gray8 = gray_ramp_8x8();
    let gray16 = gray_ramp_16x16();
    let gray8_u16 = gray_ramp_8x8_u16();
    let checker = checker_8x8();
    let gradient = rgb_gradient_8x8();
    let gradient_u16 = rgb_gradient_8x8_u16();
    let gradient_f32 = rgb_gradient_8x8_f32();
    let cube = rgb_cube_strip();

    assert_eq!(gray8.len(), 64);
    assert_eq!(gray16.len(), 256);
    assert_eq!(gray8_u16.len(), 64);
    assert_eq!(checker.len(), 64);
    assert_eq!(gradient.len(), 8 * 8 * 3);
    assert_eq!(gradient_u16.len(), 8 * 8 * 3);
    assert_eq!(gradient_f32.len(), 8 * 8 * 3);
    assert_eq!(cube.len(), 27 * 3);

    assert_eq!(checker.iter().filter(|&&value| value == 0).count(), 32);
    assert_eq!(checker.iter().filter(|&&value| value == 255).count(), 32);
    assert_eq!(gray8_u16.first().copied(), Some(0));
    assert_eq!(gray8_u16.last().copied(), Some(65_535));
    assert_eq!(gradient_u16.iter().copied().min(), Some(0));
    assert_eq!(gradient_u16.iter().copied().max(), Some(65_535));
    assert!(gradient_f32.iter().all(|&v| (0.0..=1.0).contains(&v)));

    assert_eq!(fnv1a64(&gray8), fnv1a64(&gray_ramp_8x8()));
    assert_eq!(fnv1a64(&gray16), fnv1a64(&gray_ramp_16x16()));
    assert_eq!(fnv1a64(&checker), fnv1a64(&checker_8x8()));
    assert_eq!(fnv1a64(&gradient), fnv1a64(&rgb_gradient_8x8()));
    assert_eq!(fnv1a64(&cube), fnv1a64(&rgb_cube_strip()));
}

#[test]
fn ordered_public_api_packed_constructor_smoke() {
    let mut data = gray_ramp_8x8();
    let mut buffer =
        dithr::gray_u8_packed(&mut data, 8, 8).expect("valid packed gray buffer should construct");
    bayer_2x2_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("bayer should succeed");
    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[cfg(feature = "rayon")]
#[test]
fn bayer_8x8_parallel_matches_sequential() {
    use dithr::bayer_8x8_in_place_par;

    let mut seq = gray_ramp_16x16();
    let mut par = seq.clone();
    let mut seq_buffer =
        dithr::gray_u8(&mut seq, 16, 16, 16).expect("valid buffer should construct");
    let mut par_buffer =
        dithr::gray_u8(&mut par, 16, 16, 16).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut seq_buffer, QuantizeMode::gray_bits(1))
        .expect("sequential should succeed");
    bayer_8x8_in_place_par(&mut par_buffer, QuantizeMode::gray_bits(1))
        .expect("parallel should succeed");

    assert_eq!(seq, par);
}

#[cfg(feature = "rayon")]
#[test]
fn cluster_dot_8x8_parallel_matches_sequential() {
    use dithr::cluster_dot_8x8_in_place_par;

    let mut seq = gray_ramp_16x16();
    let mut par = seq.clone();
    let mut seq_buffer =
        dithr::gray_u8(&mut seq, 16, 16, 16).expect("valid buffer should construct");
    let mut par_buffer =
        dithr::gray_u8(&mut par, 16, 16, 16).expect("valid buffer should construct");

    cluster_dot_8x8_in_place(&mut seq_buffer, QuantizeMode::gray_bits(1))
        .expect("sequential should succeed");
    cluster_dot_8x8_in_place_par(&mut par_buffer, QuantizeMode::gray_bits(1))
        .expect("parallel should succeed");

    assert_eq!(seq, par);
}

#[cfg(feature = "rayon")]
#[test]
fn custom_ordered_parallel_matches_sequential() {
    use dithr::custom_ordered_in_place_par;

    let map = [0_u8, 2, 3, 1];
    let mut seq = gray_ramp_16x16();
    let mut par = seq.clone();
    let mut seq_buffer =
        dithr::gray_u8(&mut seq, 16, 16, 16).expect("valid buffer should construct");
    let mut par_buffer =
        dithr::gray_u8(&mut par, 16, 16, 16).expect("valid buffer should construct");

    custom_ordered_in_place(&mut seq_buffer, QuantizeMode::gray_bits(1), &map, 2, 2, 64)
        .expect("sequential should succeed");
    custom_ordered_in_place_par(&mut par_buffer, QuantizeMode::gray_bits(1), &map, 2, 2, 64)
        .expect("parallel should succeed");

    assert_eq!(seq, par);
}

fn assert_unique_square_coverage_2(map: [[u8; 2]; 2]) {
    let mut seen = [false; 4];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 4);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}

fn assert_unique_square_coverage_4(map: [[u8; 4]; 4]) {
    let mut seen = [false; 16];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 16);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}

fn assert_unique_square_coverage_8(map: [[u8; 8]; 8]) {
    let mut seen = [false; 64];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 64);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}
