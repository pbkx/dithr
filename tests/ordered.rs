use dithr::data::{
    generate_bayer_16x16, BAYER_2X2, BAYER_4X4, BAYER_8X8, CLUSTER_DOT_4X4, CLUSTER_DOT_8X8,
};
use dithr::ordered::{ordered_dither_in_place, ordered_threshold_for_xy};
use dithr::{
    bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place,
    cluster_dot_4x4_in_place, cluster_dot_8x8_in_place, Buffer, Palette, PixelFormat, QuantizeMode,
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
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    bayer_2x2_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn bayer_4x4_periodicity_matches_4() {
    let mut data = vec![127_u8; 64];
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    bayer_4x4_in_place(&mut buffer, QuantizeMode::GrayBits(1));

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
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1));

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
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    bayer_16x16_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(data.len(), 256);
    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn cluster_dot_4x4_runs_and_quantizes() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    cluster_dot_4x4_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn cluster_dot_8x8_runs_and_quantizes() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    cluster_dot_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ordered_engine_gray_uses_only_quantized_values() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    ordered_dither_in_place(
        &mut buffer,
        QuantizeMode::GrayBits(1),
        &BAYER_2X2_FLAT,
        2,
        2,
        64,
    );

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
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = Buffer {
        data: &mut data,
        width: 4,
        height: 4,
        stride: 12,
        format: PixelFormat::Rgb8,
    };

    ordered_dither_in_place(
        &mut buffer,
        QuantizeMode::Palette(&palette),
        &BAYER_4X4_FLAT,
        4,
        4,
        64,
    );

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

    let mut buffer_a = Buffer {
        data: &mut a,
        width: 8,
        height: 5,
        stride: 24,
        format: PixelFormat::Rgb8,
    };
    let mut buffer_b = Buffer {
        data: &mut b,
        width: 8,
        height: 5,
        stride: 24,
        format: PixelFormat::Rgb8,
    };

    ordered_dither_in_place(
        &mut buffer_a,
        QuantizeMode::RgbBits(3),
        &BAYER_4X4_FLAT,
        4,
        4,
        40,
    );
    ordered_dither_in_place(
        &mut buffer_b,
        QuantizeMode::RgbBits(3),
        &BAYER_4X4_FLAT,
        4,
        4,
        40,
    );

    assert_eq!(a, b);
}

#[test]
fn ordered_engine_preserves_alpha_channel() {
    let mut data: Vec<u8> = (0_u16..80)
        .map(|value| ((value * 19 + 11) % 256) as u8)
        .collect();
    let before_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 4,
        height: 5,
        stride: 16,
        format: PixelFormat::Rgba8,
    };

    ordered_dither_in_place(
        &mut buffer,
        QuantizeMode::RgbBits(2),
        &BAYER_2X2_FLAT,
        2,
        2,
        48,
    );

    let after_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_alpha, after_alpha);
}

#[test]
fn ordered_threshold_lookup_tiles_correctly() {
    assert_eq!(ordered_threshold_for_xy(0, 0, &BAYER_2X2_FLAT, 2, 2), 0);
    assert_eq!(ordered_threshold_for_xy(1, 0, &BAYER_2X2_FLAT, 2, 2), 2);
    assert_eq!(ordered_threshold_for_xy(2, 0, &BAYER_2X2_FLAT, 2, 2), 0);
    assert_eq!(ordered_threshold_for_xy(3, 3, &BAYER_2X2_FLAT, 2, 2), 1);
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
