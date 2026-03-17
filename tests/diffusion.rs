mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_8x8, rgb_cube_strip, rgb_gradient_8x8,
};
use dithr::{
    atkinson_in_place, burkes_in_place, false_floyd_steinberg_in_place, fan_in_place,
    floyd_steinberg_in_place, gradient_based_error_diffusion_in_place,
    jarvis_judice_ninke_in_place, ostromoukhov_in_place, shiau_fan_2_in_place, shiau_fan_in_place,
    sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place, stucki_in_place,
    two_row_sierra_in_place, zhou_fang_in_place, Buffer, Palette, PixelFormat, QuantizeMode,
};

#[test]
fn diffusion_engine_gray_binary_output_for_graybits1() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn floyd_steinberg_gray_bits1_binary_only() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn false_floyd_steinberg_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    false_floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("false floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn jarvis_judice_ninke_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    jarvis_judice_ninke_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("jarvis-judice-ninke should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn stucki_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    stucki_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("stucki should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn burkes_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    burkes_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("burkes should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn sierra_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    sierra_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("sierra should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn two_row_sierra_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    two_row_sierra_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("two-row sierra should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn sierra_lite_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    sierra_lite_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("sierra lite should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn stevenson_arce_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    stevenson_arce_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("stevenson-arce should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn atkinson_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    atkinson_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("atkinson should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn fan_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    fan_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn shiau_fan_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    shiau_fan_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn shiau_fan_2_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    shiau_fan_2_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ostromoukhov_runs_gray() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    ostromoukhov_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ostromoukhov_coeff_index_in_range() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    ostromoukhov_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(data.len(), 256);
}

#[test]
fn zhou_fang_runs_gray() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    zhou_fang_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn zhou_fang_modulation_in_range() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    zhou_fang_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(data.len(), 256);
}

#[test]
fn gradient_based_error_diffusion_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    gradient_based_error_diffusion_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn diffusion_engine_rgb_palette_output_is_palette_member() {
    let mut data = vec![
        16, 24, 32, 48, 56, 64, 80, 88, 96, 112, 120, 128, 144, 152, 160, 176, 184, 192, 208, 216,
        224, 240, 248, 255, 12, 68, 124, 28, 84, 140, 44, 100, 156, 60, 116, 172, 76, 132, 188, 92,
        148, 204, 108, 164, 220, 124, 180, 236,
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

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::Palette(&palette))
        .expect("floyd-steinberg should succeed");

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.as_slice().contains(&rgb));
    }
}

#[test]
fn diffusion_engine_preserves_alpha() {
    let mut data: Vec<u8> = (0_u16..120)
        .map(|value| ((value * 17 + 9) % 256) as u8)
        .collect();
    let before_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 6,
        height: 5,
        stride: 24,
        format: PixelFormat::Rgba8,
    };

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::RgbBits(2))
        .expect("floyd-steinberg should succeed");

    let after_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_alpha, after_alpha);
}

#[test]
fn diffusion_engine_does_not_panic_on_1x1() {
    let mut data = vec![128_u8];
    let mut buffer = Buffer {
        data: &mut data,
        width: 1,
        height: 1,
        stride: 1,
        format: PixelFormat::Gray8,
    };

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("floyd-steinberg should succeed");

    assert_eq!(data.len(), 1);
}

#[test]
fn diffusion_engine_is_deterministic() {
    let seed_data: Vec<u8> = (0_u16..144)
        .map(|value| ((value * 29 + 3) % 256) as u8)
        .collect();
    let mut a = seed_data.clone();
    let mut b = seed_data;

    let mut buffer_a = Buffer {
        data: &mut a,
        width: 8,
        height: 6,
        stride: 24,
        format: PixelFormat::Rgb8,
    };
    let mut buffer_b = Buffer {
        data: &mut b,
        width: 8,
        height: 6,
        stride: 24,
        format: PixelFormat::Rgb8,
    };

    floyd_steinberg_in_place(&mut buffer_a, QuantizeMode::RgbBits(3))
        .expect("floyd-steinberg should succeed");
    floyd_steinberg_in_place(&mut buffer_b, QuantizeMode::RgbBits(3))
        .expect("floyd-steinberg should succeed");

    assert_eq!(a, b);
}

#[test]
fn fixture_builders_are_deterministic() {
    let gray8 = gray_ramp_8x8();
    let gray16 = gray_ramp_16x16();
    let checker = checker_8x8();
    let gradient = rgb_gradient_8x8();
    let cube = rgb_cube_strip();

    assert_eq!(gray8.len(), 64);
    assert_eq!(gray16.len(), 256);
    assert_eq!(checker.len(), 64);
    assert_eq!(gradient.len(), 8 * 8 * 3);
    assert_eq!(cube.len(), 27 * 3);

    assert_eq!(checker.iter().filter(|&&value| value == 0).count(), 32);
    assert_eq!(checker.iter().filter(|&&value| value == 255).count(), 32);

    assert_eq!(fnv1a64(&gray8), fnv1a64(&gray_ramp_8x8()));
    assert_eq!(fnv1a64(&gray16), fnv1a64(&gray_ramp_16x16()));
    assert_eq!(fnv1a64(&checker), fnv1a64(&checker_8x8()));
    assert_eq!(fnv1a64(&gradient), fnv1a64(&rgb_gradient_8x8()));
    assert_eq!(fnv1a64(&cube), fnv1a64(&rgb_cube_strip()));
}
