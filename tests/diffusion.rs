mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_8x8, gray_ramp_8x8_u16, rgb_cube_strip,
    rgb_gradient_8x8, rgb_gradient_8x8_f32, rgb_gradient_8x8_u16,
};
use dithr::{
    atkinson_in_place, burkes_in_place, false_floyd_steinberg_in_place, fan_in_place,
    floyd_steinberg_in_place, gradient_based_error_diffusion_in_place,
    jarvis_judice_ninke_in_place, ostromoukhov_in_place, shiau_fan_2_in_place, shiau_fan_in_place,
    sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place, stucki_in_place,
    two_row_sierra_in_place, zhou_fang_in_place, Error, GrayBuffer16, Palette, QuantizeMode,
    RgbBuffer32F,
};

type DiffusionWrapperU16 = fn(&mut GrayBuffer16<'_>, QuantizeMode<'_, u16>) -> dithr::Result<()>;
type DiffusionWrapperF32 = fn(&mut RgbBuffer32F<'_>, QuantizeMode<'_, f32>) -> dithr::Result<()>;

#[test]
fn diffusion_engine_gray_binary_output_for_graybits1() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn floyd_steinberg_gray_bits1_binary_only() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn false_floyd_steinberg_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    false_floyd_steinberg_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("false floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn jarvis_judice_ninke_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    jarvis_judice_ninke_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("jarvis-judice-ninke should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn stucki_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    stucki_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("stucki should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn burkes_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    burkes_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("burkes should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn sierra_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    sierra_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("sierra should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn two_row_sierra_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    two_row_sierra_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("two-row sierra should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn sierra_lite_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    sierra_lite_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("sierra lite should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn stevenson_arce_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    stevenson_arce_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("stevenson-arce should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn atkinson_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    atkinson_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("atkinson should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn fan_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    fan_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("fan should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn shiau_fan_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    shiau_fan_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("shiau-fan should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn shiau_fan_2_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    shiau_fan_2_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("shiau-fan-2 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ostromoukhov_runs_gray() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    ostromoukhov_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("ostromoukhov should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ostromoukhov_coeff_index_in_range() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    ostromoukhov_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("ostromoukhov should succeed");

    assert_eq!(data.len(), 256);
}

#[test]
fn zhou_fang_runs_gray() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    zhou_fang_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("zhou-fang should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn zhou_fang_modulation_in_range() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    zhou_fang_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("zhou-fang should succeed");

    assert_eq!(data.len(), 256);
}

#[test]
fn gradient_based_error_diffusion_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    gradient_based_error_diffusion_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("gradient-based diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ostromoukhov_rejects_non_gray_formats() {
    let mut rgb = vec![128_u8; 4 * 4 * 3];
    let mut rgb_buffer = dithr::rgb_u8(&mut rgb, 4, 4, 12).expect("valid buffer should construct");
    let rgb_result = ostromoukhov_in_place(&mut rgb_buffer, QuantizeMode::gray_bits(1));
    assert!(matches!(
        rgb_result,
        Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only"
        ))
    ));

    let mut rgba = vec![128_u8; 4 * 4 * 4];
    let mut rgba_buffer =
        dithr::rgba_u8(&mut rgba, 4, 4, 16).expect("valid buffer should construct");
    let rgba_result = ostromoukhov_in_place(&mut rgba_buffer, QuantizeMode::gray_bits(1));
    assert!(matches!(
        rgba_result,
        Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only"
        ))
    ));
}

#[test]
fn zhou_fang_rejects_non_gray_formats() {
    let mut rgb = vec![128_u8; 4 * 4 * 3];
    let mut rgb_buffer = dithr::rgb_u8(&mut rgb, 4, 4, 12).expect("valid buffer should construct");
    let rgb_result = zhou_fang_in_place(&mut rgb_buffer, QuantizeMode::gray_bits(1));
    assert!(matches!(
        rgb_result,
        Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only"
        ))
    ));

    let mut rgba = vec![128_u8; 4 * 4 * 4];
    let mut rgba_buffer =
        dithr::rgba_u8(&mut rgba, 4, 4, 16).expect("valid buffer should construct");
    let rgba_result = zhou_fang_in_place(&mut rgba_buffer, QuantizeMode::gray_bits(1));
    assert!(matches!(
        rgba_result,
        Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only"
        ))
    ));
}

#[test]
fn gradient_based_error_diffusion_rejects_non_gray_formats() {
    let mut rgb = vec![128_u8; 4 * 4 * 3];
    let mut rgb_buffer = dithr::rgb_u8(&mut rgb, 4, 4, 12).expect("valid buffer should construct");
    let rgb_result =
        gradient_based_error_diffusion_in_place(&mut rgb_buffer, QuantizeMode::gray_bits(1));
    assert!(matches!(
        rgb_result,
        Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only"
        ))
    ));

    let mut rgba = vec![128_u8; 4 * 4 * 4];
    let mut rgba_buffer =
        dithr::rgba_u8(&mut rgba, 4, 4, 16).expect("valid buffer should construct");
    let rgba_result =
        gradient_based_error_diffusion_in_place(&mut rgba_buffer, QuantizeMode::gray_bits(1));
    assert!(matches!(
        rgba_result,
        Err(Error::UnsupportedFormat(
            "variable diffusion algorithms support grayscale formats only"
        ))
    ));
}

#[test]
fn diffusion_engine_rgb_palette_output_is_palette_member() {
    let mut data = vec![
        16, 24, 32, 48, 56, 64, 80, 88, 96, 112, 120, 128, 144, 152, 160, 176, 184, 192, 208, 216,
        224, 240, 248, 255, 12, 68, 124, 28, 84, 140, 44, 100, 156, 60, 116, 172, 76, 132, 188, 92,
        148, 204, 108, 164, 220, 124, 180, 236,
    ];
    let palette = Palette::new(vec![
        [0_u8, 0, 0],
        [255_u8, 255, 255],
        [255_u8, 0, 0],
        [0_u8, 255, 0],
        [0_u8, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = dithr::rgb_u8(&mut data, 4, 4, 12).expect("valid buffer should construct");

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
    let mut buffer = dithr::rgba_u8(&mut data, 6, 5, 24).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::rgb_bits(2))
        .expect("floyd-steinberg should succeed");

    let after_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_alpha, after_alpha);
}

#[test]
fn diffusion_engine_does_not_panic_on_1x1() {
    let mut data = vec![128_u8];
    let mut buffer = dithr::gray_u8(&mut data, 1, 1, 1).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::gray_bits(1))
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

    let mut buffer_a = dithr::rgb_u8(&mut a, 8, 6, 24).expect("valid buffer should construct");
    let mut buffer_b = dithr::rgb_u8(&mut b, 8, 6, 24).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer_a, QuantizeMode::rgb_bits(3))
        .expect("floyd-steinberg should succeed");
    floyd_steinberg_in_place(&mut buffer_b, QuantizeMode::rgb_bits(3))
        .expect("floyd-steinberg should succeed");

    assert_eq!(a, b);
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
fn diffusion_u16_every_wrapper_smoke_gray() {
    let wrappers: [DiffusionWrapperU16; 16] = [
        floyd_steinberg_in_place,
        false_floyd_steinberg_in_place,
        jarvis_judice_ninke_in_place,
        stucki_in_place,
        burkes_in_place,
        sierra_in_place,
        two_row_sierra_in_place,
        sierra_lite_in_place,
        stevenson_arce_in_place,
        atkinson_in_place,
        fan_in_place,
        shiau_fan_in_place,
        shiau_fan_2_in_place,
        ostromoukhov_in_place,
        zhou_fang_in_place,
        gradient_based_error_diffusion_in_place,
    ];

    for wrapper in wrappers {
        let mut data: Vec<u16> = (0_u32..256)
            .map(|value| ((value * 257) % 65_536) as u16)
            .collect();
        let mut buffer =
            dithr::gray_u16(&mut data, 16, 16, 16).expect("valid buffer should construct");

        wrapper(&mut buffer, QuantizeMode::GrayLevels(2)).expect("u16 wrapper should succeed");
        assert!(data.iter().all(|&value| value == 0 || value == 65_535));
    }
}

#[test]
fn diffusion_f32_every_wrapper_smoke_gray() {
    let classic_extended_wrappers: [DiffusionWrapperF32; 13] = [
        floyd_steinberg_in_place,
        false_floyd_steinberg_in_place,
        jarvis_judice_ninke_in_place,
        stucki_in_place,
        burkes_in_place,
        sierra_in_place,
        two_row_sierra_in_place,
        sierra_lite_in_place,
        stevenson_arce_in_place,
        atkinson_in_place,
        fan_in_place,
        shiau_fan_in_place,
        shiau_fan_2_in_place,
    ];

    for wrapper in classic_extended_wrappers {
        let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
            .map(|value| (value % 256) as f32 / 255.0)
            .collect();
        let mut buffer =
            dithr::rgb_32f(&mut data, 16, 16, 48).expect("valid buffer should construct");

        wrapper(&mut buffer, QuantizeMode::GrayLevels(2)).expect("f32 wrapper should succeed");
        assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
    }

    let variable_wrappers: [DiffusionWrapperF32; 3] = [
        ostromoukhov_in_place,
        zhou_fang_in_place,
        gradient_based_error_diffusion_in_place,
    ];

    for wrapper in variable_wrappers {
        let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
            .map(|value| (value % 256) as f32 / 255.0)
            .collect();
        let mut buffer =
            dithr::rgb_32f(&mut data, 16, 16, 48).expect("valid buffer should construct");

        let result = wrapper(&mut buffer, QuantizeMode::GrayLevels(2));
        assert!(matches!(
            result,
            Err(Error::UnsupportedFormat(
                "variable diffusion algorithms support grayscale formats only"
            ))
        ));
    }
}

#[test]
fn diffusion_u16_binary_output_for_gray_levels_2() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 16, 16, 16).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn diffusion_f32_binary_output_for_gray_levels_2() {
    let mut data: Vec<f32> = (0_u32..256).map(|value| value as f32 / 255.0).collect();
    let mut buffer = dithr::gray_32f(&mut data, 16, 16, 16).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

#[test]
fn diffusion_alpha_preserved_all_sample_types() {
    let mut rgba_u8: Vec<u8> = (0_u16..80)
        .map(|value| ((value * 19 + 11) % 256) as u8)
        .collect();
    let before_u8: Vec<u8> = rgba_u8.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_u8 =
        dithr::rgba_u8(&mut rgba_u8, 4, 5, 16).expect("valid buffer should construct");
    floyd_steinberg_in_place(&mut buffer_u8, QuantizeMode::RgbLevels(2))
        .expect("u8 diffusion should succeed");
    let after_u8: Vec<u8> = rgba_u8.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_u8, after_u8);

    let mut rgba_u16: Vec<u16> = (0_u32..80)
        .map(|value| ((value * 977) % 65_536) as u16)
        .collect();
    let before_u16: Vec<u16> = rgba_u16.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_u16 =
        dithr::rgba_u16(&mut rgba_u16, 4, 5, 16).expect("valid buffer should construct");
    floyd_steinberg_in_place(&mut buffer_u16, QuantizeMode::RgbLevels(2))
        .expect("u16 diffusion should succeed");
    let after_u16: Vec<u16> = rgba_u16.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_u16, after_u16);

    let mut rgba_f32: Vec<f32> = (0_u32..80).map(|value| value as f32 / 79.0).collect();
    let before_f32: Vec<f32> = rgba_f32.iter().skip(3).step_by(4).copied().collect();
    let mut buffer_f32 =
        dithr::rgba_32f(&mut rgba_f32, 4, 5, 16).expect("valid buffer should construct");
    floyd_steinberg_in_place(&mut buffer_f32, QuantizeMode::RgbLevels(2))
        .expect("f32 diffusion should succeed");
    let after_f32: Vec<f32> = rgba_f32.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_f32, after_f32);
}

#[test]
fn diffusion_same_kernel_invariants_all_sample_types() {
    let mut gray_u8 = gray_ramp_16x16();
    let mut gray_u16: Vec<u16> = gray_u8.iter().map(|&v| u16::from(v) * 257).collect();
    let mut gray_f32: Vec<f32> = gray_u8.iter().map(|&v| f32::from(v) / 255.0).collect();

    let mut buffer_u8 =
        dithr::gray_u8(&mut gray_u8, 16, 16, 16).expect("valid buffer should construct");
    let mut buffer_u16 =
        dithr::gray_u16(&mut gray_u16, 16, 16, 16).expect("valid buffer should construct");
    let mut buffer_f32 =
        dithr::gray_32f(&mut gray_f32, 16, 16, 16).expect("valid buffer should construct");

    burkes_in_place(&mut buffer_u8, QuantizeMode::GrayLevels(2)).expect("u8 run should succeed");
    burkes_in_place(&mut buffer_u16, QuantizeMode::GrayLevels(2)).expect("u16 run should succeed");
    burkes_in_place(&mut buffer_f32, QuantizeMode::GrayLevels(2)).expect("f32 run should succeed");

    assert!(gray_u8.iter().all(|&value| value == 0 || value == 255));
    assert!(gray_u16.iter().all(|&value| value == 0 || value == 65_535));
    assert!(gray_f32.iter().all(|&value| value == 0.0 || value == 1.0));
}

#[test]
fn variable_diffusion_coeff_range_post_cleanup() {
    let coeffs = &dithr::data::OSTROMOUKHOV_COEFFS;
    assert_eq!(coeffs.len(), 256);
    for &(forward, down_diag, down, den) in coeffs {
        assert!(forward >= 0);
        assert!(down_diag >= 0);
        assert!(down >= 0);
        assert!(den > 0);
        assert!(i32::from(forward) <= i32::from(den));
        assert!(i32::from(down_diag) <= i32::from(den));
        assert!(i32::from(down) <= i32::from(den));
    }

    let modulation = &dithr::data::ZHOU_FANG_MODULATION;
    assert_eq!(modulation.len(), 256);
    assert_eq!(modulation[0], 0);
    assert!(modulation[128] >= modulation[96]);
    assert!(modulation[128] >= modulation[160]);
    assert!(modulation.iter().all(|&entry| entry <= 199));
}

#[test]
fn classic_diffusion_no_separate_int_float_engine_smoke() {
    let mut gray_u8 = gray_ramp_16x16();
    let mut gray_f32: Vec<f32> = gray_u8.iter().map(|&v| f32::from(v) / 255.0).collect();

    let mut buffer_u8 =
        dithr::gray_u8(&mut gray_u8, 16, 16, 16).expect("valid buffer should construct");
    let mut buffer_f32 =
        dithr::gray_32f(&mut gray_f32, 16, 16, 16).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer_u8, QuantizeMode::GrayLevels(2))
        .expect("u8 run should succeed");
    floyd_steinberg_in_place(&mut buffer_f32, QuantizeMode::GrayLevels(2))
        .expect("f32 run should succeed");

    let mask_u8: Vec<u8> = gray_u8.iter().map(|&v| u8::from(v > 127)).collect();
    let mask_f32: Vec<u8> = gray_f32.iter().map(|&v| u8::from(v > 0.5)).collect();
    assert_eq!(mask_u8, mask_f32);
}

#[test]
fn diffusion_public_api_packed_constructor_smoke() {
    let mut data = gray_ramp_8x8();
    let mut buffer =
        dithr::gray_u8_packed(&mut data, 8, 8).expect("valid packed gray buffer should construct");
    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("floyd-steinberg should succeed");
    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn ostromoukhov_coeff_index_range_test() {
    for luma in 0_u16..=255 {
        let index = usize::from(luma as u8);
        assert!(index < dithr::data::OSTROMOUKHOV_COEFFS.len());
    }
}

#[test]
fn zhou_fang_coeff_index_range_test() {
    let len = dithr::data::ZHOU_FANG_MODULATION.len();
    for luma in 0_u16..=255 {
        let index = (usize::from(luma as u8) * len) / 256;
        assert!(index < len);
    }
}
