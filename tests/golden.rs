mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_8x8, gray_ramp_8x8_u16, rgb_cube_strip,
    rgb_gradient_8x8, rgb_gradient_8x8_f32,
};
use dithr::{
    atkinson_in_place, bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place,
    bayer_8x8_in_place, burkes_in_place, cluster_dot_4x4_in_place, cluster_dot_8x8_in_place,
    custom_ordered_in_place, direct_binary_search_in_place, electrostatic_halftoning_in_place,
    false_floyd_steinberg_in_place, fan_in_place, floyd_steinberg_in_place,
    gradient_based_error_diffusion_in_place, jarvis_judice_ninke_in_place,
    knuth_dot_diffusion_in_place, lattice_boltzmann_in_place, ostromoukhov_in_place,
    random_binary_in_place, riemersma_in_place, shiau_fan_2_in_place, shiau_fan_in_place,
    sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place, stucki_in_place,
    threshold_binary_in_place, two_row_sierra_in_place, yliluoma_1_in_place, yliluoma_2_in_place,
    yliluoma_3_in_place, zhou_fang_in_place, Palette, QuantizeMode,
};

fn variable_gray_challenge_64x64() -> Vec<u8> {
    let mut out = Vec::with_capacity(64 * 64);

    for y in 0..64_usize {
        for x in 0..64_usize {
            out.push(((x * 17 + y * 31 + ((x * y) % 97)) % 256) as u8);
        }
    }

    out
}

#[test]
fn golden_fixture_hashes() {
    let checker = checker_8x8();
    let cube = rgb_cube_strip();

    assert_eq!(fnv1a64(&checker), 1_053_467_341_045_678_885_u64);
    assert_eq!(fnv1a64(&cube), 13_513_412_517_116_417_213_u64);
}

#[test]
fn golden_threshold_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    threshold_binary_in_place(&mut buffer, QuantizeMode::GrayBits(1), 127)
        .expect("threshold binary should succeed");

    assert_eq!(fnv1a64(&data), 4_864_876_028_568_798_213_u64);
}

#[test]
fn golden_random_seed_1_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    random_binary_in_place(&mut buffer, QuantizeMode::GrayBits(1), 1, 64)
        .expect("random binary should succeed");

    assert_eq!(fnv1a64(&data), 4_707_737_849_936_150_024_u64);
}

#[test]
fn golden_bayer_2x2_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    bayer_2x2_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("bayer 2x2 should succeed");

    assert_eq!(fnv1a64(&data), 5_176_068_339_558_256_461_u64);
}

#[test]
fn golden_bayer_4x4_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_4x4_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("bayer 4x4 should succeed");

    assert_eq!(fnv1a64(&data), 11_223_927_337_015_380_774_u64);
}

#[test]
fn golden_bayer_8x8_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("bayer 8x8 should succeed");

    assert_eq!(fnv1a64(&data), 1_956_760_498_679_199_251_u64);
}

#[test]
fn golden_bayer_16x16_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    bayer_16x16_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("bayer 16x16 should succeed");

    assert_eq!(fnv1a64(&data), 13_072_875_211_936_825_827_u64);
}

#[test]
fn golden_cluster_dot_4x4_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    cluster_dot_4x4_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("cluster-dot 4x4 should succeed");

    assert_eq!(fnv1a64(&data), 9_783_687_876_575_450_447_u64);
}

#[test]
fn golden_cluster_dot_8x8_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    cluster_dot_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("cluster-dot 8x8 should succeed");

    assert_eq!(fnv1a64(&data), 15_436_130_700_200_729_221_u64);
}

#[test]
fn golden_custom_ordered_2x2_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");
    let map = [0_u8, 2, 3, 1];

    custom_ordered_in_place(&mut buffer, QuantizeMode::GrayBits(1), &map, 2, 2, 64)
        .expect("custom ordered dither should succeed");

    assert_eq!(fnv1a64(&data), 5_176_068_339_558_256_461_u64);
}

#[test]
fn golden_yliluoma_1_rgb_gradient_8x8() {
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

    assert_eq!(fnv1a64(&data), 15_541_327_241_764_811_552_u64);
}

#[test]
fn golden_yliluoma_2_rgb_gradient_8x8() {
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

    assert_eq!(fnv1a64(&data), 6_371_937_729_658_429_102_u64);
}

#[test]
fn golden_yliluoma_3_rgb_gradient_8x8() {
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

    assert_eq!(fnv1a64(&data), 9_812_579_000_523_236_581_u64);
}

#[test]
fn golden_floyd_steinberg_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("floyd-steinberg should succeed");

    assert_eq!(fnv1a64(&data), 6_646_466_914_246_654_362_u64);
}

#[test]
fn golden_false_floyd_steinberg_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    false_floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("false floyd-steinberg should succeed");

    assert_eq!(fnv1a64(&data), 9_957_759_496_808_609_139_u64);
}

#[test]
fn golden_jarvis_judice_ninke_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    jarvis_judice_ninke_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("jarvis-judice-ninke should succeed");

    assert_eq!(fnv1a64(&data), 5_117_112_180_964_174_573_u64);
}

#[test]
fn golden_stucki_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    stucki_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("stucki should succeed");

    assert_eq!(fnv1a64(&data), 12_830_851_821_987_011_482_u64);
}

#[test]
fn golden_burkes_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    burkes_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("burkes should succeed");

    assert_eq!(fnv1a64(&data), 15_615_790_824_215_818_469_u64);
}

#[test]
fn golden_sierra_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    sierra_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("sierra should succeed");

    assert_eq!(fnv1a64(&data), 6_017_484_232_485_597_234_u64);
}

#[test]
fn golden_two_row_sierra_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    two_row_sierra_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("two-row sierra should succeed");

    assert_eq!(fnv1a64(&data), 7_660_510_162_696_616_438_u64);
}

#[test]
fn golden_sierra_lite_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    sierra_lite_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("sierra lite should succeed");

    assert_eq!(fnv1a64(&data), 4_380_690_145_979_882_262_u64);
}

#[test]
fn golden_stevenson_arce_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    stevenson_arce_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("stevenson-arce should succeed");

    assert_eq!(fnv1a64(&data), 8_441_572_485_774_456_170_u64);
}

#[test]
fn golden_atkinson_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    atkinson_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("atkinson should succeed");

    assert_eq!(fnv1a64(&data), 3_764_225_652_723_977_986_u64);
}

#[test]
fn golden_fan_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    fan_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("fan should succeed");

    assert_eq!(fnv1a64(&data), 13_669_358_573_283_721_591_u64);
}

#[test]
fn golden_shiau_fan_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    shiau_fan_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("shiau-fan should succeed");

    assert_eq!(fnv1a64(&data), 12_932_304_124_049_052_197_u64);
}

#[test]
fn golden_shiau_fan_2_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    shiau_fan_2_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("shiau-fan-2 should succeed");

    assert_eq!(fnv1a64(&data), 11_882_711_592_529_338_042_u64);
}

#[test]
fn golden_ostromoukhov_gray_challenge_64x64() {
    let mut data = variable_gray_challenge_64x64();
    let mut buffer = dithr::gray_u8(&mut data, 64, 64, 64).expect("valid buffer should construct");

    ostromoukhov_in_place(&mut buffer, QuantizeMode::GrayBits(2))
        .expect("ostromoukhov should succeed");

    assert_eq!(fnv1a64(&data), 8_932_555_615_463_454_615_u64);
}

#[test]
fn golden_zhou_fang_gray_challenge_64x64() {
    let mut data = variable_gray_challenge_64x64();
    let mut buffer = dithr::gray_u8(&mut data, 64, 64, 64).expect("valid buffer should construct");

    zhou_fang_in_place(&mut buffer, QuantizeMode::GrayBits(2)).expect("zhou-fang should succeed");

    assert_eq!(fnv1a64(&data), 4_494_267_880_590_439_003_u64);
}

#[test]
fn golden_variable_diffusion_distinguishes_ostromoukhov_and_zhou_fang() {
    let mut ostromoukhov_data = variable_gray_challenge_64x64();
    let mut zhou_fang_data = ostromoukhov_data.clone();

    let mut ostromoukhov_buffer =
        dithr::gray_u8(&mut ostromoukhov_data, 64, 64, 64).expect("valid buffer should construct");
    let mut zhou_fang_buffer =
        dithr::gray_u8(&mut zhou_fang_data, 64, 64, 64).expect("valid buffer should construct");

    ostromoukhov_in_place(&mut ostromoukhov_buffer, QuantizeMode::GrayBits(2))
        .expect("ostromoukhov should succeed");
    zhou_fang_in_place(&mut zhou_fang_buffer, QuantizeMode::GrayBits(2))
        .expect("zhou-fang should succeed");

    assert_ne!(fnv1a64(&ostromoukhov_data), fnv1a64(&zhou_fang_data));
}

#[test]
fn golden_gradient_based_error_diffusion_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    gradient_based_error_diffusion_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("gradient-based diffusion should succeed");

    assert_eq!(fnv1a64(&data), 9_303_906_841_652_194_344_u64);
}

#[test]
fn golden_riemersma_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    riemersma_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("riemersma should succeed");

    assert_eq!(fnv1a64(&data), 4_759_045_697_198_729_208_u64);
}

#[test]
fn golden_knuth_dot_diffusion_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("knuth dot diffusion should succeed");

    assert_eq!(fnv1a64(&data), 14_706_985_719_462_917_693_u64);
}

#[test]
fn golden_dbs_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    direct_binary_search_in_place(&mut buffer, 4).expect("direct binary search should succeed");

    assert_eq!(fnv1a64(&data), 1_738_359_872_340_429_752_u64);
}

#[test]
fn golden_lattice_boltzmann_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    lattice_boltzmann_in_place(&mut buffer, 8).expect("lattice-boltzmann should succeed");

    assert_eq!(fnv1a64(&data), 4_864_876_028_568_798_213_u64);
}

#[test]
fn golden_electrostatic_halftoning_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    electrostatic_halftoning_in_place(&mut buffer, 10)
        .expect("electrostatic halftoning should succeed");

    assert_eq!(fnv1a64(&data), 1_985_050_605_501_357_403_u64);
}

#[test]
fn golden_bayer_8x8_gray_ramp_8x8_u16_binary_invariant() {
    let mut data = gray_ramp_8x8_u16();
    let mut buffer = dithr::gray_u16(&mut data, 8, 8, 8).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("bayer 8x8 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn golden_floyd_steinberg_gray_ramp_8x8_u16_binary_invariant() {
    let mut data = gray_ramp_8x8_u16();
    let mut buffer = dithr::gray_u16(&mut data, 8, 8, 8).expect("valid buffer should construct");

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn golden_bayer_8x8_rgb_gradient_8x8_f32_binary_invariant() {
    let mut data = rgb_gradient_8x8_f32();
    let mut buffer = dithr::rgb_f32(&mut data, 8, 8, 24).expect("valid buffer should construct");

    bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2)).expect("bayer 8x8 should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}
