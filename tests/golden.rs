use dithr::{
    bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place,
    cluster_dot_4x4_in_place, random_in_place, threshold_in_place, Buffer, PixelFormat,
    QuantizeMode,
};

#[test]
fn golden_threshold_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    threshold_in_place(&mut buffer, QuantizeMode::GrayBits(1), 127);

    assert_eq!(fnv1a64(&data), 4_864_876_028_568_798_213_u64);
}

#[test]
fn golden_random_seed_1_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    random_in_place(&mut buffer, QuantizeMode::GrayBits(1), 1, 64);

    assert_eq!(fnv1a64(&data), 4_707_737_849_936_150_024_u64);
}

#[test]
fn golden_bayer_2x2_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    bayer_2x2_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(fnv1a64(&data), 5_176_068_339_558_256_461_u64);
}

#[test]
fn golden_bayer_4x4_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    bayer_4x4_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(fnv1a64(&data), 5_176_068_339_558_256_461_u64);
}

#[test]
fn golden_bayer_8x8_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(fnv1a64(&data), 1_956_760_498_679_199_251_u64);
}

#[test]
fn golden_bayer_16x16_gray_ramp_16x16() {
    let mut data = gray_ramp_16x16();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    bayer_16x16_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(fnv1a64(&data), 13_072_875_211_936_825_827_u64);
}

#[test]
fn golden_cluster_dot_4x4_gray_ramp_8x8() {
    let mut data = gray_ramp_8x8();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    cluster_dot_4x4_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert_eq!(fnv1a64(&data), 9_783_687_876_575_450_447_u64);
}

fn gray_ramp_8x8() -> Vec<u8> {
    (0_u16..64).map(|value| (value * 4) as u8).collect()
}

fn gray_ramp_16x16() -> Vec<u8> {
    (0_u16..256).map(|value| value as u8).collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;

    for &value in bytes {
        hash ^= u64::from(value);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3_u64);
    }

    hash
}
