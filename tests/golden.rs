use dithr::{threshold_in_place, Buffer, PixelFormat, QuantizeMode};

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

fn gray_ramp_8x8() -> Vec<u8> {
    (0_u16..64).map(|value| (value * 4) as u8).collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;

    for &value in bytes {
        hash ^= u64::from(value);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3_u64);
    }

    hash
}
