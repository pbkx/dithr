#![allow(dead_code)]

pub fn gray_ramp_8x8_u8() -> Vec<u8> {
    (0_u16..64).map(|value| (value * 4) as u8).collect()
}

pub fn gray_ramp_16x16_u8() -> Vec<u8> {
    (0_u16..256).map(|value| value as u8).collect()
}

pub fn gray_ramp_8x8_u16() -> Vec<u16> {
    (0_u32..64)
        .map(|value| ((value * 65_535) / 63) as u16)
        .collect()
}

pub fn checker_8x8() -> Vec<u8> {
    let mut out = Vec::with_capacity(64);

    for y in 0..8_usize {
        for x in 0..8_usize {
            out.push(if (x + y) % 2 == 0 { 0 } else { 255 });
        }
    }

    out
}

pub fn rgb_gradient_8x8_u8() -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * 8 * 3);

    for y in 0..8_u8 {
        for x in 0..8_u8 {
            out.push(x.saturating_mul(32));
            out.push(y.saturating_mul(32));
            out.push((x ^ y).saturating_mul(32));
        }
    }

    out
}

pub fn rgb_gradient_8x8_u16() -> Vec<u16> {
    let mut out = Vec::with_capacity(8 * 8 * 3);

    for y in 0_u16..8 {
        for x in 0_u16..8 {
            out.push((u32::from(x) * 65_535 / 7) as u16);
            out.push((u32::from(y) * 65_535 / 7) as u16);
            out.push((u32::from(x ^ y) * 65_535 / 7) as u16);
        }
    }

    out
}

pub fn rgb_gradient_8x8_f32() -> Vec<f32> {
    let mut out = Vec::with_capacity(8 * 8 * 3);

    for y in 0_u16..8 {
        for x in 0_u16..8 {
            out.push(f32::from(x) / 7.0);
            out.push(f32::from(y) / 7.0);
            out.push(f32::from(x ^ y) / 7.0);
        }
    }

    out
}

pub fn gray_ramp_8x8() -> Vec<u8> {
    gray_ramp_8x8_u8()
}

pub fn gray_ramp_16x16() -> Vec<u8> {
    gray_ramp_16x16_u8()
}

pub fn rgb_gradient_8x8() -> Vec<u8> {
    rgb_gradient_8x8_u8()
}

pub fn rgb_cube_strip() -> Vec<u8> {
    let mut out = Vec::with_capacity(27 * 3);
    let levels = [0_u8, 127_u8, 255_u8];

    for &r in &levels {
        for &g in &levels {
            for &b in &levels {
                out.push(r);
                out.push(g);
                out.push(b);
            }
        }
    }

    out
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;

    for &value in bytes {
        hash ^= u64::from(value);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3_u64);
    }

    hash
}
