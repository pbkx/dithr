#[must_use]
pub fn rgb_distance_sq(a: [u8; 3], b: [u8; 3]) -> u32 {
    let dr = i32::from(a[0]) - i32::from(b[0]);
    let dg = i32::from(a[1]) - i32::from(b[1]);
    let db = i32::from(a[2]) - i32::from(b[2]);

    (dr * dr + dg * dg + db * db) as u32
}

#[must_use]
pub fn luma_u8(rgb: [u8; 3]) -> u8 {
    ((77_u32 * u32::from(rgb[0]) + 150_u32 * u32::from(rgb[1]) + 29_u32 * u32::from(rgb[2]) + 128)
        >> 8) as u8
}

#[cfg(test)]
mod tests {
    use super::{luma_u8, rgb_distance_sq};

    #[test]
    fn rgb_distance_sq_zero_for_equal() {
        assert_eq!(rgb_distance_sq([12, 34, 56], [12, 34, 56]), 0);
    }

    #[test]
    fn luma_u8_monotonic_for_gray() {
        let mut prev = 0_u8;

        for value in 0_u16..=255 {
            let gray = value as u8;
            let y = luma_u8([gray, gray, gray]);
            assert!(y >= prev);
            prev = y;
        }

        assert_eq!(luma_u8([0, 0, 0]), 0);
        assert_eq!(luma_u8([255, 255, 255]), 255);
    }
}
