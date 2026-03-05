#[must_use]
pub fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[must_use]
pub fn clamp_i16(v: i32) -> i16 {
    v.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::{clamp_i16, clamp_u8};

    #[test]
    fn clamp_u8_bounds() {
        assert_eq!(clamp_u8(-1), 0);
        assert_eq!(clamp_u8(0), 0);
        assert_eq!(clamp_u8(255), 255);
        assert_eq!(clamp_u8(256), 255);
    }

    #[test]
    fn clamp_i16_bounds() {
        assert_eq!(clamp_i16(i32::from(i16::MIN) - 1), i16::MIN);
        assert_eq!(clamp_i16(i32::from(i16::MAX) + 1), i16::MAX);
        assert_eq!(clamp_i16(1234), 1234_i16);
    }
}
