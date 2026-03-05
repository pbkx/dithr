#[must_use]
pub fn mul_div_i32(value: i32, numer: i32, denom: i32) -> i32 {
    assert!(denom != 0, "denominator must not be zero");

    let product = i64::from(value) * i64::from(numer);
    let quotient = product / i64::from(denom);

    i32::try_from(quotient).expect("mul_div_i32 overflow")
}

#[must_use]
pub fn round_shift_i32(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        return value;
    }

    assert!(shift < 32, "shift must be less than 32");

    let half = 1_i64 << (shift - 1);
    let adjusted = if value >= 0 {
        i64::from(value) + half
    } else {
        i64::from(value) - half
    };
    let shifted = adjusted >> shift;

    i32::try_from(shifted).expect("round_shift_i32 overflow")
}

#[cfg(test)]
mod tests {
    use super::mul_div_i32;

    #[test]
    fn mul_div_i32_matches_simple_cases() {
        assert_eq!(mul_div_i32(10, 3, 2), 15);
        assert_eq!(mul_div_i32(7, 2, 3), 4);
        assert_eq!(mul_div_i32(-10, 3, 2), -15);
        assert_eq!(mul_div_i32(10, -3, 2), -15);
    }
}
