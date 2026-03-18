#[must_use]
pub fn mul_div_i32(value: i32, numer: i32, denom: i32) -> i32 {
    if denom == 0 {
        return 0;
    }

    let product = i64::from(value) * i64::from(numer);
    let quotient = product / i64::from(denom);
    let max = i64::from(i32::MAX);
    let min = i64::from(i32::MIN);

    if quotient > max {
        i32::MAX
    } else if quotient < min {
        i32::MIN
    } else {
        quotient as i32
    }
}

#[must_use]
pub fn round_shift_i32(value: i32, shift: u32) -> i32 {
    if shift == 0 {
        return value;
    }

    if shift >= 32 {
        return if value >= 0 { 0 } else { -1 };
    }

    let half = 1_i64 << (shift - 1);
    let adjusted = if value >= 0 {
        i64::from(value) + half
    } else {
        i64::from(value) - half
    };
    let shifted = adjusted >> shift;
    let max = i64::from(i32::MAX);
    let min = i64::from(i32::MIN);

    if shifted > max {
        i32::MAX
    } else if shifted < min {
        i32::MIN
    } else {
        shifted as i32
    }
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
