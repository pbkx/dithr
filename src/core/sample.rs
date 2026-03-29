pub trait Sample: Copy + Send + Sync + 'static {
    const IS_FLOAT: bool;

    fn to_unit_f32(self) -> f32;
    fn from_unit_f32(x: f32) -> Self;
}

pub trait SampleMath: Sample {
    fn diff_unit_f32(a: Self, b: Self) -> f32;
}

impl Sample for u8 {
    const IS_FLOAT: bool = false;

    fn to_unit_f32(self) -> f32 {
        f32::from(self) / 255.0
    }

    fn from_unit_f32(x: f32) -> Self {
        (x.clamp(0.0, 1.0) * 255.0).round() as u8
    }
}

impl SampleMath for u8 {
    fn diff_unit_f32(a: Self, b: Self) -> f32 {
        a.to_unit_f32() - b.to_unit_f32()
    }
}

impl Sample for u16 {
    const IS_FLOAT: bool = false;

    fn to_unit_f32(self) -> f32 {
        f32::from(self) / 65_535.0
    }

    fn from_unit_f32(x: f32) -> Self {
        (x.clamp(0.0, 1.0) * 65_535.0).round() as u16
    }
}

impl SampleMath for u16 {
    fn diff_unit_f32(a: Self, b: Self) -> f32 {
        a.to_unit_f32() - b.to_unit_f32()
    }
}

impl Sample for f32 {
    const IS_FLOAT: bool = true;

    fn to_unit_f32(self) -> f32 {
        self
    }

    fn from_unit_f32(x: f32) -> Self {
        x.clamp(0.0, 1.0)
    }
}

impl SampleMath for f32 {
    fn diff_unit_f32(a: Self, b: Self) -> f32 {
        a - b
    }
}

#[cfg(test)]
mod tests {
    use super::Sample;

    #[test]
    fn sample_u8_roundtrip_endpoints() {
        assert_eq!(u8::from_unit_f32(0.0), 0);
        assert_eq!(u8::from_unit_f32(1.0), 255);
        assert_eq!(0_u8.to_unit_f32(), 0.0);
        assert_eq!(255_u8.to_unit_f32(), 1.0);
    }

    #[test]
    fn sample_u16_roundtrip_endpoints() {
        assert_eq!(u16::from_unit_f32(0.0), 0);
        assert_eq!(u16::from_unit_f32(1.0), 65_535);
        assert_eq!(0_u16.to_unit_f32(), 0.0);
        assert_eq!(65_535_u16.to_unit_f32(), 1.0);
    }
}
