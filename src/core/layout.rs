pub trait PixelLayout: Copy + Send + Sync + 'static {
    const CHANNELS: usize;
    const COLOR_CHANNELS: usize;
    const HAS_ALPHA: bool;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Gray;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rgb;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rgba;

impl PixelLayout for Gray {
    const CHANNELS: usize = 1;
    const COLOR_CHANNELS: usize = 1;
    const HAS_ALPHA: bool = false;
}

impl PixelLayout for Rgb {
    const CHANNELS: usize = 3;
    const COLOR_CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
}

impl PixelLayout for Rgba {
    const CHANNELS: usize = 4;
    const COLOR_CHANNELS: usize = 3;
    const HAS_ALPHA: bool = true;
}

#[cfg(test)]
mod tests {
    use super::{Gray, PixelLayout, Rgb, Rgba};

    #[test]
    fn layout_constants_match_expected() {
        assert_eq!(Gray::CHANNELS, 1);
        assert_eq!(Gray::COLOR_CHANNELS, 1);
        assert!(!std::hint::black_box(Gray::HAS_ALPHA));

        assert_eq!(Rgb::CHANNELS, 3);
        assert_eq!(Rgb::COLOR_CHANNELS, 3);
        assert!(!std::hint::black_box(Rgb::HAS_ALPHA));

        assert_eq!(Rgba::CHANNELS, 4);
        assert_eq!(Rgba::COLOR_CHANNELS, 3);
        assert!(std::hint::black_box(Rgba::HAS_ALPHA));
    }
}
