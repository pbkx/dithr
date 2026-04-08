pub trait PixelLayout: Copy + Send + Sync + 'static {
    const CHANNELS: usize;
    const COLOR_CHANNELS: usize;
    const HAS_ALPHA: bool;
    const IS_GRAY: bool;
}

pub(crate) fn validate_layout_invariants<L: PixelLayout>() -> crate::Result<()> {
    if L::CHANNELS == 0 {
        return Err(crate::Error::UnsupportedFormat(
            "pixel layout must define at least one channel",
        ));
    }

    if L::COLOR_CHANNELS == 0 {
        return Err(crate::Error::UnsupportedFormat(
            "pixel layout must define at least one color channel",
        ));
    }

    if L::COLOR_CHANNELS > L::CHANNELS {
        return Err(crate::Error::UnsupportedFormat(
            "pixel layout color channels cannot exceed total channels",
        ));
    }

    if L::HAS_ALPHA && L::COLOR_CHANNELS == L::CHANNELS {
        return Err(crate::Error::UnsupportedFormat(
            "alpha pixel layouts must reserve a non-color channel",
        ));
    }

    if L::IS_GRAY {
        if L::COLOR_CHANNELS != 1 {
            return Err(crate::Error::UnsupportedFormat(
                "gray pixel layouts must use exactly one color channel",
            ));
        }
        if L::HAS_ALPHA {
            return Err(crate::Error::UnsupportedFormat(
                "gray pixel layouts with alpha are not supported",
            ));
        }
    } else if L::COLOR_CHANNELS < 3 {
        return Err(crate::Error::UnsupportedFormat(
            "color pixel layouts must define at least three color channels",
        ));
    }

    Ok(())
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
    const IS_GRAY: bool = true;
}

impl PixelLayout for Rgb {
    const CHANNELS: usize = 3;
    const COLOR_CHANNELS: usize = 3;
    const HAS_ALPHA: bool = false;
    const IS_GRAY: bool = false;
}

impl PixelLayout for Rgba {
    const CHANNELS: usize = 4;
    const COLOR_CHANNELS: usize = 3;
    const HAS_ALPHA: bool = true;
    const IS_GRAY: bool = false;
}

#[cfg(test)]
mod tests {
    use super::{validate_layout_invariants, Gray, PixelLayout, Rgb, Rgba};
    use crate::Error;

    #[test]
    fn layout_constants_match_expected() {
        assert_eq!(Gray::CHANNELS, 1);
        assert_eq!(Gray::COLOR_CHANNELS, 1);
        assert!(!std::hint::black_box(Gray::HAS_ALPHA));
        assert!(std::hint::black_box(Gray::IS_GRAY));

        assert_eq!(Rgb::CHANNELS, 3);
        assert_eq!(Rgb::COLOR_CHANNELS, 3);
        assert!(!std::hint::black_box(Rgb::HAS_ALPHA));
        assert!(!std::hint::black_box(Rgb::IS_GRAY));

        assert_eq!(Rgba::CHANNELS, 4);
        assert_eq!(Rgba::COLOR_CHANNELS, 3);
        assert!(std::hint::black_box(Rgba::HAS_ALPHA));
        assert!(!std::hint::black_box(Rgba::IS_GRAY));
    }

    #[derive(Clone, Copy)]
    struct InvalidZeroChannels;

    impl PixelLayout for InvalidZeroChannels {
        const CHANNELS: usize = 0;
        const COLOR_CHANNELS: usize = 1;
        const HAS_ALPHA: bool = false;
        const IS_GRAY: bool = true;
    }

    #[derive(Clone, Copy)]
    struct InvalidColorOverflow;

    impl PixelLayout for InvalidColorOverflow {
        const CHANNELS: usize = 3;
        const COLOR_CHANNELS: usize = 4;
        const HAS_ALPHA: bool = false;
        const IS_GRAY: bool = false;
    }

    #[test]
    fn validate_layout_invariants_accepts_builtin_layouts() {
        assert_eq!(validate_layout_invariants::<Gray>(), Ok(()));
        assert_eq!(validate_layout_invariants::<Rgb>(), Ok(()));
        assert_eq!(validate_layout_invariants::<Rgba>(), Ok(()));
    }

    #[test]
    fn validate_layout_invariants_rejects_zero_channels() {
        assert_eq!(
            validate_layout_invariants::<InvalidZeroChannels>(),
            Err(Error::UnsupportedFormat(
                "pixel layout must define at least one channel"
            ))
        );
    }

    #[test]
    fn validate_layout_invariants_rejects_color_overflow() {
        assert_eq!(
            validate_layout_invariants::<InvalidColorOverflow>(),
            Err(Error::UnsupportedFormat(
                "pixel layout color channels cannot exceed total channels"
            ))
        );
    }
}
