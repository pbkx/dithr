use super::{layout::PixelLayout, sample::Sample};
use crate::{Error, Result};

pub fn read_unit_pixel<S: Sample, L: PixelLayout>(pixel: &[S]) -> Result<[f32; 4]> {
    if pixel.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }
    validate_layout_shape::<L>()?;

    if L::IS_GRAY {
        let gray = pixel[0].to_unit_f32();
        return Ok([gray, gray, gray, 1.0]);
    }

    let mut out = [0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32];

    for (channel, dst) in out.iter_mut().take(3).enumerate() {
        *dst = pixel[channel].to_unit_f32();
    }

    if let Some(alpha_lane) = alpha_index::<L>() {
        out[3] = pixel[alpha_lane].to_unit_f32();
    }

    Ok(out)
}

pub fn write_unit_pixel<S: Sample, L: PixelLayout>(pixel: &mut [S], rgba: [f32; 4]) -> Result<()> {
    if pixel.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }
    validate_layout_shape::<L>()?;

    if L::IS_GRAY {
        let gray = 0.299_f32 * rgba[0] + 0.587_f32 * rgba[1] + 0.114_f32 * rgba[2];
        pixel[0] = S::from_unit_f32(gray);
        return Ok(());
    }

    for (channel, &value) in rgba.iter().take(3).enumerate() {
        pixel[channel] = S::from_unit_f32(value);
    }

    if let Some(alpha_lane) = alpha_index::<L>() {
        pixel[alpha_lane] = S::from_unit_f32(rgba[3]);
    }

    Ok(())
}

pub fn alpha_index<L: PixelLayout>() -> Option<usize> {
    if L::HAS_ALPHA {
        L::CHANNELS.checked_sub(1)
    } else {
        None
    }
}

fn validate_layout_shape<L: PixelLayout>() -> Result<()> {
    if L::IS_GRAY {
        if L::CHANNELS == 0 {
            return Err(Error::UnsupportedFormat(
                "gray pixel layout must define at least one channel",
            ));
        }
        return Ok(());
    }

    if L::CHANNELS < 3 {
        return Err(Error::UnsupportedFormat(
            "color pixel layout must define at least three channels",
        ));
    }

    if L::HAS_ALPHA && L::CHANNELS < 4 {
        return Err(Error::UnsupportedFormat(
            "alpha pixel layout must define at least four channels",
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{read_unit_pixel, write_unit_pixel};
    use crate::{
        core::layout::{Gray, PixelLayout, Rgb, Rgba},
        Error,
    };

    #[derive(Clone, Copy)]
    struct GrayZero;

    impl PixelLayout for GrayZero {
        const CHANNELS: usize = 0;
        const COLOR_CHANNELS: usize = 1;
        const HAS_ALPHA: bool = false;
        const IS_GRAY: bool = true;
    }

    #[derive(Clone, Copy)]
    struct RgbTwo;

    impl PixelLayout for RgbTwo {
        const CHANNELS: usize = 2;
        const COLOR_CHANNELS: usize = 2;
        const HAS_ALPHA: bool = false;
        const IS_GRAY: bool = false;
    }

    #[test]
    fn sample_f32_clamps_on_write() {
        let mut px = [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];
        write_unit_pixel::<f32, Rgba>(&mut px, [-1.0, 0.5, 2.0, 4.0])
            .expect("pixel write should succeed");
        assert_eq!(px, [0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn gray_read_unit_pixel_expands_gray_to_rgb() {
        let px = [64_u8];
        let rgba = read_unit_pixel::<u8, Gray>(&px).expect("pixel read should succeed");
        assert_eq!(rgba[0], rgba[1]);
        assert_eq!(rgba[1], rgba[2]);
        assert_eq!(rgba[3], 1.0);
    }

    #[test]
    fn rgb_read_unit_pixel_sets_alpha_to_1() {
        let px = [10_u8, 20_u8, 30_u8];
        let rgba = read_unit_pixel::<u8, Rgb>(&px).expect("pixel read should succeed");
        assert_eq!(rgba[3], 1.0);
    }

    #[test]
    fn rgba_write_unit_pixel_preserves_alpha_lane() {
        let mut px = [0_u8, 0_u8, 0_u8, 0_u8];
        write_unit_pixel::<u8, Rgba>(&mut px, [0.25, 0.5, 0.75, 0.8])
            .expect("pixel write should succeed");
        assert_eq!(px[3], 204);
    }

    #[test]
    fn read_unit_pixel_rejects_gray_layout_without_channels() {
        let px: [u8; 0] = [];
        assert_eq!(
            read_unit_pixel::<u8, GrayZero>(&px),
            Err(Error::UnsupportedFormat(
                "gray pixel layout must define at least one channel",
            ))
        );
    }

    #[test]
    fn write_unit_pixel_rejects_color_layout_with_too_few_channels() {
        let mut px = [0_u8, 0_u8];
        assert_eq!(
            write_unit_pixel::<u8, RgbTwo>(&mut px, [0.1, 0.2, 0.3, 1.0]),
            Err(Error::UnsupportedFormat(
                "color pixel layout must define at least three channels",
            ))
        );
    }
}
