use super::{layout::PixelLayout, sample::Sample};

pub fn read_unit_pixel<S: Sample, L: PixelLayout>(pixel: &[S]) -> [f32; 4] {
    if L::IS_GRAY {
        let gray = pixel.first().copied().map(S::to_unit_f32).unwrap_or(0.0);
        return [gray, gray, gray, 1.0];
    }

    let mut out = [0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32];

    for (channel, dst) in out.iter_mut().take(3).enumerate() {
        if let Some(&sample) = pixel.get(channel) {
            *dst = sample.to_unit_f32();
        }
    }

    if let Some(alpha_lane) = alpha_index::<L>() {
        out[3] = pixel
            .get(alpha_lane)
            .copied()
            .map(S::to_unit_f32)
            .unwrap_or(1.0);
    }

    out
}

pub fn write_unit_pixel<S: Sample, L: PixelLayout>(pixel: &mut [S], rgba: [f32; 4]) {
    if L::IS_GRAY {
        let gray = 0.299_f32 * rgba[0] + 0.587_f32 * rgba[1] + 0.114_f32 * rgba[2];
        if let Some(slot) = pixel.first_mut() {
            *slot = S::from_unit_f32(gray);
        }
        return;
    }

    for (channel, &value) in rgba.iter().take(3).enumerate() {
        if let Some(slot) = pixel.get_mut(channel) {
            *slot = S::from_unit_f32(value);
        }
    }

    if let Some(alpha_lane) = alpha_index::<L>() {
        if let Some(slot) = pixel.get_mut(alpha_lane) {
            *slot = S::from_unit_f32(rgba[3]);
        }
    }
}

pub fn alpha_index<L: PixelLayout>() -> Option<usize> {
    if L::HAS_ALPHA {
        L::CHANNELS.checked_sub(1)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::{read_unit_pixel, write_unit_pixel};
    use crate::core::layout::{Gray, Rgb, Rgba};

    #[test]
    fn sample_f32_clamps_on_write() {
        let mut px = [0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32];
        write_unit_pixel::<f32, Rgba>(&mut px, [-1.0, 0.5, 2.0, 4.0]);
        assert_eq!(px, [0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn gray_read_unit_pixel_expands_gray_to_rgb() {
        let px = [64_u8];
        let rgba = read_unit_pixel::<u8, Gray>(&px);
        assert_eq!(rgba[0], rgba[1]);
        assert_eq!(rgba[1], rgba[2]);
        assert_eq!(rgba[3], 1.0);
    }

    #[test]
    fn rgb_read_unit_pixel_sets_alpha_to_1() {
        let px = [10_u8, 20_u8, 30_u8];
        let rgba = read_unit_pixel::<u8, Rgb>(&px);
        assert_eq!(rgba[3], 1.0);
    }

    #[test]
    fn rgba_write_unit_pixel_preserves_alpha_lane() {
        let mut px = [0_u8, 0_u8, 0_u8, 0_u8];
        write_unit_pixel::<u8, Rgba>(&mut px, [0.25, 0.5, 0.75, 0.8]);
        assert_eq!(px[3], 204);
    }
}
