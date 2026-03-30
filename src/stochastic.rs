use crate::{
    core::{alpha_index, read_unit_pixel, PixelLayout, Sample},
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub fn threshold_binary_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    threshold: S,
) -> Result<()> {
    stochastic_in_place(buffer, mode, move |_, _| threshold.to_unit_f32())
}

pub fn random_binary_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    seed: u64,
    strength: u8,
) -> Result<()> {
    let mut prng = XorShift64::new(seed);
    stochastic_in_place(buffer, mode, move |x, y| {
        perturbed_threshold_unit(&mut prng, seed, x, y, strength)
    })
}

#[cfg(feature = "rayon")]
pub fn threshold_binary_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    threshold: S,
) -> Result<()> {
    stochastic_in_place_par(buffer, mode, move |_, _| threshold.to_unit_f32())
}

#[cfg(feature = "rayon")]
pub fn random_binary_in_place_par<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    seed: u64,
    strength: u8,
) -> Result<()> {
    buffer.validate()?;
    if !stochastic_layout_supported::<L>() {
        return Err(Error::UnsupportedFormat(
            "stochastic dithering supports Gray, Rgb, and Rgba layouts only",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;
    let threshold_len = width.checked_mul(height).ok_or(BufferError::OutOfBounds)?;
    let mut thresholds = vec![0.0_f32; threshold_len];
    let mut prng = XorShift64::new(seed);
    for y in 0..height {
        for x in 0..width {
            thresholds[y * width + x] = perturbed_threshold_unit(&mut prng, seed, x, y, strength);
        }
    }

    stochastic_in_place_par(buffer, mode, |x, y| thresholds[y * width + x])
}

pub fn threshold_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    threshold: S,
) -> Result<()> {
    threshold_binary_in_place(buffer, mode, threshold)
}

pub fn random_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    seed: u64,
    strength: u8,
) -> Result<()> {
    random_binary_in_place(buffer, mode, seed, strength)
}

fn stochastic_in_place<S: Sample, L: PixelLayout, F>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    mut threshold_for: F,
) -> Result<()>
where
    F: FnMut(usize, usize) -> f32,
{
    buffer.validate()?;
    if !stochastic_layout_supported::<L>() {
        return Err(Error::UnsupportedFormat(
            "stochastic dithering supports Gray, Rgb, and Rgba layouts only",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;

    for y in 0..height {
        let row = buffer.try_row_mut(y)?;
        for x in 0..width {
            let offset = x.checked_mul(L::CHANNELS).ok_or(BufferError::OutOfBounds)?;
            let end = offset
                .checked_add(L::CHANNELS)
                .ok_or(BufferError::OutOfBounds)?;
            let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
            stochastic_apply_pixel::<S, L>(pixel, mode, threshold_for(x, y))?;
        }
    }

    Ok(())
}

#[cfg(feature = "rayon")]
fn stochastic_in_place_par<S: Sample, L: PixelLayout, F>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    threshold_for: F,
) -> Result<()>
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    buffer.validate()?;
    if !stochastic_layout_supported::<L>() {
        return Err(Error::UnsupportedFormat(
            "stochastic dithering supports Gray, Rgb, and Rgba layouts only",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;
    let stride = buffer.stride;

    buffer
        .data
        .par_chunks_mut(stride)
        .take(height)
        .enumerate()
        .try_for_each(|(y, row)| -> Result<()> {
            for x in 0..width {
                let offset = x.checked_mul(L::CHANNELS).ok_or(BufferError::OutOfBounds)?;
                let end = offset
                    .checked_add(L::CHANNELS)
                    .ok_or(BufferError::OutOfBounds)?;
                let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
                stochastic_apply_pixel::<S, L>(pixel, mode, threshold_for(x, y))?;
            }
            Ok(())
        })?;

    Ok(())
}

fn stochastic_apply_pixel<S: Sample, L: PixelLayout>(
    pixel: &mut [S],
    mode: QuantizeMode<'_, S>,
    threshold: f32,
) -> Result<()> {
    if pixel.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }

    let threshold = threshold.clamp(0.0, 1.0);
    let rgba = read_unit_pixel::<S, L>(pixel);
    let luma = if L::COLOR_CHANNELS == 1 {
        rgba[0]
    } else {
        (0.299_f32 * rgba[0] + 0.587_f32 * rgba[1] + 0.114_f32 * rgba[2]).clamp(0.0, 1.0)
    };
    let binary = if luma > threshold { 1.0_f32 } else { 0.0_f32 };

    let sample = [
        S::from_unit_f32(binary),
        S::from_unit_f32(binary),
        S::from_unit_f32(binary),
        S::from_unit_f32(rgba[3]),
    ];
    let quantized = quantize_pixel::<S, L>(&sample[..L::CHANNELS], mode)?;
    pixel[..L::COLOR_CHANNELS].copy_from_slice(&quantized[..L::COLOR_CHANNELS]);
    if let Some(alpha_lane) = alpha_index::<L>() {
        pixel[alpha_lane] = S::from_unit_f32(rgba[3]);
    }

    Ok(())
}

fn perturbed_threshold_unit(
    prng: &mut XorShift64,
    seed: u64,
    x: usize,
    y: usize,
    strength: u8,
) -> f32 {
    if strength == 0 {
        return 127.0 / 255.0;
    }

    let span = u64::from(strength) * 2 + 1;
    let mixed = prng.next_u64() ^ coordinate_mix(seed, x, y);
    let jitter = (mixed % span) as i32 - i32::from(strength);
    let threshold = (127 + jitter).clamp(0, 255);
    threshold as f32 / 255.0
}

fn coordinate_mix(seed: u64, x: usize, y: usize) -> u64 {
    let mut value = seed
        ^ (x as u64).wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
        ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64);
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd_u64);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53_u64);
    value ^ (value >> 33)
}

const fn stochastic_layout_supported<L: PixelLayout>() -> bool {
    (L::HAS_ALPHA && L::CHANNELS == 4) || (!L::HAS_ALPHA && (L::CHANNELS == 1 || L::CHANNELS == 3))
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15_u64
        } else {
            seed
        };

        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
}
