use crate::{
    math::{color::luma_u8, utils::clamp_u8},
    quantize_pixel, Buffer, PixelFormat, QuantizeMode,
};

pub fn threshold_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>, threshold: u8) {
    buffer
        .validate()
        .expect("buffer must be valid for threshold dithering");

    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();

    for y in 0..height {
        let row = buffer.row_mut(y);

        for x in 0..width {
            let offset = x.checked_mul(bpp).expect("pixel offset overflow in row");

            match format {
                PixelFormat::Gray8 => {
                    let light = row[offset] > threshold;
                    let sample = if light { [255_u8] } else { [0_u8] };
                    let quantized = quantize_pixel(PixelFormat::Gray8, &sample, mode);
                    row[offset] = luma_u8([quantized[0], quantized[1], quantized[2]]);
                }
                PixelFormat::Rgb8 => {
                    let source = [row[offset], row[offset + 1], row[offset + 2]];
                    let light = luma_u8(source) > threshold;
                    let sample = if light {
                        [255_u8, 255_u8, 255_u8]
                    } else {
                        [0_u8, 0_u8, 0_u8]
                    };
                    let quantized = quantize_pixel(PixelFormat::Rgb8, &sample, mode);
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                }
                PixelFormat::Rgba8 => {
                    let alpha = row[offset + 3];
                    let source = [row[offset], row[offset + 1], row[offset + 2]];
                    let light = luma_u8(source) > threshold;
                    let sample = if light {
                        [255_u8, 255_u8, 255_u8, alpha]
                    } else {
                        [0_u8, 0_u8, 0_u8, alpha]
                    };
                    let quantized = quantize_pixel(PixelFormat::Rgba8, &sample, mode);
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                    row[offset + 3] = alpha;
                }
            }
        }
    }
}

pub fn random_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>, seed: u64, strength: u8) {
    buffer
        .validate()
        .expect("buffer must be valid for random dithering");

    let width = buffer.width;
    let height = buffer.height;
    let format = buffer.format;
    let bpp = format.bytes_per_pixel();
    let mut prng = XorShift64::new(seed);

    for y in 0..height {
        let row = buffer.row_mut(y);

        for x in 0..width {
            let threshold = perturbed_threshold(&mut prng, seed, x, y, strength);
            let offset = x.checked_mul(bpp).expect("pixel offset overflow in row");

            match format {
                PixelFormat::Gray8 => {
                    let light = row[offset] > threshold;
                    let sample = if light { [255_u8] } else { [0_u8] };
                    let quantized = quantize_pixel(PixelFormat::Gray8, &sample, mode);
                    row[offset] = luma_u8([quantized[0], quantized[1], quantized[2]]);
                }
                PixelFormat::Rgb8 => {
                    let source = [row[offset], row[offset + 1], row[offset + 2]];
                    let light = luma_u8(source) > threshold;
                    let sample = if light {
                        [255_u8, 255_u8, 255_u8]
                    } else {
                        [0_u8, 0_u8, 0_u8]
                    };
                    let quantized = quantize_pixel(PixelFormat::Rgb8, &sample, mode);
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                }
                PixelFormat::Rgba8 => {
                    let alpha = row[offset + 3];
                    let source = [row[offset], row[offset + 1], row[offset + 2]];
                    let light = luma_u8(source) > threshold;
                    let sample = if light {
                        [255_u8, 255_u8, 255_u8, alpha]
                    } else {
                        [0_u8, 0_u8, 0_u8, alpha]
                    };
                    let quantized = quantize_pixel(PixelFormat::Rgba8, &sample, mode);
                    row[offset] = quantized[0];
                    row[offset + 1] = quantized[1];
                    row[offset + 2] = quantized[2];
                    row[offset + 3] = alpha;
                }
            }
        }
    }
}

fn perturbed_threshold(prng: &mut XorShift64, seed: u64, x: usize, y: usize, strength: u8) -> u8 {
    if strength == 0 {
        return 127;
    }

    let span = u64::from(strength) * 2 + 1;
    let mixed = prng.next_u64() ^ coordinate_mix(seed, x, y);
    let jitter = (mixed % span) as i32 - i32::from(strength);

    clamp_u8(127 + jitter)
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
