use crate::{math::color::luma_u8, quantize_pixel, Buffer, PixelFormat, QuantizeMode};

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
