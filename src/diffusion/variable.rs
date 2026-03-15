use super::{core::add_error_to_pixel, error_diffuse_in_place};
use crate::{
    data::FLOYD_STEINBERG,
    math::{
        color::luma_u8,
        fixed::mul_div_i32,
        utils::{clamp_i16, clamp_u8},
    },
    quantize_pixel, Buffer, PixelFormat, QuantizeMode,
};

const OSTROMOUKHOV_COEFFS: [(i16, i16, i16, i16); 16] = [
    (8, 2, 6, 16),
    (8, 2, 6, 16),
    (8, 2, 6, 16),
    (8, 2, 6, 16),
    (7, 3, 6, 16),
    (7, 3, 6, 16),
    (7, 3, 6, 16),
    (7, 3, 6, 16),
    (7, 3, 5, 15),
    (7, 3, 5, 15),
    (7, 3, 5, 15),
    (7, 3, 5, 15),
    (6, 4, 5, 15),
    (6, 4, 5, 15),
    (6, 4, 5, 15),
    (6, 4, 5, 15),
];

pub fn ostromoukhov_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    buffer
        .validate()
        .expect("buffer must be valid for ostromoukhov diffusion");

    if buffer.format != PixelFormat::Gray8 {
        error_diffuse_in_place(buffer, mode, &FLOYD_STEINBERG);
        return;
    }

    let width = buffer.width;
    let height = buffer.height;
    let mut errors = vec![0_i32; width.checked_mul(height).expect("image size overflow")];

    for y in 0..height {
        let row = buffer.row_mut(y);

        for (x, value) in row.iter_mut().take(width).enumerate() {
            let idx = y
                .checked_mul(width)
                .and_then(|base| base.checked_add(x))
                .expect("pixel index overflow");
            let original = *value;
            let adjusted = clamp_u8(i32::from(original) + errors[idx]);
            let quantized = quantize_pixel(PixelFormat::Gray8, &[adjusted], mode);
            let quantized_gray = luma_u8([quantized[0], quantized[1], quantized[2]]);
            *value = quantized_gray;

            let err = i32::from(adjusted) - i32::from(quantized_gray);
            let coeff = coefficient_for_luma(adjusted);
            let den = i32::from(coeff.3);

            let right = mul_div_i32(err, i32::from(coeff.0), den);
            let down_left = mul_div_i32(err, i32::from(coeff.1), den);
            let down = mul_div_i32(err, i32::from(coeff.2), den);

            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize + 1,
                y as isize,
                1,
                [i32::from(clamp_i16(right)), 0, 0],
            );
            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize - 1,
                y as isize + 1,
                1,
                [i32::from(clamp_i16(down_left)), 0, 0],
            );
            add_error_to_pixel(
                &mut errors,
                width,
                height,
                x as isize,
                y as isize + 1,
                1,
                [i32::from(clamp_i16(down)), 0, 0],
            );
        }
    }
}

fn coefficient_for_luma(luma: u8) -> (i16, i16, i16, i16) {
    let idx = (usize::from(luma) * OSTROMOUKHOV_COEFFS.len()) / 256;
    OSTROMOUKHOV_COEFFS[idx]
}
