use super::core::write_quantized_pixel;
use crate::{
    core::{read_unit_pixel, PixelLayout, Sample},
    quantize_pixel, Buffer, BufferError, Error, QuantizeMode, Result,
};

const BLOCK_WIDTH: usize = 2;
const BLOCK_HEIGHT: usize = 2;
const BLOCK_WEIGHT_RIGHT: f32 = 7.0 / 16.0;
const BLOCK_WEIGHT_DOWN_LEFT: f32 = 3.0 / 16.0;
const BLOCK_WEIGHT_DOWN: f32 = 5.0 / 16.0;
const BLOCK_WEIGHT_DOWN_RIGHT: f32 = 1.0 / 16.0;

pub fn block_error_diffusion_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;

    if L::COLOR_CHANNELS != 1 || L::HAS_ALPHA {
        return Err(Error::UnsupportedFormat(
            "block error diffusion supports grayscale formats only",
        ));
    }

    let width = buffer.width;
    let height = buffer.height;
    let channels = L::CHANNELS;
    let blocks_w = width
        .checked_add(BLOCK_WIDTH - 1)
        .ok_or(Error::InvalidArgument("block grid width overflow"))?
        / BLOCK_WIDTH;
    let blocks_h = height
        .checked_add(BLOCK_HEIGHT - 1)
        .ok_or(Error::InvalidArgument("block grid height overflow"))?
        / BLOCK_HEIGHT;
    let block_count = blocks_w
        .checked_mul(blocks_h)
        .ok_or(Error::InvalidArgument("block grid size overflow"))?;
    let mut block_errors = vec![0.0_f32; block_count];

    for by in 0..blocks_h {
        for bx in 0..blocks_w {
            let block_idx = by * blocks_w + bx;
            let carry = block_errors[block_idx];

            let x_start = bx
                .checked_mul(BLOCK_WIDTH)
                .ok_or(Error::InvalidArgument("block x start overflow"))?;
            let y_start = by
                .checked_mul(BLOCK_HEIGHT)
                .ok_or(Error::InvalidArgument("block y start overflow"))?;
            let x_end = x_start
                .checked_add(BLOCK_WIDTH)
                .ok_or(Error::InvalidArgument("block x end overflow"))?
                .min(width);
            let y_end = y_start
                .checked_add(BLOCK_HEIGHT)
                .ok_or(Error::InvalidArgument("block y end overflow"))?
                .min(height);

            let mut residual_sum = 0.0_f32;
            let mut residual_count = 0_usize;

            for y in y_start..y_end {
                let row = buffer.try_row_mut(y)?;
                for x in x_start..x_end {
                    let offset = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;
                    let end = offset
                        .checked_add(channels)
                        .ok_or(BufferError::OutOfBounds)?;
                    let pixel = row.get_mut(offset..end).ok_or(BufferError::OutOfBounds)?;
                    let source = read_unit_pixel::<S, L>(pixel)?;
                    let adjusted_unit = (source[0] + carry).clamp(0.0, 1.0);
                    let adjusted = [S::from_unit_f32(adjusted_unit); 4];
                    let quantized = quantize_pixel::<S, L>(&adjusted[..channels], mode)?;
                    let quantized_unit = read_unit_pixel::<S, L>(&quantized[..channels])?;
                    write_quantized_pixel::<S, L>(pixel, quantized);
                    residual_sum += adjusted_unit - quantized_unit[0];
                    residual_count += 1;
                }
            }

            if residual_count == 0 {
                continue;
            }

            let block_residual = residual_sum / residual_count as f32;
            add_block_error(
                &mut block_errors,
                blocks_w,
                blocks_h,
                bx as isize + 1,
                by as isize,
                block_residual * BLOCK_WEIGHT_RIGHT,
            );
            add_block_error(
                &mut block_errors,
                blocks_w,
                blocks_h,
                bx as isize - 1,
                by as isize + 1,
                block_residual * BLOCK_WEIGHT_DOWN_LEFT,
            );
            add_block_error(
                &mut block_errors,
                blocks_w,
                blocks_h,
                bx as isize,
                by as isize + 1,
                block_residual * BLOCK_WEIGHT_DOWN,
            );
            add_block_error(
                &mut block_errors,
                blocks_w,
                blocks_h,
                bx as isize + 1,
                by as isize + 1,
                block_residual * BLOCK_WEIGHT_DOWN_RIGHT,
            );
        }
    }

    Ok(())
}

fn add_block_error(
    block_errors: &mut [f32],
    blocks_w: usize,
    blocks_h: usize,
    bx: isize,
    by: isize,
    delta: f32,
) {
    if bx < 0 || by < 0 {
        return;
    }

    let bx = bx as usize;
    let by = by as usize;
    if bx >= blocks_w || by >= blocks_h {
        return;
    }

    let idx = by * blocks_w + bx;
    block_errors[idx] += delta;
}
