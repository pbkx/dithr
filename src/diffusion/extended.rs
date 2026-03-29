use super::error_diffuse_in_place;
use crate::{
    core::{PixelLayout, Sample},
    data::{FAN, SHIAU_FAN, SHIAU_FAN_2},
    Buffer, QuantizeMode, Result,
};

pub fn fan_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &FAN)
}

pub fn shiau_fan_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &SHIAU_FAN)
}

pub fn shiau_fan_2_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &SHIAU_FAN_2)
}
