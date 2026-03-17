use super::error_diffuse_in_place;
use crate::{
    data::{FAN, SHIAU_FAN, SHIAU_FAN_2},
    Buffer, DithrResult, QuantizeMode,
};

pub fn fan_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    error_diffuse_in_place(buffer, mode, &FAN)
}

pub fn shiau_fan_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    error_diffuse_in_place(buffer, mode, &SHIAU_FAN)
}

pub fn shiau_fan_2_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> DithrResult<()> {
    error_diffuse_in_place(buffer, mode, &SHIAU_FAN_2)
}
