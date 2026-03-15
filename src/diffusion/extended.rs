use super::error_diffuse_in_place;
use crate::{
    data::{FAN, SHIAU_FAN},
    Buffer, QuantizeMode,
};

pub fn fan_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &FAN);
}

pub fn shiau_fan_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &SHIAU_FAN);
}
