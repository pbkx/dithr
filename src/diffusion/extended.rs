use super::error_diffuse_in_place;
use crate::{data::FAN, Buffer, QuantizeMode};

pub fn fan_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &FAN);
}
