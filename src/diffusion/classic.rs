use super::error_diffuse_in_place;
use crate::{data::FLOYD_STEINBERG, Buffer, QuantizeMode};

pub fn floyd_steinberg_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &FLOYD_STEINBERG);
}
