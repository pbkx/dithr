use super::error_diffuse_in_place;
use crate::{
    data::{FALSE_FLOYD_STEINBERG, FLOYD_STEINBERG, JARVIS_JUDICE_NINKE},
    Buffer, QuantizeMode,
};

pub fn floyd_steinberg_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &FLOYD_STEINBERG);
}

pub fn false_floyd_steinberg_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &FALSE_FLOYD_STEINBERG);
}

pub fn jarvis_judice_ninke_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) {
    error_diffuse_in_place(buffer, mode, &JARVIS_JUDICE_NINKE);
}
