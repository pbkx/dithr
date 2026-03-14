pub mod classic;
pub(crate) mod core;

use crate::{data::ErrorKernel, Buffer, QuantizeMode};

pub use classic::{
    false_floyd_steinberg_in_place, floyd_steinberg_in_place, jarvis_judice_ninke_in_place,
    stucki_in_place,
};

#[doc(hidden)]
pub fn error_diffuse_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    kernel: &ErrorKernel,
) {
    core::error_diffuse_in_place(buffer, mode, kernel);
}
