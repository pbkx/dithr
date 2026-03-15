pub mod classic;
pub(crate) mod core;
pub mod extended;
pub mod variable;

use crate::{data::ErrorKernel, Buffer, QuantizeMode};

pub use classic::{
    atkinson_in_place, burkes_in_place, false_floyd_steinberg_in_place, floyd_steinberg_in_place,
    jarvis_judice_ninke_in_place, sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place,
    stucki_in_place, two_row_sierra_in_place,
};
pub use extended::{fan_in_place, shiau_fan_2_in_place, shiau_fan_in_place};
pub use variable::{
    gradient_based_error_diffusion_in_place, ostromoukhov_in_place, zhou_fang_in_place,
};

#[doc(hidden)]
pub fn error_diffuse_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    kernel: &ErrorKernel,
) {
    core::error_diffuse_in_place(buffer, mode, kernel);
}
