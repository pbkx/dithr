pub mod classic;
pub(crate) mod core;
pub mod extended;
pub mod variable;

use crate::{
    core::{PixelLayout, Sample},
    data::ErrorKernel,
    Buffer, QuantizeMode, Result,
};

pub use classic::{
    atkinson_in_place, burkes_in_place, false_floyd_steinberg_in_place, floyd_steinberg_in_place,
    jarvis_judice_ninke_in_place, sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place,
    stucki_in_place, two_row_sierra_in_place,
};
pub use extended::{fan_in_place, shiau_fan_2_in_place, shiau_fan_in_place};
pub use variable::{
    adaptive_vector_error_diffusion_in_place, feature_preserving_msed_in_place,
    gradient_based_error_diffusion_in_place, green_noise_msed_in_place,
    hvs_optimized_error_diffusion_in_place, linear_pixel_shuffling_in_place,
    multiscale_error_diffusion_in_place, ostromoukhov_in_place,
    semivector_error_diffusion_in_place, vector_error_diffusion_in_place, zhou_fang_in_place,
};

#[doc(hidden)]
pub(crate) fn error_diffuse_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
    kernel: &ErrorKernel,
) -> Result<()> {
    core::error_diffuse_in_place(buffer, mode, kernel)
}
