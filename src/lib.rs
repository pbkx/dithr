pub mod buffer;
pub mod data;
pub mod dbs;
pub mod diffusion;
pub mod dot_diffusion;
pub mod error;
pub mod math;
pub mod ordered;
pub mod palette;
pub mod quantize;
pub mod riemersma;
pub mod stochastic;

pub use buffer::{Buffer, BufferError, PixelFormat};
pub use data::{cga_palette, grayscale_16, grayscale_2, grayscale_4};
pub use dbs::{
    direct_binary_search_in_place, electrostatic_halftoning_in_place, lattice_boltzmann_in_place,
};
pub use diffusion::{
    atkinson_in_place, burkes_in_place, false_floyd_steinberg_in_place, fan_in_place,
    floyd_steinberg_in_place, gradient_based_error_diffusion_in_place,
    jarvis_judice_ninke_in_place, ostromoukhov_in_place, shiau_fan_2_in_place, shiau_fan_in_place,
    sierra_in_place, sierra_lite_in_place, stevenson_arce_in_place, stucki_in_place,
    two_row_sierra_in_place, zhou_fang_in_place,
};
pub use dot_diffusion::knuth_dot_diffusion_in_place;
pub use error::{DithrError, DithrResult};
pub use ordered::{
    bayer_16x16_in_place, bayer_2x2_in_place, bayer_4x4_in_place, bayer_8x8_in_place,
    cluster_dot_4x4_in_place, cluster_dot_8x8_in_place, custom_ordered_in_place,
    yliluoma_1_in_place, yliluoma_2_in_place, yliluoma_3_in_place, OrderedError,
};
#[cfg(feature = "rayon")]
pub use ordered::{
    bayer_16x16_in_place_par, bayer_2x2_in_place_par, bayer_4x4_in_place_par,
    bayer_8x8_in_place_par, cluster_dot_4x4_in_place_par, cluster_dot_8x8_in_place_par,
    custom_ordered_in_place_par,
};
pub use palette::{IndexedImage, Palette, PaletteError};
pub use quantize::{
    quantize_error, quantize_gray_u8, quantize_pixel, quantize_rgb_u8, QuantizeMode,
};
pub use riemersma::riemersma_in_place;
pub use stochastic::{
    random_binary_in_place, random_in_place, threshold_binary_in_place, threshold_in_place,
};
#[cfg(feature = "rayon")]
pub use stochastic::{random_binary_in_place_par, threshold_binary_in_place_par};
