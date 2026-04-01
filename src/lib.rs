//! Buffer-first rust dithering and halftoning library.
//!
//! Quantizing grayscale/RGB/RGBA buffers without dithering creates visible
//! banding and contouring. `dithr` provides deterministic ordered dithering,
//! diffusion, stochastic binary methods, palette-constrained workflows, and
//! advanced halftoning methods over typed mutable slices.
//!
//! ### Overview
//!
//! - **Buffer-first API**: Operates on mutable pixel slices with explicit
//!   width/height/stride.
//! - **Typed sample support**: `u8`, `u16`, and `f32`.
//! - **Typed layout support**: Gray, Rgb, and Rgba.
//! - **Shared quantization model**: [`QuantizeMode`] is used across families.
//! - **Algorithm coverage**: Stochastic, ordered, palette-oriented ordered,
//!   diffusion, variable diffusion, and advanced halftoning.
//! - **Optional integrations**: `image` adapters and `rayon` parallel wrappers.
//!
//! ### Quick Start
//!
//! ```rust
//! use dithr::{bayer_8x8_in_place, gray_u8, QuantizeMode, Result};
//!
//! fn main() -> Result<()> {
//!     let width = 64_usize;
//!     let height = 64_usize;
//!     let mut data = Vec::with_capacity(width * height);
//!
//!     for y in 0..height {
//!         for x in 0..width {
//!             let value = ((x + y * width) * 255 / (width * height - 1)) as u8;
//!             data.push(value);
//!         }
//!     }
//!
//!     let mut buffer = gray_u8(&mut data, width, height, width)?;
//!     bayer_8x8_in_place(&mut buffer, QuantizeMode::gray_bits(1)?)?;
//!
//!     assert!(data.iter().all(|&v| v == 0 || v == 255));
//!     Ok(())
//! }
//! ```
//!
//! ### Core Types and Buffer Model
//!
//! Buffer and format types:
//!
//! - [`Buffer`]
//! - [`BufferKind`] (runtime format metadata)
//! - [`PixelFormat`] (type alias of [`BufferKind`])
//! - [`BufferError`]
//!
//! Generic and typed buffer aliases:
//!
//! - [`GrayBuffer`], [`RgbBuffer`], [`RgbaBuffer`]
//! - [`GrayBuffer8`], [`RgbBuffer8`], [`RgbaBuffer8`]
//! - [`GrayBuffer16`], [`RgbBuffer16`], [`RgbaBuffer16`]
//! - [`GrayBuffer32F`], [`RgbBuffer32F`], [`RgbaBuffer32F`]
//!
//! Constructors:
//!
//! - Gray: [`gray_u8`], [`gray_u16`], [`gray_32f`]
//! - Gray packed: [`gray_u8_packed`], [`gray_u16_packed`], [`gray_32f_packed`]
//! - Rgb: [`rgb_u8`], [`rgb_u16`], [`rgb_32f`]
//! - Rgb packed: [`rgb_u8_packed`], [`rgb_u16_packed`], [`rgb_32f_packed`]
//! - Rgba: [`rgba_u8`], [`rgba_u16`], [`rgba_32f`]
//! - Rgba packed: [`rgba_u8_packed`], [`rgba_u16_packed`], [`rgba_32f_packed`]
//!
//! Runtime kinds covered by [`BufferKind`]:
//!
//! - `Gray8`, `Rgb8`, `Rgba8`
//! - `Gray16`, `Rgb16`, `Rgba16`
//! - `Gray32F`, `Rgb32F`, `Rgba32F`
//!
//! ### Quantization API
//!
//! Quantization mode enum:
//!
//! - [`QuantizeMode::GrayLevels`]
//! - [`QuantizeMode::RgbLevels`]
//! - [`QuantizeMode::Palette`]
//! - [`QuantizeMode::SingleColor`]
//!
//! Quantization constructors/helpers:
//!
//! - [`QuantizeMode::gray_levels`]
//! - [`QuantizeMode::rgb_levels`]
//! - [`QuantizeMode::palette`]
//! - [`QuantizeMode::single_color`]
//! - [`QuantizeMode::gray_bits`] (`u8` specialization)
//! - [`QuantizeMode::rgb_bits`] (`u8` specialization)
//! - [`levels_from_bits`]
//!
//! Quantization functions:
//!
//! - [`quantize_gray`]
//! - [`quantize_rgb`]
//! - [`quantize_pixel`]
//! - [`quantize_error`]
//! - [`quantize_gray_u8`]
//! - [`quantize_rgb_u8`]
//!
//! Error alias:
//!
//! - [`QuantizeError`] (alias of crate [`Error`])
//!
//! ### Palette and Indexed Output API
//!
//! Palette types:
//!
//! - [`Palette`]
//! - [`Palette8`], [`Palette16`], [`Palette32F`]
//! - [`PaletteError`]
//!
//! Indexed image types:
//!
//! - [`IndexedImage`]
//! - [`IndexedImage8`], [`IndexedImage16`], [`IndexedImage32F`]
//!
//! Built-in palette helpers:
//!
//! - [`cga_palette`]
//! - [`grayscale_2`]
//! - [`grayscale_4`]
//! - [`grayscale_16`]
//!
//! ### Algorithm Families
//!
//! Binary stochastic:
//!
//! - [`threshold_binary_in_place`]
//! - [`random_binary_in_place`]
//! - compatibility aliases: [`threshold_in_place`], [`random_in_place`]
//!
//! Ordered:
//!
//! - Bayer: [`bayer_2x2_in_place`], [`bayer_4x4_in_place`],
//!   [`bayer_8x8_in_place`], [`bayer_16x16_in_place`]
//! - Cluster-dot: [`cluster_dot_4x4_in_place`], [`cluster_dot_8x8_in_place`]
//! - Custom map: [`custom_ordered_in_place`]
//!
//! Palette-oriented ordered (Yliluoma):
//!
//! - [`yliluoma_1_in_place`]
//! - [`yliluoma_2_in_place`]
//! - [`yliluoma_3_in_place`]
//!
//! Diffusion (classic):
//!
//! - [`floyd_steinberg_in_place`]
//! - [`false_floyd_steinberg_in_place`]
//! - [`jarvis_judice_ninke_in_place`]
//! - [`stucki_in_place`]
//! - [`burkes_in_place`]
//! - [`sierra_in_place`]
//! - [`two_row_sierra_in_place`]
//! - [`sierra_lite_in_place`]
//! - [`stevenson_arce_in_place`]
//! - [`atkinson_in_place`]
//!
//! Diffusion (extended):
//!
//! - [`fan_in_place`]
//! - [`shiau_fan_in_place`]
//! - [`shiau_fan_2_in_place`]
//!
//! Variable diffusion:
//!
//! - [`ostromoukhov_in_place`]
//! - [`zhou_fang_in_place`]
//! - [`gradient_based_error_diffusion_in_place`]
//!
//! Advanced halftoning:
//!
//! - [`riemersma_in_place`]
//! - [`knuth_dot_diffusion_in_place`]
//! - [`direct_binary_search_in_place`]
//! - [`lattice_boltzmann_in_place`]
//! - [`electrostatic_halftoning_in_place`]
//!
//! Scope notes:
//!
//! - [`ostromoukhov_in_place`], [`zhou_fang_in_place`], and
//!   [`gradient_based_error_diffusion_in_place`] are grayscale-only.
//! - [`direct_binary_search_in_place`], [`lattice_boltzmann_in_place`], and
//!   [`electrostatic_halftoning_in_place`] are integer grayscale-only.
//!
//! ### Parallel API (`rayon` feature)
//!
//! Parallel ordered wrappers:
//!
//! - `bayer_2x2_in_place_par`
//! - `bayer_4x4_in_place_par`
//! - `bayer_8x8_in_place_par`
//! - `bayer_16x16_in_place_par`
//! - `cluster_dot_4x4_in_place_par`
//! - `cluster_dot_8x8_in_place_par`
//! - `custom_ordered_in_place_par`
//!
//! Parallel stochastic wrappers:
//!
//! - `threshold_binary_in_place_par`
//! - `random_binary_in_place_par`
//!
//! ### Image Adapter API (`image` feature)
//!
//! Adapter enum:
//!
//! - `DynamicImageBuffer`
//!   - `Gray8`, `Rgb8`, `Rgba8`
//!   - `Gray16`, `Rgb16`, `Rgba16`
//!   - `Rgb32F`, `Rgba32F`
//!
//! Typed adapters:
//!
//! - `gray8_image_as_buffer`
//! - `rgb8_image_as_buffer`
//! - `rgba8_image_as_buffer`
//! - `gray16_image_as_buffer`
//! - `rgb16_image_as_buffer`
//! - `rgba16_image_as_buffer`
//! - `rgb32f_image_as_buffer`
//! - `rgba32f_image_as_buffer`
//!
//! Compatibility adapter names:
//!
//! - `gray_image_as_buffer`
//! - `rgb_image_as_buffer`
//! - `rgba_image_as_buffer`
//!
//! Dynamic adapter:
//!
//! - `dynamic_image_as_buffer`
//!
//! ### Errors and Result Types
//!
//! Crate-level error/result:
//!
//! - [`Error`]
//! - [`Result`]
//!
//! Family-specific error types re-exported at crate root:
//!
//! - [`OrderedError`]
//! - [`PaletteError`]
//! - [`BufferError`]
//!
//! ### Convenience Wrappers
//!
//! - [`floyd_steinberg_gray_u16_in_place`]
//! - [`bayer_8x8_rgb16_in_place`]
//!
//! For complete workflows and runnable examples, see `examples/` and the
//! repository README.
//!
#[cfg(feature = "image")]
pub mod adapter;
pub mod buffer;
pub mod core;
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

#[cfg(feature = "image")]
pub use adapter::{
    dynamic_image_as_buffer, gray16_image_as_buffer, gray8_image_as_buffer, gray_image_as_buffer,
    rgb16_image_as_buffer, rgb32f_image_as_buffer, rgb8_image_as_buffer, rgb_image_as_buffer,
    rgba16_image_as_buffer, rgba32f_image_as_buffer, rgba8_image_as_buffer, rgba_image_as_buffer,
    DynamicImageBuffer,
};
pub use buffer::{
    gray_32f, gray_32f_packed, gray_u16, gray_u16_packed, gray_u8, gray_u8_packed, rgb_32f,
    rgb_32f_packed, rgb_u16, rgb_u16_packed, rgb_u8, rgb_u8_packed, rgba_32f, rgba_32f_packed,
    rgba_u16, rgba_u16_packed, rgba_u8, rgba_u8_packed, Buffer, BufferError, BufferKind,
    GrayBuffer, GrayBuffer16, GrayBuffer32F, GrayBuffer8, PixelFormat, RgbBuffer, RgbBuffer16,
    RgbBuffer32F, RgbBuffer8, RgbaBuffer, RgbaBuffer16, RgbaBuffer32F, RgbaBuffer8,
};
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
pub use error::{Error, Result};
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
pub use palette::{
    IndexedImage, IndexedImage16, IndexedImage32F, IndexedImage8, Palette, Palette16, Palette32F,
    Palette8, PaletteError,
};
pub use quantize::{
    levels_from_bits, quantize_error, quantize_gray, quantize_gray_u8, quantize_pixel,
    quantize_rgb, quantize_rgb_u8, QuantizeError, QuantizeMode,
};
pub use riemersma::riemersma_in_place;
pub use stochastic::{
    random_binary_in_place, random_in_place, threshold_binary_in_place, threshold_in_place,
};
#[cfg(feature = "rayon")]
pub use stochastic::{random_binary_in_place_par, threshold_binary_in_place_par};

pub fn floyd_steinberg_gray_u16_in_place(buffer: &mut GrayBuffer16<'_>, levels: u16) -> Result<()> {
    floyd_steinberg_in_place(buffer, QuantizeMode::GrayLevels(levels))
}

pub fn bayer_8x8_rgb16_in_place(buffer: &mut RgbBuffer16<'_>, levels: u16) -> Result<()> {
    bayer_8x8_in_place(buffer, QuantizeMode::RgbLevels(levels))
}
