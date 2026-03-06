pub mod buffer;
pub mod data;
pub mod diffusion;
pub mod math;
pub mod ordered;
pub mod palette;
pub mod quantize;
pub mod stochastic;

pub use buffer::{Buffer, BufferError, PixelFormat};
pub use data::{cga_palette, grayscale_16, grayscale_2, grayscale_4};
pub use palette::{IndexedImage, Palette, PaletteError};
pub use quantize::{
    quantize_error, quantize_gray_u8, quantize_pixel, quantize_rgb_u8, QuantizeMode,
};
pub use stochastic::threshold_in_place;
