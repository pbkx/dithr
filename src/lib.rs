pub mod buffer;
pub mod data;
pub mod math;
pub mod palette;
pub mod quantize;

pub use buffer::{Buffer, BufferError, PixelFormat};
pub use data::{cga_palette, grayscale_16, grayscale_2, grayscale_4};
pub use palette::{IndexedImage, Palette, PaletteError};
pub use quantize::{
    quantize_error, quantize_gray_u8, quantize_pixel, quantize_rgb_u8, QuantizeMode,
};
