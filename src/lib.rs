pub mod buffer;
pub mod palette;
pub mod quantize;

pub use buffer::{Buffer, BufferError, PixelFormat};
pub use palette::{IndexedImage, Palette, PaletteError};
pub use quantize::{
    quantize_error, quantize_gray_u8, quantize_pixel, quantize_rgb_u8, QuantizeMode,
};
