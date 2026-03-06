pub mod kernels;
pub mod maps;
pub mod palettes;

pub use kernels::{
    ErrorKernel, KernelTap, ATKINSON, BURKES, FALSE_FLOYD_STEINBERG, FAN, FLOYD_STEINBERG,
    JARVIS_JUDICE_NINKE, SHIAU_FAN, SHIAU_FAN_2, SIERRA, SIERRA_2_4A, SIERRA_LITE, STEVENSON_ARCE,
    STUCKI, TWO_ROW_SIERRA,
};
pub use maps::{
    generate_bayer_16x16, BAYER_2X2, BAYER_4X4, BAYER_8X8, CLUSTER_DOT_4X4, CLUSTER_DOT_8X8,
};
pub use palettes::{cga_palette, grayscale_16, grayscale_2, grayscale_4};
