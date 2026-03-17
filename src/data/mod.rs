pub mod kernels;
pub mod maps;
pub mod palettes;
pub mod variable;

pub use kernels::{
    ErrorKernel, KernelTap, ATKINSON, BURKES, FALSE_FLOYD_STEINBERG, FAN, FLOYD_STEINBERG,
    JARVIS_JUDICE_NINKE, SHIAU_FAN, SHIAU_FAN_2, SIERRA, SIERRA_LITE, STEVENSON_ARCE, STUCKI,
    TWO_ROW_SIERRA,
};
pub use maps::{
    generate_bayer_16x16, generate_bayer_16x16_flat, BAYER_2X2, BAYER_2X2_FLAT, BAYER_4X4,
    BAYER_4X4_FLAT, BAYER_8X8, BAYER_8X8_FLAT, CLUSTER_DOT_4X4, CLUSTER_DOT_4X4_FLAT,
    CLUSTER_DOT_8X8, CLUSTER_DOT_8X8_FLAT,
};
pub use palettes::{cga_palette, grayscale_16, grayscale_2, grayscale_4};
pub use variable::{OSTROMOUKHOV_COEFFS, ZHOU_FANG_MODULATION};
