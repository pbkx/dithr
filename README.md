# dithr

[![crates.io](https://img.shields.io/crates/v/dithr.svg)](https://crates.io/crates/dithr)

buffer-first rust dithering and halftoning library.

core crate does in-memory pixel transforms only:

- no png/jpeg/gif/webp decoding in core
- no ffmpeg/video decoding in core
- no cli parsing in core
- no filesystem path handling in core

the crate is designed for:

- image pipelines
- video frame pipelines
- game/rendering integrations
- wasm/canvas pipelines
- custom tooling

# install instructions

for crates.io:
```cargo add dithr```

from git:
```cargo add dithr --git https://github.com/pbkx/dithr```

# build instructions

dependencies are: rust>=1.75.0

```
git clone https://github.com/maxch/dithr
cd dithr
cargo build --release
cargo test --all-features
```

# usage

```
use dithr::{
    bayer_8x8_in_place, Buffer, PixelFormat, QuantizeMode, Result,
};

fn main() -> Result<()> {
    let width = 16;
    let height = 16;
    let mut data = (0_u16..256).map(|v| v as u8).collect::<Vec<_>>();

    let mut buffer = Buffer::new(&mut data, width, height, width, PixelFormat::Gray8)?;
    bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1))?;

    Ok(())
}
```

core types:

- Buffer
- PixelFormat
- Palette
- IndexedImage
- QuantizeMode
- Error
- Result

pixel formats:

- Gray8
- Rgb8
- Rgba8

quantize modes:

- GrayBits(bits)
- RgbBits(bits)
- Palette(&Palette)
- SingleColor { fg, bits }

public quantization helpers:

- quantize_gray_u8
- quantize_rgb_u8
- quantize_pixel
- quantize_error

buffer methods:

- new
- validate
- required_len
- try_width_bytes
- try_row
- try_row_mut
- try_pixel_offset
- row
- row_mut
- pixel_offset
- width_bytes

palette methods:

- new
- len
- is_empty
- as_slice
- get
- contains
- nearest_rgb_index
- nearest_rgb_color

ordered dithering:

- bayer_2x2_in_place
- bayer_4x4_in_place
- bayer_8x8_in_place
- bayer_16x16_in_place
- cluster_dot_4x4_in_place
- cluster_dot_8x8_in_place
- custom_ordered_in_place
- yliluoma_1_in_place
- yliluoma_2_in_place
- yliluoma_3_in_place

classic error diffusion:

- floyd_steinberg_in_place
- false_floyd_steinberg_in_place
- jarvis_judice_ninke_in_place
- stucki_in_place
- burkes_in_place
- sierra_in_place
- two_row_sierra_in_place
- sierra_lite_in_place
- stevenson_arce_in_place
- atkinson_in_place

extended and variable diffusion:

- fan_in_place
- shiau_fan_in_place
- shiau_fan_2_in_place
- ostromoukhov_in_place
- zhou_fang_in_place
- gradient_based_error_diffusion_in_place

stochastic:

- threshold_binary_in_place
- random_binary_in_place
- threshold_in_place
- random_in_place

other families:

- riemersma_in_place
- knuth_dot_diffusion_in_place
- direct_binary_search_in_place
- lattice_boltzmann_in_place
- electrostatic_halftoning_in_place

notes about format support:

- direct_binary_search_in_place supports Gray8 only
- lattice_boltzmann_in_place supports Gray8 only
- electrostatic_halftoning_in_place supports Gray8 only
- all functions are deterministic for same input and parameters

# features

default features:

- std

optional features:

- rayon
- image

rayon adds explicit parallel apis:

- bayer_2x2_in_place_par
- bayer_4x4_in_place_par
- bayer_8x8_in_place_par
- bayer_16x16_in_place_par
- cluster_dot_4x4_in_place_par
- cluster_dot_8x8_in_place_par
- custom_ordered_in_place_par
- threshold_binary_in_place_par
- random_binary_in_place_par

image adds thin adapters:

- gray_image_as_buffer
- rgb_image_as_buffer
- rgba_image_as_buffer
- dynamic_image_as_buffer
- DynamicImageBuffer::{Gray,Rgb,Rgba}

dynamic image support:

- supported: Luma8, Rgb8, Rgba8
- rejected: LumaA8, Luma16, LumaA16, Rgb16, Rgba16, Rgb32F, Rgba32F

# examples

raw buffer examples:

```
cargo run --example gray_buffer
cargo run --example rgb_buffer
cargo run --example indexed_palette
```

image workflow examples:

```
cargo run --example image_bayer_png --features image -- input.png output.png
cargo run --example image_palette_png --features image -- input.png output.png
```

# benchmarks

criterion benches are split by family:

- stochastic
- ordered
- yliluoma
- diffusion
- advanced

commands:

```
cargo bench --no-run
cargo bench --bench stochastic
cargo bench --bench ordered
cargo bench --bench yliluoma
cargo bench --bench diffusion
cargo bench --bench advanced
```

with rayon:

```
cargo bench --features rayon --no-run
```

# tests

```
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-targets --all-features
```

test layout:

- tests/basic.rs
- tests/ordered.rs
- tests/diffusion.rs
- tests/advanced.rs
- tests/golden.rs

golden tests use deterministic fixtures and fnv-1a hashes for regression locking.
