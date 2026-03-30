# dithr

[![crates.io](https://img.shields.io/crates/v/dithr.svg)](https://crates.io/crates/dithr)

buffer-first rust dithering and halftoning library.

core crate provides in-memory pixel transforms:

- generic core over sample/layout abstractions
- concrete ergonomic APIs for `u8`, `u16`, and `f32`
- deterministic ordered dithering, error diffusion, stochastic dithering, and advanced halftoning
- optional `rayon` parallel paths for ordered and binary stochastic families
- optional `image` adapters for zero-copy conversion from `DynamicImage` variants

# install instructions

for crates.io:
```cargo add dithr```

from git:
```cargo add dithr --git https://github.com/pbkx/dithr```

# build instructions

dependencies are: rust>=1.75.0

```
git clone https://github.com/pbkx/dithr
cd dithr
cargo build --release
cargo test --all-features
```

# usage

```rust
use dithr::{bayer_8x8_in_place, Buffer, PixelFormat, QuantizeMode, Result};

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

- `Buffer`
- `PixelFormat`
- `Palette`
- `IndexedImage`
- `QuantizeMode`
- `Error`
- `Result`

pixel formats:

- `Gray8`, `Rgb8`, `Rgba8`
- `Gray16`, `Rgb16`, `Rgba16`
- `Rgb32F`, `Rgba32F`

quantize modes:

- `GrayBits(bits)` / `GrayLevels(levels)`
- `RgbBits(bits)` / `RgbLevels(levels)`
- `Palette(&Palette<_>)`
- `SingleColor { fg, levels }`

algorithm families:

- stochastic: `threshold_binary_in_place`, `random_binary_in_place`
- ordered: Bayer, cluster-dot, custom ordered, Yliluoma 1/2/3
- classic diffusion: Floyd-Steinberg, JJN, Stucki, Burkes, Sierra variants, Stevenson-Arce, Atkinson
- extended diffusion: Fan, Shiau-Fan, Shiau-Fan-2
- variable diffusion: Ostromoukhov, Zhou-Fang, gradient-based diffusion
- other families: Riemersma, Knuth dot diffusion, DBS, lattice-Boltzmann, electrostatic

format support notes:

- `ostromoukhov_in_place`, `zhou_fang_in_place`, and `gradient_based_error_diffusion_in_place` are grayscale-only
- `direct_binary_search_in_place`, `lattice_boltzmann_in_place`, and `electrostatic_halftoning_in_place` are grayscale-first research-grade methods

optional image adapters:

- `GrayImage`, `RgbImage`, `RgbaImage`
- `ImageLuma16`, `ImageRgb16`, `ImageRgba16`
- `ImageRgb32F`, `ImageRgba32F`

# basic example usage

```
$ cargo run --example gray_buffer
pixels=256 black=128 white=128

$ cargo run --example rgb_buffer
pixels=4096 binary_channels=true

$ cargo run --example indexed_palette
pixels=1024 palette_entries=4 used_indices=4

$ cargo run --example image_bayer_png --features image -- input.png output.png
wrote output.png

$ cargo run --example image_palette_png --features image -- input.png output.png
wrote output.png
```

# more advanced usage

```rust
use dithr::{
    floyd_steinberg_in_place, random_binary_in_place, Buffer, Palette, PixelFormat, QuantizeMode,
    Result,
};

fn main() -> Result<()> {
    let width = 128;
    let height = 128;
    let mut rgb = vec![0_u8; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            rgb[i] = (x * 255 / (width - 1)) as u8;
            rgb[i + 1] = (y * 255 / (height - 1)) as u8;
            rgb[i + 2] = ((x + y) * 255 / (width + height - 2)) as u8;
        }
    }

    let palette = Palette::new(vec![
        [0, 0, 0],
        [255, 255, 255],
        [255, 85, 85],
        [85, 170, 255],
    ])?;

    let mut rgb_buffer = Buffer::new(&mut rgb, width, height, width * 3, PixelFormat::Rgb8)?;
    floyd_steinberg_in_place(&mut rgb_buffer, QuantizeMode::Palette(&palette))?;

    let mut gray = (0..(width * height))
        .map(|i| (i * 255 / (width * height - 1)) as u8)
        .collect::<Vec<_>>();
    let mut gray_buffer = Buffer::new(&mut gray, width, height, width, PixelFormat::Gray8)?;
    random_binary_in_place(&mut gray_buffer, QuantizeMode::GrayBits(1), 42, 64)?;

    Ok(())
}
```

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

- `tests/basic.rs`
- `tests/ordered.rs`
- `tests/diffusion.rs`
- `tests/advanced.rs`
- `tests/golden.rs`

golden tests use deterministic fixtures and fnv-1a hashes for stable u8 regression locking.

# references

|     | source |
| --- | --- |
| [1] | Dither<br>https://en.wikipedia.org/wiki/Dither |
| [2] | Ordered dithering<br>https://en.wikipedia.org/wiki/Ordered_dithering |
| [3] | Error diffusion<br>https://en.wikipedia.org/wiki/Error_diffusion |
| [4] | Yliluoma positional dithering<br>https://bisqwit.iki.fi/story/howto/dither/jy/ |
| [5] | Dithering eleven algorithms<br>https://tannerhelland.com/2012/12/28/dithering-eleven-algorithms-source-code.html |
| [6] | Error diffusion study<br>https://caca.zoy.org/study/part3.html |
| [7] | Ostromoukhov variable-coefficient error diffusion<br>https://www.iro.umontreal.ca/~ostrom/publications/pdf/SIGGRAPH01_varcoeffED.pdf<br>https://doi.org/10.1145/383259.383326 |
| [8] | Zhou-Fang threshold modulation<br>https://history.siggraph.org/learning/improving-mid-tone-quality-of-variable-coefficient-error-diffusion-using-threshold-modulation-by-zhou-and-fang/ |
| [9] | Riemersma dithering<br>https://www.compuphase.com/riemer.htm |
| [10] | Knuth dot diffusion<br>https://dl.acm.org/doi/10.1145/35039.35040<br>https://doi.org/10.1109/TIP.2009.2023455<br>https://doi.org/10.2352/ISSN.2169-4451.1999.15.1.art00091_1 |
| [11] | Direct binary search halftoning<br>https://www.spiedigitallibrary.org/conference-proceedings-of-spie/1666/1/Model-based-halftoning-using-direct-binary-search/10.1117/12.135959.full<br>https://doi.org/10.1117/12.135959<br>https://doi.org/10.1109/MSP.2003.1215228 |
| [12] | Lattice-Boltzmann halftoning<br>https://www.mia.uni-saarland.de/Publications/hagenburg-isvc09.pdf |
| [13] | Electrostatic halftoning<br>https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2010.01716.x |
