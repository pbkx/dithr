#[cfg(feature = "image")]
use dithr::{
    core::{PixelLayout, Sample},
    dynamic_image_as_buffer, floyd_steinberg_in_place, yliluoma_2_in_place, Buffer,
    DynamicImageBuffer, Palette, QuantizeMode,
};
#[cfg(feature = "image")]
use std::path::PathBuf;

#[cfg(feature = "image")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let (input, output) = match args {
        CliArgs::Help => {
            print_usage();
            return Ok(());
        }
        CliArgs::Paths(input, output) => (input, output),
        CliArgs::Invalid => {
            print_usage();
            return Ok(());
        }
    };

    let mut image = image::open(&input)?;
    let base_palette = vec![
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
    ];
    let palette8 = Palette::new(base_palette.clone())?;
    let palette16 = Palette::new(palette_from_u8_to_u16(&base_palette))?;
    let palette32f = Palette::new(palette_from_u8_to_f32(&base_palette))?;

    match dynamic_image_as_buffer(&mut image)? {
        DynamicImageBuffer::Gray8(mut buffer) => {
            apply_gray_fallback(&mut buffer)?;
        }
        DynamicImageBuffer::Rgb8(mut buffer) => {
            apply_palette_dither(&mut buffer, &palette8)?;
        }
        DynamicImageBuffer::Rgba8(mut buffer) => {
            apply_palette_dither(&mut buffer, &palette8)?;
        }
        DynamicImageBuffer::Gray16(mut buffer) => {
            apply_gray_fallback(&mut buffer)?;
        }
        DynamicImageBuffer::Rgb16(mut buffer) => {
            apply_palette_dither(&mut buffer, &palette16)?;
        }
        DynamicImageBuffer::Rgba16(mut buffer) => {
            apply_palette_dither(&mut buffer, &palette16)?;
        }
        DynamicImageBuffer::Rgb32F(mut buffer) => {
            apply_palette_dither(&mut buffer, &palette32f)?;
        }
        DynamicImageBuffer::Rgba32F(mut buffer) => {
            apply_palette_dither(&mut buffer, &palette32f)?;
        }
    }

    image.save(&output)?;
    println!("wrote {}", output.display());
    Ok(())
}

#[cfg(feature = "image")]
fn apply_gray_fallback<S: Sample>(
    buffer: &mut Buffer<'_, S, dithr::core::Gray>,
) -> dithr::Result<()> {
    floyd_steinberg_in_place(buffer, QuantizeMode::GrayLevels(4))
}

#[cfg(feature = "image")]
fn apply_palette_dither<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    palette: &Palette<S>,
) -> dithr::Result<()> {
    yliluoma_2_in_place(buffer, palette)
}

#[cfg(feature = "image")]
fn palette_from_u8_to_u16(colors: &[[u8; 3]]) -> Vec<[u16; 3]> {
    colors
        .iter()
        .map(|&[r, g, b]| [u16::from(r) * 257, u16::from(g) * 257, u16::from(b) * 257])
        .collect()
}

#[cfg(feature = "image")]
fn palette_from_u8_to_f32(colors: &[[u8; 3]]) -> Vec<[f32; 3]> {
    colors
        .iter()
        .map(|&[r, g, b]| {
            [
                f32::from(r) / 255.0,
                f32::from(g) / 255.0,
                f32::from(b) / 255.0,
            ]
        })
        .collect()
}

#[cfg(feature = "image")]
enum CliArgs {
    Help,
    Paths(PathBuf, PathBuf),
    Invalid,
}

#[cfg(feature = "image")]
fn parse_args() -> CliArgs {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next(), args.next()) {
        (Some(flag), None, None) if flag == "--help" || flag == "-h" => CliArgs::Help,
        (Some(input), Some(output), None) => CliArgs::Paths(input.into(), output.into()),
        _ => CliArgs::Invalid,
    }
}

#[cfg(feature = "image")]
fn print_usage() {
    let program = std::env::args()
        .next()
        .unwrap_or_else(|| "image_palette_png".to_string());
    println!("usage: {program} <input> <output>");
    println!("example: cargo run --example image_palette_png --features image -- in.png out.png");
}

#[cfg(not(feature = "image"))]
fn main() {
    println!("enable the `image` feature to run this example");
}
