#[cfg(feature = "image")]
use dithr::core::{Gray, PixelLayout, Sample};
#[cfg(feature = "image")]
use dithr::{
    bayer_8x8_in_place, dynamic_image_as_buffer, Buffer, DynamicImageBuffer, QuantizeMode,
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
    match dynamic_image_as_buffer(&mut image)? {
        DynamicImageBuffer::Gray8(mut buffer) => {
            apply_gray_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Rgb8(mut buffer) => {
            apply_color_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Rgba8(mut buffer) => {
            apply_color_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Gray16(mut buffer) => {
            apply_gray_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Rgb16(mut buffer) => {
            apply_color_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Rgba16(mut buffer) => {
            apply_color_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Rgb32F(mut buffer) => {
            apply_color_ordered(&mut buffer)?;
        }
        DynamicImageBuffer::Rgba32F(mut buffer) => {
            apply_color_ordered(&mut buffer)?;
        }
    }

    image.save(&output)?;
    println!("wrote {}", output.display());
    Ok(())
}

#[cfg(feature = "image")]
fn apply_gray_ordered<S: Sample>(buffer: &mut Buffer<'_, S, Gray>) -> dithr::Result<()> {
    bayer_8x8_in_place(buffer, QuantizeMode::GrayLevels(2))
}

#[cfg(feature = "image")]
fn apply_color_ordered<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
) -> dithr::Result<()> {
    bayer_8x8_in_place(buffer, QuantizeMode::RgbLevels(2))
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
        .unwrap_or_else(|| "image_bayer_png".to_string());
    println!("usage: {program} <input> <output>");
    println!("example: cargo run --example image_bayer_png --features image -- in.png out.png");
}

#[cfg(not(feature = "image"))]
fn main() {
    println!("enable the `image` feature to run this example");
}
