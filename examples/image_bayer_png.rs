#[cfg(feature = "image")]
use dithr::{bayer_8x8_in_place, dynamic_image_as_buffer, DynamicImageBuffer, QuantizeMode};
#[cfg(feature = "image")]
use std::path::PathBuf;

#[cfg(feature = "image")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some((input, output)) = parse_paths() else {
        print_usage();
        return Ok(());
    };

    let mut image = image::open(&input)?;
    match dynamic_image_as_buffer(&mut image)? {
        DynamicImageBuffer::Gray8(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::gray_bits(1))?;
        }
        DynamicImageBuffer::Rgb8(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::rgb_bits(1))?;
        }
        DynamicImageBuffer::Rgba8(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::rgb_bits(1))?;
        }
        DynamicImageBuffer::Gray16(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayLevels(2))?;
        }
        DynamicImageBuffer::Rgb16(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2))?;
        }
        DynamicImageBuffer::Rgba16(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2))?;
        }
        DynamicImageBuffer::Rgb32F(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2))?;
        }
        DynamicImageBuffer::Rgba32F(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::RgbLevels(2))?;
        }
    }

    image.save(&output)?;
    println!("wrote {}", output.display());
    Ok(())
}

#[cfg(feature = "image")]
fn parse_paths() -> Option<(PathBuf, PathBuf)> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next()?);
    let output = PathBuf::from(args.next()?);
    Some((input, output))
}

#[cfg(feature = "image")]
fn print_usage() {
    let program = std::env::args()
        .next()
        .unwrap_or_else(|| "image_bayer_png".to_string());
    println!("usage: {program} <input.png> <output.png>");
}

#[cfg(not(feature = "image"))]
fn main() {
    println!("enable the `image` feature to run this example");
}
