#[cfg(feature = "image")]
use dithr::{
    bayer_8x8_in_place, cga_palette, dynamic_image_as_buffer, DynamicImageBuffer, Error,
    QuantizeMode, Result,
};
#[cfg(feature = "image")]
use std::path::PathBuf;

#[cfg(feature = "image")]
fn main() -> Result<()> {
    let Some((input, output)) = parse_paths() else {
        print_usage();
        return Ok(());
    };

    let mut image =
        image::open(&input).map_err(|_| Error::InvalidArgument("failed to open image"))?;
    let palette = cga_palette();

    match dynamic_image_as_buffer(&mut image)? {
        DynamicImageBuffer::Gray(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::GrayBits(1))?;
        }
        DynamicImageBuffer::Rgb(mut buffer) | DynamicImageBuffer::Rgba(mut buffer) => {
            bayer_8x8_in_place(&mut buffer, QuantizeMode::Palette(&palette))?;
        }
    }

    image
        .save(&output)
        .map_err(|_| Error::InvalidArgument("failed to save image"))?;
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
