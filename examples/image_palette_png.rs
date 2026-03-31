#[cfg(feature = "image")]
use dithr::{
    dynamic_image_as_buffer, floyd_steinberg_in_place, yliluoma_2_in_place, DynamicImageBuffer,
    Palette, QuantizeMode,
};
#[cfg(feature = "image")]
use std::path::PathBuf;

#[cfg(feature = "image")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some((input, output)) = parse_paths() else {
        print_usage();
        return Ok(());
    };

    let mut image = image::open(&input)?;
    let palette8 = Palette::new(vec![
        [0, 0, 0],
        [32, 32, 32],
        [96, 96, 96],
        [160, 160, 160],
        [224, 224, 224],
        [255, 255, 255],
        [255, 96, 0],
        [0, 160, 255],
    ])?;
    let palette16 = Palette::new(
        palette8
            .as_slice()
            .iter()
            .map(|&color| {
                [
                    u16::from(color[0]) * 257,
                    u16::from(color[1]) * 257,
                    u16::from(color[2]) * 257,
                ]
            })
            .collect(),
    )?;
    let palette32f = Palette::new(
        palette8
            .as_slice()
            .iter()
            .map(|&color| {
                [
                    f32::from(color[0]) / 255.0,
                    f32::from(color[1]) / 255.0,
                    f32::from(color[2]) / 255.0,
                ]
            })
            .collect(),
    )?;

    match dynamic_image_as_buffer(&mut image)? {
        DynamicImageBuffer::Gray8(mut buffer) => {
            floyd_steinberg_in_place(
                &mut buffer,
                QuantizeMode::gray_bits(1).expect("valid bit depth"),
            )?;
        }
        DynamicImageBuffer::Rgb8(mut buffer) => {
            yliluoma_2_in_place(&mut buffer, &palette8)?;
        }
        DynamicImageBuffer::Rgba8(mut buffer) => {
            yliluoma_2_in_place(&mut buffer, &palette8)?;
        }
        DynamicImageBuffer::Gray16(mut buffer) => {
            floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayLevels(2))?;
        }
        DynamicImageBuffer::Rgb16(mut buffer) => {
            yliluoma_2_in_place(&mut buffer, &palette16)?;
        }
        DynamicImageBuffer::Rgba16(mut buffer) => {
            yliluoma_2_in_place(&mut buffer, &palette16)?;
        }
        DynamicImageBuffer::Rgb32F(mut buffer) => {
            yliluoma_2_in_place(&mut buffer, &palette32f)?;
        }
        DynamicImageBuffer::Rgba32F(mut buffer) => {
            yliluoma_2_in_place(&mut buffer, &palette32f)?;
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
        .unwrap_or_else(|| "image_palette_png".to_string());
    println!("usage: {program} <input.png> <output.png>");
}

#[cfg(not(feature = "image"))]
fn main() {
    println!("enable the `image` feature to run this example");
}
