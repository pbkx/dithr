use crate::{Buffer, Error, PixelFormat, Result};

pub enum DynamicImageBuffer<'a> {
    Gray(Buffer<'a>),
    Rgb(Buffer<'a>),
    Rgba(Buffer<'a>),
}

pub fn gray_image_as_buffer(img: &mut image::GrayImage) -> Result<Buffer<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    Ok(Buffer {
        data: img.as_mut(),
        width,
        height,
        stride: width,
        format: PixelFormat::Gray8,
    })
}

pub fn rgb_image_as_buffer(img: &mut image::RgbImage) -> Result<Buffer<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(3)
        .ok_or(Error::InvalidArgument("rgb stride overflow"))?;
    Ok(Buffer {
        data: img.as_mut(),
        width,
        height,
        stride,
        format: PixelFormat::Rgb8,
    })
}

pub fn rgba_image_as_buffer(img: &mut image::RgbaImage) -> Result<Buffer<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("rgba stride overflow"))?;
    Ok(Buffer {
        data: img.as_mut(),
        width,
        height,
        stride,
        format: PixelFormat::Rgba8,
    })
}

pub fn dynamic_image_as_buffer(img: &mut image::DynamicImage) -> Result<DynamicImageBuffer<'_>> {
    match img {
        image::DynamicImage::ImageLuma8(inner) => {
            gray_image_as_buffer(inner).map(DynamicImageBuffer::Gray)
        }
        image::DynamicImage::ImageRgb8(inner) => {
            rgb_image_as_buffer(inner).map(DynamicImageBuffer::Rgb)
        }
        image::DynamicImage::ImageRgba8(inner) => {
            rgba_image_as_buffer(inner).map(DynamicImageBuffer::Rgba)
        }
        image::DynamicImage::ImageLumaA8(_) => Err(Error::UnsupportedFormat(
            "DynamicImage LumaA8 is unsupported",
        )),
        image::DynamicImage::ImageLuma16(_) => Err(Error::UnsupportedFormat(
            "DynamicImage Luma16 is unsupported",
        )),
        image::DynamicImage::ImageLumaA16(_) => Err(Error::UnsupportedFormat(
            "DynamicImage LumaA16 is unsupported",
        )),
        image::DynamicImage::ImageRgb16(_) => Err(Error::UnsupportedFormat(
            "DynamicImage Rgb16 is unsupported",
        )),
        image::DynamicImage::ImageRgba16(_) => Err(Error::UnsupportedFormat(
            "DynamicImage Rgba16 is unsupported",
        )),
        image::DynamicImage::ImageRgb32F(_) => Err(Error::UnsupportedFormat(
            "DynamicImage Rgb32F is unsupported",
        )),
        image::DynamicImage::ImageRgba32F(_) => Err(Error::UnsupportedFormat(
            "DynamicImage Rgba32F is unsupported",
        )),
        _ => Err(Error::UnsupportedFormat(
            "DynamicImage format is unsupported",
        )),
    }
}

fn image_dims(width: u32, height: u32) -> Result<(usize, usize)> {
    let width = usize::try_from(width)
        .map_err(|_| Error::InvalidArgument("image width does not fit usize"))?;
    let height = usize::try_from(height)
        .map_err(|_| Error::InvalidArgument("image height does not fit usize"))?;
    Ok((width, height))
}

#[cfg(test)]
mod tests {
    use super::{dynamic_image_as_buffer, gray_image_as_buffer, DynamicImageBuffer};
    use crate::{Error, PixelFormat};

    #[test]
    fn gray_image_adapter_uses_packed_stride() {
        let mut img = image::GrayImage::new(4, 3);
        let buffer = gray_image_as_buffer(&mut img).expect("gray should be supported");

        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 3);
        assert_eq!(buffer.stride, 4);
        assert_eq!(buffer.format, PixelFormat::Gray8);
    }

    #[test]
    fn dynamic_image_adapter_supports_rgb8() {
        let mut img = image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("rgb8 should be supported");

        match converted {
            DynamicImageBuffer::Rgb(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 6);
                assert_eq!(buffer.format, PixelFormat::Rgb8);
            }
            _ => panic!("expected rgb buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_rejects_lumaa8() {
        let mut img = image::DynamicImage::ImageLumaA8(image::GrayAlphaImage::new(2, 2));
        let result = dynamic_image_as_buffer(&mut img);

        match result {
            Err(Error::UnsupportedFormat(message)) => {
                assert_eq!(message, "DynamicImage LumaA8 is unsupported");
            }
            _ => panic!("expected unsupported format error"),
        }
    }
}
