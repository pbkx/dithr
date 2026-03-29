use crate::{
    Buffer, Error, GrayBuffer16, GrayBuffer8, Result, RgbBuffer16, RgbBuffer32F, RgbBuffer8,
    RgbaBuffer16, RgbaBuffer32F, RgbaBuffer8,
};

pub enum DynamicImageBuffer<'a> {
    Gray8(GrayBuffer8<'a>),
    Rgb8(RgbBuffer8<'a>),
    Rgba8(RgbaBuffer8<'a>),
    Gray16(GrayBuffer16<'a>),
    Rgb16(RgbBuffer16<'a>),
    Rgba16(RgbaBuffer16<'a>),
    Rgb32F(RgbBuffer32F<'a>),
    Rgba32F(RgbaBuffer32F<'a>),
}

pub type Gray16Image = image::ImageBuffer<image::Luma<u16>, Vec<u16>>;
pub type Rgb16Image = image::ImageBuffer<image::Rgb<u16>, Vec<u16>>;
pub type Rgba16Image = image::ImageBuffer<image::Rgba<u16>, Vec<u16>>;

pub fn gray_image_as_buffer(img: &mut image::GrayImage) -> Result<GrayBuffer8<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, width)?)
}

pub fn rgb_image_as_buffer(img: &mut image::RgbImage) -> Result<RgbBuffer8<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(3)
        .ok_or(Error::InvalidArgument("rgb stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn rgba_image_as_buffer(img: &mut image::RgbaImage) -> Result<RgbaBuffer8<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("rgba stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn gray16_image_as_buffer(img: &mut Gray16Image) -> Result<GrayBuffer16<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, width)?)
}

pub fn rgb16_image_as_buffer(img: &mut Rgb16Image) -> Result<RgbBuffer16<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(3)
        .ok_or(Error::InvalidArgument("rgb16 stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn rgba16_image_as_buffer(img: &mut Rgba16Image) -> Result<RgbaBuffer16<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("rgba16 stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn rgb32f_image_as_buffer(img: &mut image::Rgb32FImage) -> Result<RgbBuffer32F<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(3)
        .ok_or(Error::InvalidArgument("rgb32f stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn rgba32f_image_as_buffer(img: &mut image::Rgba32FImage) -> Result<RgbaBuffer32F<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("rgba32f stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn dynamic_image_as_buffer(img: &mut image::DynamicImage) -> Result<DynamicImageBuffer<'_>> {
    match img {
        image::DynamicImage::ImageLuma8(inner) => {
            gray_image_as_buffer(inner).map(DynamicImageBuffer::Gray8)
        }
        image::DynamicImage::ImageRgb8(inner) => {
            rgb_image_as_buffer(inner).map(DynamicImageBuffer::Rgb8)
        }
        image::DynamicImage::ImageRgba8(inner) => {
            rgba_image_as_buffer(inner).map(DynamicImageBuffer::Rgba8)
        }
        image::DynamicImage::ImageLuma16(inner) => {
            gray16_image_as_buffer(inner).map(DynamicImageBuffer::Gray16)
        }
        image::DynamicImage::ImageRgb16(inner) => {
            rgb16_image_as_buffer(inner).map(DynamicImageBuffer::Rgb16)
        }
        image::DynamicImage::ImageRgba16(inner) => {
            rgba16_image_as_buffer(inner).map(DynamicImageBuffer::Rgba16)
        }
        image::DynamicImage::ImageRgb32F(inner) => {
            rgb32f_image_as_buffer(inner).map(DynamicImageBuffer::Rgb32F)
        }
        image::DynamicImage::ImageRgba32F(inner) => {
            rgba32f_image_as_buffer(inner).map(DynamicImageBuffer::Rgba32F)
        }
        image::DynamicImage::ImageLumaA8(_) => Err(Error::UnsupportedFormat(
            "DynamicImage LumaA8 is unsupported",
        )),
        image::DynamicImage::ImageLumaA16(_) => Err(Error::UnsupportedFormat(
            "DynamicImage LumaA16 is unsupported",
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
    use super::{
        dynamic_image_as_buffer, gray16_image_as_buffer, gray_image_as_buffer, DynamicImageBuffer,
    };
    use crate::{BufferKind, Error};

    #[test]
    fn gray_image_adapter_uses_packed_stride() {
        let mut img = image::GrayImage::new(4, 3);
        let buffer = gray_image_as_buffer(&mut img).expect("gray should be supported");

        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 3);
        assert_eq!(buffer.stride, 4);
        assert_eq!(buffer.kind(), BufferKind::Gray8);
    }

    #[test]
    fn gray16_image_adapter_uses_packed_stride() {
        let mut img = super::Gray16Image::new(4, 3);
        let buffer = gray16_image_as_buffer(&mut img).expect("gray16 should be supported");

        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 3);
        assert_eq!(buffer.stride, 4);
        assert_eq!(buffer.kind(), BufferKind::Gray16);
    }

    #[test]
    fn dynamic_image_adapter_supports_rgb8() {
        let mut img = image::DynamicImage::ImageRgb8(image::RgbImage::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("rgb8 should be supported");

        match converted {
            DynamicImageBuffer::Rgb8(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 6);
                assert_eq!(buffer.kind(), BufferKind::Rgb8);
            }
            _ => panic!("expected rgb8 buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_supports_gray16() {
        let mut img = image::DynamicImage::ImageLuma16(super::Gray16Image::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("gray16 should be supported");

        match converted {
            DynamicImageBuffer::Gray16(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 2);
                assert_eq!(buffer.kind(), BufferKind::Gray16);
            }
            _ => panic!("expected gray16 buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_supports_rgb16() {
        let mut img = image::DynamicImage::ImageRgb16(super::Rgb16Image::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("rgb16 should be supported");

        match converted {
            DynamicImageBuffer::Rgb16(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 6);
                assert_eq!(buffer.kind(), BufferKind::Rgb16);
            }
            _ => panic!("expected rgb16 buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_supports_rgba16() {
        let mut img = image::DynamicImage::ImageRgba16(super::Rgba16Image::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("rgba16 should be supported");

        match converted {
            DynamicImageBuffer::Rgba16(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 8);
                assert_eq!(buffer.kind(), BufferKind::Rgba16);
            }
            _ => panic!("expected rgba16 buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_supports_rgb32f() {
        let mut img = image::DynamicImage::ImageRgb32F(image::Rgb32FImage::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("rgb32f should be supported");

        match converted {
            DynamicImageBuffer::Rgb32F(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 6);
                assert_eq!(buffer.kind(), BufferKind::Rgb32F);
            }
            _ => panic!("expected rgb32f buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_supports_rgba32f() {
        let mut img = image::DynamicImage::ImageRgba32F(image::Rgba32FImage::new(2, 2));
        let converted = dynamic_image_as_buffer(&mut img).expect("rgba32f should be supported");

        match converted {
            DynamicImageBuffer::Rgba32F(buffer) => {
                assert_eq!(buffer.width, 2);
                assert_eq!(buffer.height, 2);
                assert_eq!(buffer.stride, 8);
                assert_eq!(buffer.kind(), BufferKind::Rgba32F);
            }
            _ => panic!("expected rgba32f buffer"),
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
