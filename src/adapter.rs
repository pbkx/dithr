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

pub fn gray8_image_as_buffer(img: &mut image::GrayImage) -> Result<GrayBuffer8<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, width)?)
}

pub fn rgb8_image_as_buffer(img: &mut image::RgbImage) -> Result<RgbBuffer8<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(3)
        .ok_or(Error::InvalidArgument("rgb stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn rgba8_image_as_buffer(img: &mut image::RgbaImage) -> Result<RgbaBuffer8<'_>> {
    let (width, height) = image_dims(img.width(), img.height())?;
    let stride = width
        .checked_mul(4)
        .ok_or(Error::InvalidArgument("rgba stride overflow"))?;
    Ok(Buffer::new_typed(img.as_mut(), width, height, stride)?)
}

pub fn gray_image_as_buffer(img: &mut image::GrayImage) -> Result<GrayBuffer8<'_>> {
    gray8_image_as_buffer(img)
}

pub fn rgb_image_as_buffer(img: &mut image::RgbImage) -> Result<RgbBuffer8<'_>> {
    rgb8_image_as_buffer(img)
}

pub fn rgba_image_as_buffer(img: &mut image::RgbaImage) -> Result<RgbaBuffer8<'_>> {
    rgba8_image_as_buffer(img)
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
            gray8_image_as_buffer(inner).map(DynamicImageBuffer::Gray8)
        }
        image::DynamicImage::ImageRgb8(inner) => {
            rgb8_image_as_buffer(inner).map(DynamicImageBuffer::Rgb8)
        }
        image::DynamicImage::ImageRgba8(inner) => {
            rgba8_image_as_buffer(inner).map(DynamicImageBuffer::Rgba8)
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
        image::DynamicImage::ImageLumaA8(_) => {
            promote_lumaa8_to_rgba8(img)?;
            match img {
                image::DynamicImage::ImageRgba8(inner) => {
                    rgba8_image_as_buffer(inner).map(DynamicImageBuffer::Rgba8)
                }
                _ => Err(Error::UnsupportedFormat(
                    "DynamicImage LumaA8 promotion failed",
                )),
            }
        }
        image::DynamicImage::ImageLumaA16(_) => {
            promote_lumaa16_to_rgba16(img)?;
            match img {
                image::DynamicImage::ImageRgba16(inner) => {
                    rgba16_image_as_buffer(inner).map(DynamicImageBuffer::Rgba16)
                }
                _ => Err(Error::UnsupportedFormat(
                    "DynamicImage LumaA16 promotion failed",
                )),
            }
        }
        _ => Err(Error::UnsupportedFormat(
            "DynamicImage format is unsupported",
        )),
    }
}

fn promote_lumaa8_to_rgba8(img: &mut image::DynamicImage) -> Result<()> {
    let lumaa = match img {
        image::DynamicImage::ImageLumaA8(inner) => {
            core::mem::replace(inner, image::GrayAlphaImage::new(0, 0))
        }
        _ => return Err(Error::InvalidArgument("expected DynamicImage::ImageLumaA8")),
    };
    let rgba = image::DynamicImage::ImageLumaA8(lumaa).into_rgba8();
    *img = image::DynamicImage::ImageRgba8(rgba);
    Ok(())
}

fn promote_lumaa16_to_rgba16(img: &mut image::DynamicImage) -> Result<()> {
    let lumaa = match img {
        image::DynamicImage::ImageLumaA16(inner) => {
            core::mem::replace(inner, image::ImageBuffer::new(0, 0))
        }
        _ => {
            return Err(Error::InvalidArgument(
                "expected DynamicImage::ImageLumaA16",
            ))
        }
    };
    let rgba = image::DynamicImage::ImageLumaA16(lumaa).into_rgba16();
    *img = image::DynamicImage::ImageRgba16(rgba);
    Ok(())
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
        dynamic_image_as_buffer, gray16_image_as_buffer, gray8_image_as_buffer,
        gray_image_as_buffer, rgb8_image_as_buffer, rgba8_image_as_buffer, DynamicImageBuffer,
    };
    use crate::BufferKind;

    #[test]
    fn gray_image_adapter_uses_packed_stride() {
        let mut img = image::GrayImage::new(4, 3);
        let buffer = gray_image_as_buffer(&mut img).expect("gray should be supported");

        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 3);
        assert_eq!(buffer.stride, 4);
        assert_eq!(
            buffer.kind().expect("kind should resolve"),
            BufferKind::Gray8
        );
    }

    #[test]
    fn gray16_image_adapter_uses_packed_stride() {
        let mut img = super::Gray16Image::new(4, 3);
        let buffer = gray16_image_as_buffer(&mut img).expect("gray16 should be supported");

        assert_eq!(buffer.width, 4);
        assert_eq!(buffer.height, 3);
        assert_eq!(buffer.stride, 4);
        assert_eq!(
            buffer.kind().expect("kind should resolve"),
            BufferKind::Gray16
        );
    }

    #[test]
    fn explicit_8bit_adapter_names_match_compatibility_names() {
        let mut gray = image::GrayImage::new(3, 2);
        let mut rgb = image::RgbImage::new(3, 2);
        let mut rgba = image::RgbaImage::new(3, 2);

        let gray_new = gray8_image_as_buffer(&mut gray).expect("gray8 should be supported");
        let rgb_new = rgb8_image_as_buffer(&mut rgb).expect("rgb8 should be supported");
        let rgba_new = rgba8_image_as_buffer(&mut rgba).expect("rgba8 should be supported");

        assert_eq!(gray_new.stride, 3);
        assert_eq!(rgb_new.stride, 9);
        assert_eq!(rgba_new.stride, 12);
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
                assert_eq!(
                    buffer.kind().expect("kind should resolve"),
                    BufferKind::Rgb8
                );
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
                assert_eq!(
                    buffer.kind().expect("kind should resolve"),
                    BufferKind::Gray16
                );
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
                assert_eq!(
                    buffer.kind().expect("kind should resolve"),
                    BufferKind::Rgb16
                );
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
                assert_eq!(
                    buffer.kind().expect("kind should resolve"),
                    BufferKind::Rgba16
                );
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
                assert_eq!(
                    buffer.kind().expect("kind should resolve"),
                    BufferKind::Rgb32F
                );
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
                assert_eq!(
                    buffer.kind().expect("kind should resolve"),
                    BufferKind::Rgba32F
                );
            }
            _ => panic!("expected rgba32f buffer"),
        }
    }

    #[test]
    fn dynamic_image_adapter_promotes_lumaa8_to_rgba8() {
        let mut lumaa = image::GrayAlphaImage::new(2, 1);
        lumaa.put_pixel(0, 0, image::LumaA([10, 20]));
        lumaa.put_pixel(1, 0, image::LumaA([200, 255]));
        let mut img = image::DynamicImage::ImageLumaA8(lumaa);

        {
            let converted = dynamic_image_as_buffer(&mut img).expect("lumaa8 should be supported");

            match converted {
                DynamicImageBuffer::Rgba8(buffer) => {
                    assert_eq!(buffer.width, 2);
                    assert_eq!(buffer.height, 1);
                    assert_eq!(buffer.stride, 8);
                    assert_eq!(
                        buffer.kind().expect("kind should resolve"),
                        BufferKind::Rgba8
                    );
                }
                _ => panic!("expected rgba8 buffer"),
            }
        }

        match &img {
            image::DynamicImage::ImageRgba8(promoted) => {
                assert_eq!(promoted.get_pixel(0, 0).0, [10, 10, 10, 20]);
                assert_eq!(promoted.get_pixel(1, 0).0, [200, 200, 200, 255]);
            }
            _ => panic!("expected promoted rgba8 image"),
        }
    }

    #[test]
    fn dynamic_image_adapter_promotes_lumaa16_to_rgba16() {
        let mut lumaa = image::ImageBuffer::<image::LumaA<u16>, Vec<u16>>::new(2, 1);
        lumaa.put_pixel(0, 0, image::LumaA([1024, 2048]));
        lumaa.put_pixel(1, 0, image::LumaA([50000, 65535]));
        let mut img = image::DynamicImage::ImageLumaA16(lumaa);

        {
            let converted = dynamic_image_as_buffer(&mut img).expect("lumaa16 should be supported");

            match converted {
                DynamicImageBuffer::Rgba16(buffer) => {
                    assert_eq!(buffer.width, 2);
                    assert_eq!(buffer.height, 1);
                    assert_eq!(buffer.stride, 8);
                    assert_eq!(
                        buffer.kind().expect("kind should resolve"),
                        BufferKind::Rgba16
                    );
                }
                _ => panic!("expected rgba16 buffer"),
            }
        }

        match &img {
            image::DynamicImage::ImageRgba16(promoted) => {
                assert_eq!(promoted.get_pixel(0, 0).0, [1024, 1024, 1024, 2048]);
                assert_eq!(promoted.get_pixel(1, 0).0, [50000, 50000, 50000, 65535]);
            }
            _ => panic!("expected promoted rgba16 image"),
        }
    }
}
