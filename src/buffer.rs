use crate::core::{Gray, PixelLayout, Rgb, Rgba, Sample};
use std::mem::size_of;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat<L = ()> {
    Gray8,
    Rgb8,
    Rgba8,
    Gray16,
    Rgb16,
    Rgba16,
    Rgb32F,
    Rgba32F,
    #[doc(hidden)]
    __Layout(std::marker::PhantomData<L>),
}

impl<L> Copy for PixelFormat<L> {}

impl<L> Clone for PixelFormat<L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<L> PixelFormat<L> {
    #[must_use]
    pub const fn with_layout<NL>(self) -> PixelFormat<NL> {
        match self {
            Self::Gray8 => PixelFormat::Gray8,
            Self::Rgb8 => PixelFormat::Rgb8,
            Self::Rgba8 => PixelFormat::Rgba8,
            Self::Gray16 => PixelFormat::Gray16,
            Self::Rgb16 => PixelFormat::Rgb16,
            Self::Rgba16 => PixelFormat::Rgba16,
            Self::Rgb32F => PixelFormat::Rgb32F,
            Self::Rgba32F => PixelFormat::Rgba32F,
            Self::__Layout(_) => PixelFormat::__Layout(std::marker::PhantomData),
        }
    }

    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::Rgb8 | Self::Rgb16 | Self::Rgb32F => 3,
            Self::Rgba8 | Self::Rgba16 | Self::Rgba32F => 4,
            Self::__Layout(_) => 0,
        }
    }

    #[must_use]
    pub const fn has_alpha(self) -> bool {
        match self {
            Self::Rgba8 | Self::Rgba16 | Self::Rgba32F => true,
            Self::Gray8
            | Self::Rgb8
            | Self::Gray16
            | Self::Rgb16
            | Self::Rgb32F
            | Self::__Layout(_) => false,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            Self::Rgb32F | Self::Rgba32F => true,
            Self::Gray8
            | Self::Rgb8
            | Self::Rgba8
            | Self::Gray16
            | Self::Rgb16
            | Self::Rgba16
            | Self::__Layout(_) => false,
        }
    }

    #[must_use]
    pub const fn bytes_per_channel(self) -> usize {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 => 1,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 => 2,
            Self::Rgb32F | Self::Rgba32F => 4,
            Self::__Layout(_) => 0,
        }
    }

    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_channel()
    }

    #[must_use]
    pub fn supports_sample<S: Sample>(self) -> bool {
        if self.is_float() != S::IS_FLOAT {
            return false;
        }
        self.bytes_per_channel() == size_of::<S>()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    StrideTooSmall,
    DataTooShort,
    ZeroDimensions,
    RowOutOfBounds,
    PixelOutOfBounds,
    FormatSampleMismatch,
    OutOfBounds,
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StrideTooSmall => f.write_str("buffer stride is smaller than row width"),
            Self::DataTooShort => f.write_str("buffer data is shorter than required length"),
            Self::ZeroDimensions => f.write_str("buffer width and height must be non-zero"),
            Self::RowOutOfBounds => f.write_str("row index is out of bounds"),
            Self::PixelOutOfBounds => f.write_str("pixel coordinates are out of bounds"),
            Self::FormatSampleMismatch => {
                f.write_str("pixel format and sample type are not compatible")
            }
            Self::OutOfBounds => f.write_str("buffer arithmetic overflow or out-of-bounds access"),
        }
    }
}

impl std::error::Error for BufferError {}

pub struct Buffer<'a, S: Sample = u8, L: PixelLayout = ()> {
    pub data: &'a mut [S],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub format: PixelFormat<L>,
}

pub type GrayBuffer<'a, S> = Buffer<'a, S, Gray>;
pub type RgbBuffer<'a, S> = Buffer<'a, S, Rgb>;
pub type RgbaBuffer<'a, S> = Buffer<'a, S, Rgba>;

pub type GrayBuffer8<'a> = Buffer<'a, u8, Gray>;
pub type RgbBuffer8<'a> = Buffer<'a, u8, Rgb>;
pub type RgbaBuffer8<'a> = Buffer<'a, u8, Rgba>;

pub type GrayBuffer16<'a> = Buffer<'a, u16, Gray>;
pub type RgbBuffer16<'a> = Buffer<'a, u16, Rgb>;
pub type RgbaBuffer16<'a> = Buffer<'a, u16, Rgba>;

pub type RgbBuffer32F<'a> = Buffer<'a, f32, Rgb>;
pub type RgbaBuffer32F<'a> = Buffer<'a, f32, Rgba>;

impl<'a, S: Sample, L: PixelLayout> Buffer<'a, S, L> {
    pub fn new(
        data: &'a mut [S],
        width: usize,
        height: usize,
        stride: usize,
        format: PixelFormat,
    ) -> Result<Self, BufferError> {
        let buffer = Self {
            data,
            width,
            height,
            stride,
            format: format.with_layout(),
        };
        buffer.validate()?;
        Ok(buffer)
    }

    pub fn width_bytes(&self) -> Result<usize, BufferError> {
        self.try_width_bytes()
    }

    pub fn try_width_bytes(&self) -> Result<usize, BufferError> {
        self.try_width_samples()?
            .checked_mul(self.format.bytes_per_channel())
            .ok_or(BufferError::OutOfBounds)
    }

    pub fn required_len(&self) -> Result<usize, BufferError> {
        if self.width == 0 || self.height == 0 {
            return Err(BufferError::ZeroDimensions);
        }

        let width_samples = self.try_width_samples()?;
        if self.stride < width_samples {
            return Err(BufferError::StrideTooSmall);
        }

        self.stride
            .checked_mul(self.height)
            .ok_or(BufferError::OutOfBounds)
    }

    pub fn validate(&self) -> Result<(), BufferError> {
        if !self.format.supports_sample::<S>() {
            return Err(BufferError::FormatSampleMismatch);
        }

        let required_len = self.required_len()?;
        if self.data.len() < required_len {
            return Err(BufferError::DataTooShort);
        }

        Ok(())
    }

    pub fn row(&self, y: usize) -> Result<&[S], BufferError> {
        self.try_row(y)
    }

    pub fn try_row(&self, y: usize) -> Result<&[S], BufferError> {
        self.validate()?;
        if y >= self.height {
            return Err(BufferError::RowOutOfBounds);
        }

        let start = y.checked_mul(self.stride).ok_or(BufferError::OutOfBounds)?;
        let end = start
            .checked_add(self.stride)
            .ok_or(BufferError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(BufferError::DataTooShort);
        }

        Ok(&self.data[start..end])
    }

    pub fn row_mut(&mut self, y: usize) -> Result<&mut [S], BufferError> {
        self.try_row_mut(y)
    }

    pub fn try_row_mut(&mut self, y: usize) -> Result<&mut [S], BufferError> {
        self.validate()?;
        if y >= self.height {
            return Err(BufferError::RowOutOfBounds);
        }

        let start = y.checked_mul(self.stride).ok_or(BufferError::OutOfBounds)?;
        let end = start
            .checked_add(self.stride)
            .ok_or(BufferError::OutOfBounds)?;
        if end > self.data.len() {
            return Err(BufferError::DataTooShort);
        }

        Ok(&mut self.data[start..end])
    }

    pub fn pixel_offset(&self, x: usize, y: usize) -> Result<usize, BufferError> {
        self.try_pixel_offset(x, y)
    }

    pub fn try_pixel_offset(&self, x: usize, y: usize) -> Result<usize, BufferError> {
        self.validate()?;
        if x >= self.width || y >= self.height {
            return Err(BufferError::PixelOutOfBounds);
        }

        let channels = self.channels_per_pixel();
        let row_start = y.checked_mul(self.stride).ok_or(BufferError::OutOfBounds)?;
        let in_row = x.checked_mul(channels).ok_or(BufferError::OutOfBounds)?;
        let offset = row_start
            .checked_add(in_row)
            .ok_or(BufferError::OutOfBounds)?;

        let pixel_end = offset
            .checked_add(channels)
            .ok_or(BufferError::OutOfBounds)?;
        if pixel_end > self.data.len() {
            return Err(BufferError::DataTooShort);
        }

        Ok(offset)
    }

    pub fn pixel(&self, x: usize, y: usize) -> Result<&[S], BufferError> {
        let offset = self.try_pixel_offset(x, y)?;
        let channels = self.channels_per_pixel();
        let end = offset
            .checked_add(channels)
            .ok_or(BufferError::OutOfBounds)?;

        Ok(&self.data[offset..end])
    }

    pub fn pixel_mut(&mut self, x: usize, y: usize) -> Result<&mut [S], BufferError> {
        let offset = self.try_pixel_offset(x, y)?;
        let channels = self.channels_per_pixel();
        let end = offset
            .checked_add(channels)
            .ok_or(BufferError::OutOfBounds)?;

        Ok(&mut self.data[offset..end])
    }

    fn try_width_samples(&self) -> Result<usize, BufferError> {
        self.width
            .checked_mul(self.channels_per_pixel())
            .ok_or(BufferError::OutOfBounds)
    }

    fn channels_per_pixel(&self) -> usize {
        self.format.channels()
    }
}

pub fn gray_u8<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<GrayBuffer8<'a>, BufferError> {
    Buffer::new(data, width, height, stride, PixelFormat::Gray8)
}

pub fn rgb_u8<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbBuffer8<'a>, BufferError> {
    Buffer::new(data, width, height, stride, PixelFormat::Rgb8)
}

pub fn rgba_u8<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbaBuffer8<'a>, BufferError> {
    Buffer::new(data, width, height, stride, PixelFormat::Rgba8)
}

pub fn gray_u16<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<GrayBuffer16<'a>, BufferError> {
    Buffer::new(data, width, height, stride, PixelFormat::Gray16)
}

pub fn rgb_u16<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbBuffer16<'a>, BufferError> {
    Buffer::new(data, width, height, stride, PixelFormat::Rgb16)
}

pub fn rgba_u16<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbaBuffer16<'a>, BufferError> {
    Buffer::new(data, width, height, stride, PixelFormat::Rgba16)
}
