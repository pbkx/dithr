use crate::core::{Gray, PixelLayout, Rgb, Rgba, Sample};
use std::marker::PhantomData;
use std::mem::size_of;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferKind {
    Gray8,
    Rgb8,
    Rgba8,
    Gray16,
    Rgb16,
    Rgba16,
    Gray32F,
    Rgb32F,
    Rgba32F,
}

impl BufferKind {
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 | Self::Gray32F => 1,
            Self::Rgb8 | Self::Rgb16 | Self::Rgb32F => 3,
            Self::Rgba8 | Self::Rgba16 | Self::Rgba32F => 4,
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
            | Self::Gray32F
            | Self::Rgb32F => false,
        }
    }

    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            Self::Gray32F | Self::Rgb32F | Self::Rgba32F => true,
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 | Self::Gray16 | Self::Rgb16 | Self::Rgba16 => {
                false
            }
        }
    }

    #[must_use]
    pub const fn bytes_per_channel(self) -> usize {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 => 1,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 => 2,
            Self::Gray32F | Self::Rgb32F | Self::Rgba32F => 4,
        }
    }

    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_channel()
    }
}

pub type PixelFormat = BufferKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    StrideTooSmall,
    DataTooShort,
    ZeroDimensions,
    RowOutOfBounds,
    PixelOutOfBounds,
    KindMismatch,
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
            Self::KindMismatch => f.write_str("buffer kind does not match sample/layout type"),
            Self::OutOfBounds => f.write_str("buffer arithmetic overflow or out-of-bounds access"),
        }
    }
}

impl std::error::Error for BufferError {}

pub struct Buffer<'a, S: Sample, L: PixelLayout> {
    pub(crate) data: &'a mut [S],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) stride: usize,
    _layout: PhantomData<L>,
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

pub type GrayBuffer32F<'a> = Buffer<'a, f32, Gray>;
pub type RgbBuffer32F<'a> = Buffer<'a, f32, Rgb>;
pub type RgbaBuffer32F<'a> = Buffer<'a, f32, Rgba>;

impl<'a, S: Sample, L: PixelLayout> Buffer<'a, S, L> {
    pub fn new(
        data: &'a mut [S],
        width: usize,
        height: usize,
        stride: usize,
        kind: BufferKind,
    ) -> Result<Self, BufferError> {
        let buffer = Self::new_typed(data, width, height, stride)?;
        if buffer.kind()? != kind {
            return Err(BufferError::KindMismatch);
        }
        Ok(buffer)
    }

    pub fn new_typed(
        data: &'a mut [S],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<Self, BufferError> {
        let buffer = Self {
            data,
            width,
            height,
            stride,
            _layout: PhantomData,
        };
        buffer.validate()?;
        Ok(buffer)
    }

    pub fn new_packed(
        data: &'a mut [S],
        width: usize,
        height: usize,
        kind: BufferKind,
    ) -> Result<Self, BufferError> {
        let stride = width
            .checked_mul(L::CHANNELS)
            .ok_or(BufferError::OutOfBounds)?;
        Self::new(data, width, height, stride, kind)
    }

    pub fn new_packed_typed(
        data: &'a mut [S],
        width: usize,
        height: usize,
    ) -> Result<Self, BufferError> {
        let stride = width
            .checked_mul(L::CHANNELS)
            .ok_or(BufferError::OutOfBounds)?;
        Self::new_typed(data, width, height, stride)
    }

    pub fn kind(&self) -> Result<BufferKind, BufferError> {
        kind_for::<S, L>()
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    #[must_use]
    pub fn data(&self) -> &[S] {
        self.data
    }

    pub fn data_mut(&mut self) -> &mut [S] {
        self.data
    }

    pub fn width_bytes(&self) -> Result<usize, BufferError> {
        self.try_width_bytes()
    }

    pub fn try_width_bytes(&self) -> Result<usize, BufferError> {
        self.try_width_samples()?
            .checked_mul(size_of::<S>())
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
        L::CHANNELS
    }
}

fn kind_for<S: Sample, L: PixelLayout>() -> Result<BufferKind, BufferError> {
    match (L::CHANNELS, L::HAS_ALPHA, S::IS_FLOAT, size_of::<S>()) {
        (1, false, false, 1) => Ok(BufferKind::Gray8),
        (3, false, false, 1) => Ok(BufferKind::Rgb8),
        (4, true, false, 1) => Ok(BufferKind::Rgba8),
        (1, false, false, 2) => Ok(BufferKind::Gray16),
        (3, false, false, 2) => Ok(BufferKind::Rgb16),
        (4, true, false, 2) => Ok(BufferKind::Rgba16),
        (1, false, true, 4) => Ok(BufferKind::Gray32F),
        (3, false, true, 4) => Ok(BufferKind::Rgb32F),
        (4, true, true, 4) => Ok(BufferKind::Rgba32F),
        _ => Err(BufferError::KindMismatch),
    }
}

pub fn gray_u8<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<GrayBuffer8<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn gray_u8_packed<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
) -> Result<GrayBuffer8<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn rgb_u8<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbBuffer8<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn rgb_u8_packed<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
) -> Result<RgbBuffer8<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn rgba_u8<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbaBuffer8<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn rgba_u8_packed<'a>(
    data: &'a mut [u8],
    width: usize,
    height: usize,
) -> Result<RgbaBuffer8<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn gray_u16<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<GrayBuffer16<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn gray_u16_packed<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
) -> Result<GrayBuffer16<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn rgb_u16<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbBuffer16<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn rgb_u16_packed<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
) -> Result<RgbBuffer16<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn rgba_u16<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbaBuffer16<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn rgba_u16_packed<'a>(
    data: &'a mut [u16],
    width: usize,
    height: usize,
) -> Result<RgbaBuffer16<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn gray_32f<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<GrayBuffer32F<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn gray_32f_packed<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
) -> Result<GrayBuffer32F<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn rgb_32f<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbBuffer32F<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn rgb_32f_packed<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
) -> Result<RgbBuffer32F<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

pub fn rgba_32f<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbaBuffer32F<'a>, BufferError> {
    Buffer::new_typed(data, width, height, stride)
}

pub fn rgba_32f_packed<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
) -> Result<RgbaBuffer32F<'a>, BufferError> {
    Buffer::new_packed_typed(data, width, height)
}

#[deprecated(since = "0.2.0", note = "use dithr::gray_32f")]
pub fn gray_f32<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<GrayBuffer32F<'a>, BufferError> {
    gray_32f(data, width, height, stride)
}

#[deprecated(since = "0.2.0", note = "use dithr::rgb_32f")]
pub fn rgb_f32<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbBuffer32F<'a>, BufferError> {
    rgb_32f(data, width, height, stride)
}

#[deprecated(since = "0.2.0", note = "use dithr::rgba_32f")]
pub fn rgba_f32<'a>(
    data: &'a mut [f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbaBuffer32F<'a>, BufferError> {
    rgba_32f(data, width, height, stride)
}
