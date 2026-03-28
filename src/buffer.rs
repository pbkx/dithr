#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    Gray8,
    Rgb8,
    Rgba8,
}

impl PixelFormat {
    #[must_use]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    StrideTooSmall,
    DataTooShort,
    ZeroDimensions,
    RowOutOfBounds,
    PixelOutOfBounds,
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
            Self::OutOfBounds => f.write_str("buffer arithmetic overflow or out-of-bounds access"),
        }
    }
}

impl std::error::Error for BufferError {}

pub struct Buffer<'a> {
    pub data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub format: PixelFormat,
}

impl<'a> Buffer<'a> {
    pub fn new(
        data: &'a mut [u8],
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
            format,
        };
        buffer.validate()?;
        Ok(buffer)
    }

    pub fn width_bytes(&self) -> Result<usize, BufferError> {
        self.try_width_bytes()
    }

    pub fn try_width_bytes(&self) -> Result<usize, BufferError> {
        self.width
            .checked_mul(self.format.bytes_per_pixel())
            .ok_or(BufferError::OutOfBounds)
    }

    pub fn required_len(&self) -> Result<usize, BufferError> {
        if self.width == 0 || self.height == 0 {
            return Err(BufferError::ZeroDimensions);
        }

        let width_bytes = self.try_width_bytes()?;

        if self.stride < width_bytes {
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

    pub fn row(&self, y: usize) -> Result<&[u8], BufferError> {
        self.try_row(y)
    }

    pub fn try_row(&self, y: usize) -> Result<&[u8], BufferError> {
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

    pub fn row_mut(&mut self, y: usize) -> Result<&mut [u8], BufferError> {
        self.try_row_mut(y)
    }

    pub fn try_row_mut(&mut self, y: usize) -> Result<&mut [u8], BufferError> {
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

        let bpp = self.format.bytes_per_pixel();
        let row_start = y.checked_mul(self.stride).ok_or(BufferError::OutOfBounds)?;
        let in_row = x.checked_mul(bpp).ok_or(BufferError::OutOfBounds)?;

        let offset = row_start
            .checked_add(in_row)
            .ok_or(BufferError::OutOfBounds)?;
        if offset >= self.data.len() {
            return Err(BufferError::DataTooShort);
        }

        Ok(offset)
    }
}
