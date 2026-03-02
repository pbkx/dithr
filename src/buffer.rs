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
    OutOfBounds,
}

pub struct Buffer<'a> {
    pub data: &'a mut [u8],
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub format: PixelFormat,
}

impl<'a> Buffer<'a> {
    #[must_use]
    pub fn width_bytes(&self) -> usize {
        self.width
            .checked_mul(self.format.bytes_per_pixel())
            .expect("width in bytes overflow")
    }

    pub fn validate(&self) -> Result<(), BufferError> {
        if self.width == 0 || self.height == 0 {
            return Err(BufferError::ZeroDimensions);
        }

        let width_bytes = self
            .width
            .checked_mul(self.format.bytes_per_pixel())
            .ok_or(BufferError::OutOfBounds)?;

        if self.stride < width_bytes {
            return Err(BufferError::StrideTooSmall);
        }

        let required_len = self
            .stride
            .checked_mul(self.height)
            .ok_or(BufferError::OutOfBounds)?;

        if self.data.len() < required_len {
            return Err(BufferError::DataTooShort);
        }

        Ok(())
    }

    #[must_use]
    pub fn row(&self, y: usize) -> &[u8] {
        assert!(y < self.height, "row index out of bounds");

        let start = y.checked_mul(self.stride).expect("row start overflow");
        let end = start.checked_add(self.stride).expect("row end overflow");

        &self.data[start..end]
    }

    #[must_use]
    pub fn row_mut(&mut self, y: usize) -> &mut [u8] {
        assert!(y < self.height, "row index out of bounds");

        let start = y.checked_mul(self.stride).expect("row start overflow");
        let end = start.checked_add(self.stride).expect("row end overflow");

        &mut self.data[start..end]
    }

    #[must_use]
    pub fn pixel_offset(&self, x: usize, y: usize) -> usize {
        assert!(x < self.width, "x index out of bounds");
        assert!(y < self.height, "y index out of bounds");

        let bpp = self.format.bytes_per_pixel();
        let row_start = y.checked_mul(self.stride).expect("row start overflow");
        let in_row = x.checked_mul(bpp).expect("pixel-in-row overflow");

        row_start
            .checked_add(in_row)
            .expect("pixel offset overflow")
    }
}
