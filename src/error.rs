use crate::{buffer::BufferError, ordered::OrderedError, palette::PaletteError};

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Buffer(BufferError),
    Palette(PaletteError),
    Ordered(OrderedError),
    UnsupportedFormat(&'static str),
    InvalidArgument(&'static str),
}

impl From<BufferError> for Error {
    fn from(value: BufferError) -> Self {
        Self::Buffer(value)
    }
}

impl From<PaletteError> for Error {
    fn from(value: PaletteError) -> Self {
        Self::Palette(value)
    }
}

impl From<OrderedError> for Error {
    fn from(value: OrderedError) -> Self {
        Self::Ordered(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, Result};
    use crate::{BufferError, OrderedError, PaletteError};

    #[test]
    fn from_buffer_error_maps_to_dithr_error() {
        let err: Error = BufferError::StrideTooSmall.into();
        assert_eq!(err, Error::Buffer(BufferError::StrideTooSmall));
    }

    #[test]
    fn from_palette_error_maps_to_dithr_error() {
        let err: Error = PaletteError::TooLarge.into();
        assert_eq!(err, Error::Palette(PaletteError::TooLarge));
    }

    #[test]
    fn from_ordered_error_maps_to_dithr_error() {
        let err: Error = OrderedError::InvalidDimensions.into();
        assert_eq!(err, Error::Ordered(OrderedError::InvalidDimensions));
    }

    #[test]
    fn dithr_result_alias_behaves_like_result() {
        let value: Result<u8> = Ok(7);
        assert_eq!(value, Ok(7));
    }
}
