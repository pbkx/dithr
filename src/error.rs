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

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Buffer(err) => write!(f, "buffer error: {err}"),
            Self::Palette(err) => write!(f, "palette error: {err}"),
            Self::Ordered(err) => write!(f, "ordered dithering error: {err}"),
            Self::UnsupportedFormat(message) => write!(f, "unsupported format: {message}"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Buffer(err) => Some(err),
            Self::Palette(err) => Some(err),
            Self::Ordered(err) => Some(err),
            Self::UnsupportedFormat(_) | Self::InvalidArgument(_) => None,
        }
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

    #[test]
    fn error_display_contains_variant_context() {
        let err = Error::UnsupportedFormat("Gray16");
        assert_eq!(err.to_string(), "unsupported format: Gray16");
    }

    #[test]
    fn wrapped_error_exposes_source() {
        let err: Error = BufferError::DataTooShort.into();
        let source = std::error::Error::source(&err).expect("source should be present");
        assert_eq!(source.to_string(), BufferError::DataTooShort.to_string());
    }
}
