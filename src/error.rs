use crate::{buffer::BufferError, ordered::OrderedError, palette::PaletteError};

pub type DithrResult<T> = Result<T, DithrError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DithrError {
    Buffer(BufferError),
    Palette(PaletteError),
    Ordered(OrderedError),
    UnsupportedFormat(&'static str),
    InvalidArgument(&'static str),
}

impl From<BufferError> for DithrError {
    fn from(value: BufferError) -> Self {
        Self::Buffer(value)
    }
}

impl From<PaletteError> for DithrError {
    fn from(value: PaletteError) -> Self {
        Self::Palette(value)
    }
}

impl From<OrderedError> for DithrError {
    fn from(value: OrderedError) -> Self {
        Self::Ordered(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{DithrError, DithrResult};
    use crate::{BufferError, OrderedError, PaletteError};

    #[test]
    fn from_buffer_error_maps_to_dithr_error() {
        let err: DithrError = BufferError::StrideTooSmall.into();
        assert_eq!(err, DithrError::Buffer(BufferError::StrideTooSmall));
    }

    #[test]
    fn from_palette_error_maps_to_dithr_error() {
        let err: DithrError = PaletteError::TooLarge.into();
        assert_eq!(err, DithrError::Palette(PaletteError::TooLarge));
    }

    #[test]
    fn from_ordered_error_maps_to_dithr_error() {
        let err: DithrError = OrderedError::InvalidDimensions.into();
        assert_eq!(err, DithrError::Ordered(OrderedError::InvalidDimensions));
    }

    #[test]
    fn dithr_result_alias_behaves_like_result() {
        let value: DithrResult<u8> = Ok(7);
        assert_eq!(value, Ok(7));
    }
}
