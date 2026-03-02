use dithr::{Buffer, BufferError, PixelFormat};

#[test]
fn buffer_validate_gray_ok() {
    let mut data = vec![0_u8; 8 * 4];
    let buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 4,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_validate_rgb_ok() {
    let mut data = vec![0_u8; 10 * 3];
    let buffer = Buffer {
        data: &mut data,
        width: 3,
        height: 3,
        stride: 10,
        format: PixelFormat::Rgb8,
    };

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_validate_rgba_ok() {
    let mut data = vec![0_u8; 12 * 2];
    let buffer = Buffer {
        data: &mut data,
        width: 3,
        height: 2,
        stride: 12,
        format: PixelFormat::Rgba8,
    };

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_validate_rejects_zero_dimensions() {
    let mut data = vec![0_u8; 4];
    let buffer = Buffer {
        data: &mut data,
        width: 0,
        height: 1,
        stride: 1,
        format: PixelFormat::Gray8,
    };

    assert_eq!(buffer.validate(), Err(BufferError::ZeroDimensions));
}

#[test]
fn buffer_validate_rejects_small_stride() {
    let mut data = vec![0_u8; 8];
    let buffer = Buffer {
        data: &mut data,
        width: 3,
        height: 2,
        stride: 3,
        format: PixelFormat::Rgb8,
    };

    assert_eq!(buffer.validate(), Err(BufferError::StrideTooSmall));
}

#[test]
fn buffer_validate_rejects_short_data() {
    let mut data = vec![0_u8; 7];
    let buffer = Buffer {
        data: &mut data,
        width: 2,
        height: 2,
        stride: 4,
        format: PixelFormat::Gray8,
    };

    assert_eq!(buffer.validate(), Err(BufferError::DataTooShort));
}

#[test]
fn buffer_pixel_offset_matches_expected() {
    let mut data = vec![0_u8; 20];
    let buffer = Buffer {
        data: &mut data,
        width: 3,
        height: 2,
        stride: 10,
        format: PixelFormat::Rgb8,
    };

    assert_eq!(buffer.pixel_offset(2, 1), 16);
}

#[test]
fn buffer_row_returns_correct_slice() {
    let mut data: Vec<u8> = (0..12).collect();
    let buffer = Buffer {
        data: &mut data,
        width: 2,
        height: 3,
        stride: 4,
        format: PixelFormat::Gray8,
    };

    assert_eq!(buffer.row(1), &[4, 5, 6, 7]);
}
