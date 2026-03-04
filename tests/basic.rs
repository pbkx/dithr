use dithr::{Buffer, BufferError, IndexedImage, Palette, PaletteError, PixelFormat};

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

#[test]
fn palette_accepts_single_color() {
    let palette = Palette::new(vec![[10, 20, 30]]).expect("single color palette should be valid");

    assert_eq!(palette.len(), 1);
    assert!(!palette.is_empty());
    assert_eq!(palette.as_slice(), &[[10, 20, 30]]);
    assert_eq!(palette.get(0), Some([10, 20, 30]));
}

#[test]
fn palette_accepts_256_colors() {
    let colors: Vec<[u8; 3]> = (0..256).map(|value| [value as u8, 0, 0]).collect();
    let palette = Palette::new(colors).expect("256 color palette should be valid");

    assert_eq!(palette.len(), 256);
}

#[test]
fn palette_rejects_empty() {
    assert_eq!(Palette::new(vec![]), Err(PaletteError::Empty));
}

#[test]
fn palette_rejects_257_colors() {
    let colors = vec![[0, 0, 0]; 257];

    assert_eq!(Palette::new(colors), Err(PaletteError::TooLarge));
}

#[test]
fn palette_nearest_rgb_returns_exact_index_for_member() {
    let palette = Palette::new(vec![[0, 0, 0], [10, 20, 30], [255, 255, 255]])
        .expect("palette should be valid");

    assert_eq!(palette.nearest_rgb([10, 20, 30]), 1);
}

#[test]
fn indexed_image_stores_dimensions_and_palette() {
    let palette = Palette::new(vec![[0, 0, 0], [255, 255, 255]]).expect("palette should be valid");
    let image = IndexedImage {
        indices: vec![0, 1, 1, 0],
        width: 2,
        height: 2,
        palette: palette.clone(),
    };

    assert_eq!(image.indices, vec![0, 1, 1, 0]);
    assert_eq!(image.width, 2);
    assert_eq!(image.height, 2);
    assert_eq!(image.palette, palette);
}
