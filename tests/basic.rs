use dithr::{
    cga_palette, grayscale_16, grayscale_2, grayscale_4, quantize_error, quantize_gray_u8,
    quantize_pixel, quantize_rgb_u8, random_in_place, threshold_in_place, Buffer, BufferError,
    IndexedImage, Palette, PaletteError, PixelFormat, QuantizeMode,
};

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

#[test]
fn quantize_gray_1bit_binary_only() {
    for value in 0_u16..=255 {
        let quantized = quantize_gray_u8(value as u8, 1);
        assert!(quantized == 0 || quantized == 255);
    }

    assert_eq!(quantize_gray_u8(127, 1), 0);
    assert_eq!(quantize_gray_u8(128, 1), 255);
}

#[test]
fn quantize_gray_8bit_identity() {
    for value in 0_u16..=255 {
        let input = value as u8;
        assert_eq!(quantize_gray_u8(input, 8), input);
    }
}

#[test]
fn quantize_rgb_1bit_channels_binary_only() {
    let quantized = quantize_rgb_u8([1, 127, 128], 1);

    assert_eq!(quantized, [0, 0, 255]);
    for channel in quantized {
        assert!(channel == 0 || channel == 255);
    }
}

#[test]
fn quantize_palette_output_is_palette_member() {
    let palette =
        Palette::new(vec![[0, 0, 0], [120, 120, 120], [255, 255, 255]]).expect("valid palette");
    let quantized = quantize_pixel(
        PixelFormat::Rgb8,
        &[100, 110, 120],
        QuantizeMode::Palette(&palette),
    );
    let rgb = [quantized[0], quantized[1], quantized[2]];

    assert!(palette.as_slice().contains(&rgb));
}

#[test]
fn quantize_single_color_uses_fg_or_black_only() {
    let fg = [12, 180, 90];
    let mode = QuantizeMode::SingleColor { fg, bits: 3 };

    for value in 0_u16..=255 {
        let quantized = quantize_pixel(PixelFormat::Gray8, &[value as u8], mode);
        let rgb = [quantized[0], quantized[1], quantized[2]];
        assert!(rgb == [0, 0, 0] || rgb == fg);
    }

    assert_eq!(
        quantize_pixel(PixelFormat::Gray8, &[0], mode),
        [0, 0, 0, 255]
    );
    assert_eq!(
        quantize_pixel(PixelFormat::Gray8, &[255], mode),
        [fg[0], fg[1], fg[2], 255]
    );
}

#[test]
fn quantize_error_sign_and_magnitude_correct() {
    let error = quantize_error(&[10, 200, 50, 255], &[20, 180, 60, 200]);

    assert_eq!(error, [-10, 20, -10, 55]);
}

#[test]
fn grayscale_2_has_len_2() {
    assert_eq!(grayscale_2().len(), 2);
}

#[test]
fn grayscale_4_has_len_4() {
    assert_eq!(grayscale_4().len(), 4);
}

#[test]
fn cga_palette_has_len_16() {
    assert_eq!(cga_palette().len(), 16);
}

#[test]
fn built_in_palettes_construct_valid_palette() {
    let palettes = [grayscale_2(), grayscale_4(), grayscale_16(), cga_palette()];

    for palette in palettes {
        assert!(!palette.is_empty());
        assert!(palette.len() <= 256);
        assert_eq!(
            Palette::new(palette.as_slice().to_vec()),
            Ok(palette.clone())
        );
    }
}

#[test]
fn threshold_gray_threshold_127_splits_expected() {
    let mut data = vec![0_u8, 64, 127, 128, 200, 255];
    let mut buffer = Buffer {
        data: &mut data,
        width: 6,
        height: 1,
        stride: 6,
        format: PixelFormat::Gray8,
    };

    threshold_in_place(&mut buffer, QuantizeMode::GrayBits(1), 127);

    assert_eq!(data, vec![0, 0, 0, 255, 255, 255]);
}

#[test]
fn threshold_rgb_uses_luma() {
    let mut data = vec![255_u8, 0, 0, 0, 255, 0, 0, 0, 255];
    let mut buffer = Buffer {
        data: &mut data,
        width: 3,
        height: 1,
        stride: 9,
        format: PixelFormat::Rgb8,
    };

    threshold_in_place(&mut buffer, QuantizeMode::RgbBits(1), 127);

    assert_eq!(data, vec![0, 0, 0, 255, 255, 255, 0, 0, 0]);
}

#[test]
fn random_same_seed_same_output() {
    let source: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut data_a = source.clone();
    let mut data_b = source;

    let mut buffer_a = Buffer {
        data: &mut data_a,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };
    let mut buffer_b = Buffer {
        data: &mut data_b,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    random_in_place(&mut buffer_a, QuantizeMode::GrayBits(1), 42, 64);
    random_in_place(&mut buffer_b, QuantizeMode::GrayBits(1), 42, 64);

    assert_eq!(data_a, data_b);
}

#[test]
fn random_different_seed_different_output() {
    let mut data_a = vec![127_u8; 64];
    let mut data_b = vec![127_u8; 64];

    let mut buffer_a = Buffer {
        data: &mut data_a,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };
    let mut buffer_b = Buffer {
        data: &mut data_b,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::Gray8,
    };

    random_in_place(&mut buffer_a, QuantizeMode::GrayBits(1), 1, 127);
    random_in_place(&mut buffer_b, QuantizeMode::GrayBits(1), 2, 127);

    assert_ne!(data_a, data_b);
}
