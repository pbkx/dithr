mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_16x16_u8, gray_ramp_8x8, gray_ramp_8x8_u16,
    rgb_cube_strip, rgb_gradient_8x8, rgb_gradient_8x8_f32, rgb_gradient_8x8_u16,
};
use dithr::core::PixelLayout;
use dithr::{
    bayer_8x8_rgb16_in_place, cga_palette, floyd_steinberg_in_place, gray_u16, gray_u8,
    grayscale_16, grayscale_2, grayscale_4, levels_from_bits, quantize_error, quantize_gray_u8,
    quantize_pixel, quantize_rgb_u8, random_binary_in_place, rgb_u16, rgb_u8, rgba_u8,
    threshold_binary_in_place, Buffer, BufferError, Error, GrayBuffer16, GrayBuffer8, IndexedImage,
    IndexedImage16, IndexedImage32F, IndexedImage8, Palette, PaletteError, PixelFormat,
    QuantizeMode, RgbBuffer32F, RgbBuffer8, RgbaBuffer8,
};

#[test]
fn buffer_validate_gray_ok() {
    let mut data = vec![0_u8; 8 * 4];
    let buffer: GrayBuffer8<'_> =
        gray_u8(&mut data, 8, 4, 8).expect("valid gray u8 buffer should construct");

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_packed_constructors_match_expected_stride() {
    let mut gray = vec![0_u8; 4 * 3];
    let gray_buf = dithr::gray_u8_packed(&mut gray, 4, 3).expect("valid gray packed buffer");
    assert_eq!(gray_buf.stride(), 4);

    let mut rgb = vec![0_u16; 4 * 3 * 3];
    let rgb_buf = dithr::rgb_u16_packed(&mut rgb, 4, 3).expect("valid rgb packed buffer");
    assert_eq!(rgb_buf.stride(), 12);

    let mut rgba = vec![0.0_f32; 4 * 3 * 4];
    let rgba_buf = dithr::rgba_32f_packed(&mut rgba, 4, 3).expect("valid rgba packed buffer");
    assert_eq!(rgba_buf.stride(), 16);
}

#[test]
fn buffer_validate_rgb_ok() {
    let mut data = vec![0_u8; 10 * 3];
    let buffer: RgbBuffer8<'_> =
        rgb_u8(&mut data, 3, 3, 10).expect("valid rgb u8 buffer should construct");

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_validate_rgba_ok() {
    let mut data = vec![0_u8; 12 * 2];
    let buffer: RgbaBuffer8<'_> =
        rgba_u8(&mut data, 3, 2, 12).expect("valid rgba u8 buffer should construct");

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_validate_rejects_zero_dimensions() {
    let mut data = vec![0_u8; 4];
    let result = dithr::gray_u8(&mut data, 0, 1, 1);
    assert!(matches!(result, Err(BufferError::ZeroDimensions)));
}

#[test]
fn buffer_validate_rejects_small_stride() {
    let mut data = vec![0_u8; 8];
    let result = dithr::rgb_u8(&mut data, 3, 2, 3);
    assert!(matches!(result, Err(BufferError::StrideTooSmall)));
}

#[test]
fn buffer_validate_rejects_short_data() {
    let mut data = vec![0_u8; 7];
    let result = dithr::gray_u8(&mut data, 2, 2, 4);
    assert!(matches!(result, Err(BufferError::DataTooShort)));
}

#[test]
fn buffer_pixel_offset_matches_expected() {
    let mut data = vec![0_u8; 20];
    let buffer: RgbBuffer8<'_> =
        dithr::rgb_u8(&mut data, 3, 2, 10).expect("valid buffer should construct");

    assert_eq!(buffer.pixel_offset(2, 1), Ok(16));
}

#[test]
fn buffer_row_returns_correct_slice() {
    let mut data: Vec<u8> = (0..12).collect();
    let buffer: GrayBuffer8<'_> =
        dithr::gray_u8(&mut data, 2, 3, 4).expect("valid buffer should construct");

    assert_eq!(buffer.row(1), Ok(&[4, 5, 6, 7][..]));
}

#[test]
fn buffer_new_validates_ok() {
    let mut data = vec![0_u8; 32];
    let buffer: GrayBuffer8<'_> =
        dithr::gray_u8(&mut data, 8, 4, 8).expect("valid buffer should construct");

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_required_len_matches_stride_times_height() {
    let mut data = vec![0_u8; 40];
    let buffer: RgbBuffer8<'_> =
        dithr::rgb_u8(&mut data, 3, 4, 10).expect("valid buffer should construct");

    assert_eq!(buffer.required_len(), Ok(40));
}

#[test]
fn buffer_try_row_rejects_y_out_of_bounds() {
    let mut data = vec![0_u8; 12];
    let buffer: GrayBuffer8<'_> =
        dithr::gray_u8(&mut data, 4, 3, 4).expect("valid buffer should construct");

    assert_eq!(buffer.try_row(3), Err(BufferError::RowOutOfBounds));
}

#[test]
fn buffer_try_row_mut_rejects_y_out_of_bounds() {
    let mut data = vec![0_u8; 12];
    let mut buffer: GrayBuffer8<'_> =
        dithr::gray_u8(&mut data, 4, 3, 4).expect("valid buffer should construct");

    assert_eq!(buffer.try_row_mut(3), Err(BufferError::RowOutOfBounds));
}

#[test]
fn buffer_try_pixel_offset_rejects_x_out_of_bounds() {
    let mut data = vec![0_u8; 36];
    let buffer: RgbaBuffer8<'_> =
        Buffer::new_typed(&mut data, 3, 3, 12).expect("valid buffer should construct");

    assert_eq!(
        buffer.try_pixel_offset(3, 1),
        Err(BufferError::PixelOutOfBounds)
    );
}

#[test]
fn buffer_try_pixel_offset_rejects_y_out_of_bounds() {
    let mut data = vec![0_u8; 36];
    let buffer: RgbaBuffer8<'_> =
        Buffer::new_typed(&mut data, 3, 3, 12).expect("valid buffer should construct");

    assert_eq!(
        buffer.try_pixel_offset(1, 3),
        Err(BufferError::PixelOutOfBounds)
    );
}

#[test]
fn buffer_validate_gray16_ok() {
    let mut data = vec![0_u16; 8 * 4];
    let buffer: GrayBuffer16<'_> =
        gray_u16(&mut data, 8, 4, 8).expect("valid gray u16 buffer should construct");

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_validate_rgb32f_ok() {
    let mut data = vec![0.0_f32; 5 * 3 * 2];
    let buffer: RgbBuffer32F<'_> =
        Buffer::new_typed(&mut data, 5, 2, 15).expect("valid rgb f32 buffer should construct");

    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_kind_matches_gray_u8() {
    let mut data = vec![0_u8; 16];
    let buffer = gray_u8(&mut data, 4, 4, 4).expect("valid gray u8 buffer should construct");
    assert_eq!(
        buffer.kind().expect("kind should resolve"),
        PixelFormat::Gray8
    );
}

#[test]
fn buffer_kind_matches_rgb_u16() {
    let mut data = vec![0_u16; 4 * 4 * 3];
    let buffer = rgb_u16(&mut data, 4, 4, 12).expect("valid rgb u16 buffer should construct");
    assert_eq!(
        buffer.kind().expect("kind should resolve"),
        PixelFormat::Rgb16
    );
}

#[test]
fn buffer_kind_matches_rgba_f32() {
    let mut data = vec![0.0_f32; 4 * 4 * 4];
    let buffer =
        dithr::rgba_32f(&mut data, 4, 4, 16).expect("valid rgba f32 buffer should construct");
    assert_eq!(
        buffer.kind().expect("kind should resolve"),
        PixelFormat::Rgba32F
    );
}

#[test]
fn buffer_validate_uses_layout_not_runtime_tag() {
    let mut data = vec![0_u8; 48];
    let result: dithr::Result<dithr::RgbBuffer8<'_>> =
        Buffer::new(&mut data, 4, 4, 12, PixelFormat::Gray8).map_err(Error::from);
    assert!(matches!(
        result,
        Err(Error::Buffer(BufferError::KindMismatch))
    ));

    let mut data = vec![0_u8; 4 * 4 * 3];
    let buffer = Buffer::<u8, dithr::core::Rgb>::new_typed(&mut data, 4, 4, 12)
        .expect("typed constructor should validate");
    assert_eq!(buffer.validate(), Ok(()));
}

#[test]
fn buffer_channels_come_from_layout() {
    let mut data = vec![0_u8; 2 * 2 * 4];
    let buffer = dithr::rgba_u8(&mut data, 2, 2, 8).expect("valid rgba u8 buffer should construct");
    assert_eq!(buffer.pixel_offset(1, 1), Ok(12));
}

#[test]
fn buffer_no_redundant_format_storage_path_smoke() {
    type LegacyLike<'a> = (&'a mut [u8], usize, usize, usize, PixelFormat);
    let current = std::mem::size_of::<GrayBuffer8<'_>>();
    let legacy = std::mem::size_of::<LegacyLike<'_>>();
    assert!(current < legacy);
}

#[test]
fn buffer_pixel_access_generic_u8() {
    let mut data = vec![0_u8; 12];
    let mut buffer: RgbBuffer8<'_> =
        rgb_u8(&mut data, 2, 2, 6).expect("valid rgb u8 buffer should construct");

    {
        let pixel = buffer
            .pixel_mut(1, 1)
            .expect("pixel mut access should succeed");
        pixel.copy_from_slice(&[10, 20, 30]);
    }

    assert_eq!(
        buffer.pixel(1, 1).expect("pixel access should succeed"),
        &[10, 20, 30]
    );
}

#[test]
fn buffer_pixel_access_generic_u16() {
    let mut data = vec![0_u16; 8];
    let mut buffer: GrayBuffer16<'_> =
        gray_u16(&mut data, 2, 4, 2).expect("valid gray u16 buffer should construct");

    {
        let pixel = buffer
            .pixel_mut(1, 2)
            .expect("pixel mut access should succeed");
        pixel[0] = 42_000;
    }

    assert_eq!(
        buffer.pixel(1, 2).expect("pixel access should succeed"),
        &[42_000]
    );
}

#[test]
fn buffer_row_length_matches_stride_samples() {
    let mut data = vec![0_u16; 24];
    let buffer: GrayBuffer16<'_> =
        gray_u16(&mut data, 4, 4, 6).expect("valid gray u16 buffer should construct");

    assert_eq!(buffer.row(0).expect("row access should succeed").len(), 6);
}

#[test]
fn pixel_format_metadata_matches_expected() {
    assert_eq!(PixelFormat::Gray8.channels(), 1);
    assert!(!PixelFormat::Gray8.has_alpha());
    assert!(!PixelFormat::Gray8.is_float());
    assert_eq!(PixelFormat::Gray8.bytes_per_channel(), 1);

    assert_eq!(PixelFormat::Rgba8.channels(), 4);
    assert!(PixelFormat::Rgba8.has_alpha());
    assert!(!PixelFormat::Rgba8.is_float());
    assert_eq!(PixelFormat::Rgba8.bytes_per_channel(), 1);

    assert_eq!(PixelFormat::Gray16.channels(), 1);
    assert!(!PixelFormat::Gray16.has_alpha());
    assert!(!PixelFormat::Gray16.is_float());
    assert_eq!(PixelFormat::Gray16.bytes_per_channel(), 2);

    assert_eq!(PixelFormat::Rgb32F.channels(), 3);
    assert!(!PixelFormat::Rgb32F.has_alpha());
    assert!(PixelFormat::Rgb32F.is_float());
    assert_eq!(PixelFormat::Rgb32F.bytes_per_channel(), 4);

    assert_eq!(PixelFormat::Rgba32F.channels(), 4);
    assert!(PixelFormat::Rgba32F.has_alpha());
    assert!(PixelFormat::Rgba32F.is_float());
    assert_eq!(PixelFormat::Rgba32F.bytes_per_channel(), 4);
}

#[test]
fn palette_accepts_single_color() {
    let palette =
        Palette::new(vec![[10_u8, 20, 30]]).expect("single color palette should be valid");

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
    assert_eq!(Palette::<u8>::new(vec![]), Err(PaletteError::Empty));
}

#[test]
fn palette_rejects_257_colors() {
    let colors = vec![[0_u8, 0, 0]; 257];

    assert_eq!(Palette::new(colors), Err(PaletteError::TooLarge));
}

#[test]
fn palette_nearest_rgb_index_returns_exact_index_for_member() {
    let palette = Palette::new(vec![[0_u8, 0, 0], [10, 20, 30], [255, 255, 255]])
        .expect("palette should be valid");

    assert_eq!(palette.nearest_rgb_index([10, 20, 30]), 1);
}

#[test]
fn palette_nearest_rgb_color_returns_palette_member() {
    let palette = Palette::new(vec![[0_u8, 0, 0], [10, 20, 30], [255, 255, 255]])
        .expect("palette should be valid");

    let nearest = palette.nearest_rgb_color([11, 22, 29]);
    assert!(palette.contains(nearest));
}

#[test]
fn palette_contains_true_for_member() {
    let palette = Palette::new(vec![[0_u8, 0, 0], [10, 20, 30], [255, 255, 255]])
        .expect("palette should be valid");

    assert!(palette.contains([10, 20, 30]));
}

#[test]
fn indexed_image8_stores_palette8() {
    let palette =
        Palette::<u8>::new(vec![[0, 0, 0], [255, 255, 255]]).expect("palette should be valid");
    let image: IndexedImage8 =
        IndexedImage::new(vec![0, 1, 1, 0], 2, 2, palette.clone()).expect("valid indexed image");

    assert_eq!(image.indices(), &[0, 1, 1, 0]);
    assert_eq!(image.width(), 2);
    assert_eq!(image.height(), 2);
    assert_eq!(image.palette(), &palette);
}

#[test]
fn indexed_image16_stores_palette16() {
    let palette = Palette::<u16>::new(vec![[0, 0, 0], [65_535, 65_535, 65_535]])
        .expect("palette should be valid");
    let image: IndexedImage16 =
        IndexedImage::new(vec![0, 1, 1, 0], 2, 2, palette.clone()).expect("valid indexed image");

    assert_eq!(image.len(), 4);
    assert!(!image.is_empty());
    assert_eq!(image.palette(), &palette);
}

#[test]
fn indexed_image32f_stores_palette32f() {
    let palette =
        Palette::<f32>::new(vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]).expect("valid palette");
    let image: IndexedImage32F =
        IndexedImage::new(vec![0, 1, 1, 0], 2, 2, palette.clone()).expect("valid indexed image");

    assert_eq!(image.len(), 4);
    assert!(!image.is_empty());
    assert_eq!(image.palette(), &palette);
}

#[test]
fn indexed_image_color_at_returns_palette_color() {
    let palette = Palette::<u16>::new(vec![[0, 0, 0], [65_535, 0, 0], [0, 65_535, 0]])
        .expect("palette should be valid");
    let image: IndexedImage16 =
        IndexedImage::new(vec![0, 1, 2, 1], 2, 2, palette).expect("valid indexed image");

    assert_eq!(image.color_at(0, 0), Some([0, 0, 0]));
    assert_eq!(image.color_at(1, 0), Some([65_535, 0, 0]));
    assert_eq!(image.color_at(0, 1), Some([0, 65_535, 0]));
    assert_eq!(image.color_at(2, 1), None);
}

#[test]
fn indexed_image_new_validates_shape_and_indices() {
    let palette = Palette::<u8>::new(vec![[0, 0, 0], [255, 255, 255]]).expect("valid palette");
    let valid = IndexedImage::new(vec![0, 1, 1, 0], 2, 2, palette.clone());
    assert!(valid.is_ok());

    let bad_len = IndexedImage::new(vec![0, 1, 1], 2, 2, palette.clone());
    assert_eq!(
        bad_len,
        Err(Error::InvalidArgument(
            "indexed image index count must match dimensions"
        ))
    );

    let bad_index = IndexedImage::new(vec![0, 2, 1, 0], 2, 2, palette);
    assert_eq!(
        bad_index,
        Err(Error::InvalidArgument(
            "indexed image contains out-of-range palette indices"
        ))
    );
}

#[test]
fn palette_length_256_still_allows_u8_indices() {
    let colors: Vec<[u8; 3]> = (0..256)
        .map(|value| {
            let v = value as u8;
            [v, v, v]
        })
        .collect();
    let palette = Palette::<u8>::new(colors).expect("palette should be valid");
    let image: IndexedImage8 =
        IndexedImage::new(vec![0, 127, 255], 3, 1, palette).expect("valid indexed image");

    assert_eq!(image.color_at(0, 0), Some([0, 0, 0]));
    assert_eq!(image.color_at(1, 0), Some([127, 127, 127]));
    assert_eq!(image.color_at(2, 0), Some([255, 255, 255]));
}

#[test]
fn quantize_gray_1bit_binary_only() {
    for value in 0_u16..=255 {
        let quantized = quantize_gray_u8(value as u8, 1).expect("valid bit depth");
        assert!(quantized == 0 || quantized == 255);
    }

    assert_eq!(quantize_gray_u8(127, 1), Ok(0));
    assert_eq!(quantize_gray_u8(128, 1), Ok(255));
}

#[test]
fn gray_bits_1_maps_to_gray_levels_2() {
    assert_eq!(levels_from_bits(1), Ok(2));
    assert_eq!(
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        QuantizeMode::GrayLevels(2)
    );
}

#[test]
fn rgb_bits_8_maps_to_rgb_levels_256() {
    assert_eq!(levels_from_bits(8), Ok(256));
    assert_eq!(
        QuantizeMode::rgb_bits(8).expect("valid bit depth"),
        QuantizeMode::RgbLevels(256)
    );
}

#[test]
fn gray_bits_rejects_out_of_range_values() {
    assert_eq!(
        QuantizeMode::gray_bits(0),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
    assert_eq!(
        QuantizeMode::gray_bits(9),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
}

#[test]
fn rgb_bits_rejects_out_of_range_values() {
    assert_eq!(
        QuantizeMode::rgb_bits(0),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
    assert_eq!(
        QuantizeMode::rgb_bits(9),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
}

#[test]
fn quantize_mode_gray_levels_is_canonical() {
    let mode = QuantizeMode::gray_bits(4).expect("valid bit depth");
    assert!(matches!(mode, QuantizeMode::GrayLevels(_)));
}

#[test]
fn quantize_mode_constructors_validate_levels() {
    let ok_gray = QuantizeMode::<u8>::gray_levels(2).expect("valid levels");
    let ok_rgb = QuantizeMode::<u16>::rgb_levels(256).expect("valid levels");
    let ok_single = QuantizeMode::<f32>::single_color([1.0, 0.5, 0.25], 8).expect("valid levels");
    assert!(matches!(ok_gray, QuantizeMode::GrayLevels(2)));
    assert!(matches!(ok_rgb, QuantizeMode::RgbLevels(256)));
    assert!(matches!(ok_single, QuantizeMode::SingleColor { .. }));

    let err = QuantizeMode::<u8>::gray_levels(1);
    assert_eq!(
        err,
        Err(Error::InvalidArgument(
            "quantization levels must be in 2..=65535"
        ))
    );
}

#[test]
fn quantize_gray_8bit_identity() {
    for value in 0_u16..=255 {
        let input = value as u8;
        assert_eq!(quantize_gray_u8(input, 8), Ok(input));
    }
}

#[test]
fn quantize_rgb_1bit_channels_binary_only() {
    let quantized = quantize_rgb_u8([1, 127, 128], 1).expect("valid bit depth");

    assert_eq!(quantized, [0, 0, 255]);
    for channel in quantized {
        assert!(channel == 0 || channel == 255);
    }
}

#[test]
fn quantize_gray_rejects_invalid_bits() {
    assert_eq!(
        quantize_gray_u8(42, 0),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
    assert_eq!(
        quantize_gray_u8(42, 9),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
}

#[test]
fn quantize_rgb_rejects_invalid_bits() {
    assert_eq!(
        quantize_rgb_u8([10, 20, 30], 0),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
    assert_eq!(
        quantize_rgb_u8([10, 20, 30], 10),
        Err(Error::InvalidArgument("quantization bits must be in 1..=8"))
    );
}

#[test]
fn quantize_gray_u16_binary_only() {
    for value in [0_u16, 1, 12_345, 32_767, 65_534, 65_535] {
        let quantized = dithr::quantize_gray(value, 2).expect("valid levels");
        assert!(quantized == 0 || quantized == 65_535);
    }
}

#[test]
fn quantize_gray_u16_levels_binary_only() {
    for value in [0_u16, 1, 12_345, 32_767, 65_534, 65_535] {
        let quantized = dithr::quantize_gray(value, 2).expect("valid levels");
        assert!(quantized == 0 || quantized == 65_535);
    }
}

#[test]
fn quantize_rgb_u16_binary_only() {
    let quantized = dithr::quantize_rgb([1_u16, 32_767, 65_535], 2).expect("valid levels");
    for channel in quantized {
        assert!(channel == 0 || channel == 65_535);
    }
}

#[test]
fn quantize_gray_f32_identity_high_levels() {
    let values = [0.0_f32, 0.1, 0.5, 0.9, 1.0];
    for value in values {
        let quantized = dithr::quantize_gray(value, 65_535).expect("valid levels");
        assert!((quantized - value).abs() < 2e-5);
    }
}

#[test]
fn quantize_rgb_f32_levels_identity_high_resolution() {
    let rgb = [0.15_f32, 0.5_f32, 0.85_f32];
    let quantized = dithr::quantize_rgb(rgb, 65_535).expect("valid levels");
    assert!((quantized[0] - rgb[0]).abs() < 2e-5);
    assert!((quantized[1] - rgb[1]).abs() < 2e-5);
    assert!((quantized[2] - rgb[2]).abs() < 2e-5);
}

#[test]
fn palette_u16_exact_member_roundtrip() {
    let palette = dithr::Palette::<u16>::new(vec![
        [0, 0, 0],
        [10_000, 20_000, 30_000],
        [65_535, 65_535, 65_535],
    ])
    .expect("palette should be valid");
    assert_eq!(palette.nearest_rgb_index([10_000, 20_000, 30_000]), 1);
}

#[test]
fn palette_f32_exact_member_roundtrip() {
    let palette =
        dithr::Palette::<f32>::new(vec![[0.0, 0.0, 0.0], [0.25, 0.5, 0.75], [1.0, 1.0, 1.0]])
            .expect("palette should be valid");
    assert_eq!(palette.nearest_rgb_index([0.25, 0.5, 0.75]), 1);
}

#[test]
fn quantize_error_returns_unit_space_difference() {
    let error = quantize_error::<u16, dithr::core::Rgb>(&[0, 32_768, 65_535], &[65_535, 32_768, 0])
        .expect("quantize error should succeed");
    assert!((error[0] + 1.0).abs() < 1e-6);
    assert!((error[1] - 0.0).abs() < 1e-6);
    assert!((error[2] - 1.0).abs() < 1e-6);
    assert!((error[3] - 0.0).abs() < 1e-6);
}

#[test]
fn quantize_palette_output_is_palette_member() {
    let palette =
        Palette::new(vec![[0, 0, 0], [120, 120, 120], [255, 255, 255]]).expect("valid palette");
    let quantized =
        quantize_pixel::<u8, dithr::core::Rgb>(&[100, 110, 120], QuantizeMode::Palette(&palette))
            .expect("palette quantization should succeed");
    let rgb = [quantized[0], quantized[1], quantized[2]];

    assert!(palette.as_slice().contains(&rgb));
}

#[test]
fn quantize_single_color_preserves_intermediate_levels() {
    let fg = [12, 180, 90];
    let mode = QuantizeMode::SingleColor { fg, levels: 8 };
    let mut saw_mid = false;

    for value in 0_u16..=255 {
        let quantized = quantize_pixel::<u8, dithr::core::Gray>(&[value as u8], mode);
        let quantized = quantized.expect("single-color quantization should succeed");
        let rgb = [quantized[0], quantized[1], quantized[2]];
        assert!(rgb[0] <= fg[0]);
        assert!(rgb[1] <= fg[1]);
        assert!(rgb[2] <= fg[2]);

        if rgb != [0, 0, 0] && rgb != fg {
            saw_mid = true;
        }
    }

    assert_eq!(
        quantize_pixel::<u8, dithr::core::Gray>(&[0], mode)
            .expect("single-color quantization should succeed"),
        [0, 0, 0, 255]
    );
    assert_eq!(
        quantize_pixel::<u8, dithr::core::Gray>(&[255], mode)
            .expect("single-color quantization should succeed"),
        [fg[0], fg[1], fg[2], 255]
    );
    assert!(saw_mid);
}

#[test]
fn single_color_levels_generic_u16() {
    let fg = [50_000_u16, 32_767_u16, 16_383_u16];
    let mode = QuantizeMode::SingleColor { fg, levels: 4 };
    let quantized = quantize_pixel::<u16, dithr::core::Gray>(&[32_768], mode)
        .expect("single-color quantization should succeed");
    assert!(quantized[0] <= fg[0]);
    assert!(quantized[1] <= fg[1]);
    assert!(quantized[2] <= fg[2]);
}

#[test]
fn single_color_levels_generic_f32() {
    let mode = QuantizeMode::SingleColor {
        fg: [1.0_f32, 0.5_f32, 0.25_f32],
        levels: 8,
    };
    let quantized = quantize_pixel::<f32, dithr::core::Rgb>(&[0.25, 0.5, 0.75], mode)
        .expect("single-color quantization should succeed");
    assert!((0.0..=1.0).contains(&quantized[0]));
    assert!((0.0..=0.5).contains(&quantized[1]));
    assert!((0.0..=0.25).contains(&quantized[2]));
}

#[test]
fn quantize_error_sign_and_magnitude_correct() {
    let error = quantize_error::<u8, dithr::core::Rgba>(&[10, 200, 50, 255], &[20, 180, 60, 200])
        .expect("quantize error should succeed");

    assert!((error[0] + 10.0 / 255.0).abs() < 1e-6);
    assert!((error[1] - 20.0 / 255.0).abs() < 1e-6);
    assert!((error[2] + 10.0 / 255.0).abs() < 1e-6);
    assert!((error[3] - 55.0 / 255.0).abs() < 1e-6);
}

#[test]
fn quantize_pixel_rejects_short_slice_for_format() {
    let result = quantize_pixel::<u8, dithr::core::Rgb>(
        &[10, 20],
        QuantizeMode::rgb_bits(2).expect("valid bit depth"),
    );
    assert_eq!(
        result,
        Err(Error::InvalidArgument(
            "pixel slice length does not match layout"
        ))
    );
}

#[test]
fn quantize_error_rejects_mismatched_lengths() {
    let result = quantize_error::<u8, dithr::core::Rgb>(&[10, 20, 30], &[10, 20]);
    assert_eq!(
        result,
        Err(Error::InvalidArgument(
            "original and quantized pixel lengths must match"
        ))
    );
}

#[test]
fn quantize_error_rejects_empty_slice() {
    let result = quantize_error::<u8, dithr::core::Gray>(&[], &[]);
    assert_eq!(
        result,
        Err(Error::InvalidArgument(
            "pixel slice length does not match layout"
        ))
    );
}

#[test]
fn quantize_error_rejects_layouts_wider_than_four_channels() {
    #[derive(Clone, Copy)]
    struct FiveChannel;

    impl PixelLayout for FiveChannel {
        const CHANNELS: usize = 5;
        const COLOR_CHANNELS: usize = 5;
        const HAS_ALPHA: bool = false;
        const IS_GRAY: bool = false;
    }

    let result = quantize_error::<u8, FiveChannel>(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4]);
    assert_eq!(
        result,
        Err(Error::UnsupportedFormat(
            "quantize error supports layouts with up to 4 channels"
        ))
    );
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
fn threshold_binary_gray_threshold_127_splits_expected() {
    let mut data = vec![0_u8, 64, 127, 128, 200, 255];
    let mut buffer = dithr::gray_u8(&mut data, 6, 1, 6).expect("valid buffer should construct");

    threshold_binary_in_place(
        &mut buffer,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        127,
    )
    .expect("threshold binary should succeed");

    assert_eq!(data, vec![0, 0, 0, 255, 255, 255]);
}

#[test]
fn threshold_binary_rgb_uses_luma() {
    let mut data = vec![255_u8, 0, 0, 0, 255, 0, 0, 0, 255];
    let mut buffer = dithr::rgb_u8(&mut data, 3, 1, 9).expect("valid buffer should construct");

    threshold_binary_in_place(
        &mut buffer,
        QuantizeMode::rgb_bits(1).expect("valid bit depth"),
        127,
    )
    .expect("threshold binary should succeed");

    assert_eq!(data, vec![0, 0, 0, 255, 255, 255, 0, 0, 0]);
}

#[test]
fn random_binary_same_seed_same_output() {
    let source: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut data_a = source.clone();
    let mut data_b = source;

    let mut buffer_a = dithr::gray_u8(&mut data_a, 8, 8, 8).expect("valid buffer should construct");
    let mut buffer_b = dithr::gray_u8(&mut data_b, 8, 8, 8).expect("valid buffer should construct");

    random_binary_in_place(
        &mut buffer_a,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        42,
        64,
    )
    .expect("random binary should succeed");
    random_binary_in_place(
        &mut buffer_b,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        42,
        64,
    )
    .expect("random binary should succeed");

    assert_eq!(data_a, data_b);
}

#[test]
fn random_binary_different_seed_different_output() {
    let mut data_a = vec![127_u8; 64];
    let mut data_b = vec![127_u8; 64];

    let mut buffer_a = dithr::gray_u8(&mut data_a, 8, 8, 8).expect("valid buffer should construct");
    let mut buffer_b = dithr::gray_u8(&mut data_b, 8, 8, 8).expect("valid buffer should construct");

    random_binary_in_place(
        &mut buffer_a,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        1,
        127,
    )
    .expect("random binary should succeed");
    random_binary_in_place(
        &mut buffer_b,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        2,
        127,
    )
    .expect("random binary should succeed");

    assert_ne!(data_a, data_b);
}

#[cfg(feature = "rayon")]
#[test]
fn threshold_binary_parallel_matches_sequential() {
    use dithr::threshold_binary_in_place_par;

    let mut seq = gray_ramp_16x16();
    let mut par = seq.clone();
    let mut seq_buffer =
        dithr::gray_u8(&mut seq, 16, 16, 16).expect("valid buffer should construct");
    let mut par_buffer =
        dithr::gray_u8(&mut par, 16, 16, 16).expect("valid buffer should construct");

    threshold_binary_in_place(
        &mut seq_buffer,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        127,
    )
    .expect("sequential threshold should succeed");
    threshold_binary_in_place_par(
        &mut par_buffer,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        127,
    )
    .expect("parallel threshold should succeed");

    assert_eq!(seq, par);
}

#[cfg(feature = "rayon")]
#[test]
fn random_binary_parallel_matches_sequential_fixed_seed() {
    use dithr::random_binary_in_place_par;

    let mut seq = gray_ramp_16x16();
    let mut par = seq.clone();
    let mut seq_buffer =
        dithr::gray_u8(&mut seq, 16, 16, 16).expect("valid buffer should construct");
    let mut par_buffer =
        dithr::gray_u8(&mut par, 16, 16, 16).expect("valid buffer should construct");

    random_binary_in_place(
        &mut seq_buffer,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        42,
        64,
    )
    .expect("sequential random should succeed");
    random_binary_in_place_par(
        &mut par_buffer,
        QuantizeMode::gray_bits(1).expect("valid bit depth"),
        42,
        64,
    )
    .expect("parallel random should succeed");

    assert_eq!(seq, par);
}

#[test]
fn fixture_builders_are_deterministic() {
    let gray8 = gray_ramp_8x8();
    let gray16 = gray_ramp_16x16();
    let gray8_u16 = gray_ramp_8x8_u16();
    let checker = checker_8x8();
    let gradient = rgb_gradient_8x8();
    let gradient_u16 = rgb_gradient_8x8_u16();
    let gradient_f32 = rgb_gradient_8x8_f32();
    let cube = rgb_cube_strip();

    assert_eq!(gray8.len(), 64);
    assert_eq!(gray16.len(), 256);
    assert_eq!(gray16, gray_ramp_16x16_u8());
    assert_eq!(gray8_u16.len(), 64);
    assert_eq!(checker.len(), 64);
    assert_eq!(gradient.len(), 8 * 8 * 3);
    assert_eq!(gradient_u16.len(), 8 * 8 * 3);
    assert_eq!(gradient_f32.len(), 8 * 8 * 3);
    assert_eq!(cube.len(), 27 * 3);

    assert_eq!(checker.iter().filter(|&&value| value == 0).count(), 32);
    assert_eq!(checker.iter().filter(|&&value| value == 255).count(), 32);
    assert_eq!(gray8_u16.first().copied(), Some(0));
    assert_eq!(gray8_u16.last().copied(), Some(65_535));
    assert_eq!(gradient_u16.iter().copied().min(), Some(0));
    assert_eq!(gradient_u16.iter().copied().max(), Some(65_535));
    assert!(gradient_f32.iter().all(|&v| (0.0..=1.0).contains(&v)));

    assert_eq!(fnv1a64(&gray8), fnv1a64(&gray_ramp_8x8()));
    assert_eq!(fnv1a64(&gray16), fnv1a64(&gray_ramp_16x16()));
    assert_eq!(fnv1a64(&checker), fnv1a64(&checker_8x8()));
    assert_eq!(fnv1a64(&gradient), fnv1a64(&rgb_gradient_8x8()));
    assert_eq!(fnv1a64(&cube), fnv1a64(&rgb_cube_strip()));
}

#[test]
fn rgb16_example_style_smoke() {
    let width = 8_usize;
    let height = 8_usize;
    let mut data = vec![0_u16; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = (x * 65_535 / (width - 1)) as u16;
            data[offset + 1] = (y * 65_535 / (height - 1)) as u16;
            data[offset + 2] = ((x + y) * 65_535 / (width + height - 2)) as u16;
        }
    }

    let mut buffer = rgb_u16(&mut data, width, height, width * 3).expect("valid rgb16 buffer");
    bayer_8x8_rgb16_in_place(&mut buffer, 2).expect("bayer rgb16 should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn rgb32f_example_style_smoke() {
    let width = 8_usize;
    let height = 8_usize;
    let mut data = vec![0.0_f32; width * height * 3];

    for y in 0..height {
        for x in 0..width {
            let offset = (y * width + x) * 3;
            data[offset] = x as f32 / (width - 1) as f32;
            data[offset + 1] = y as f32 / (height - 1) as f32;
            data[offset + 2] = (x + y) as f32 / (width + height - 2) as f32;
        }
    }

    let mut buffer: RgbBuffer32F<'_> =
        Buffer::new_typed(&mut data, width, height, width * 3).expect("valid rgb32f buffer");
    floyd_steinberg_in_place(&mut buffer, QuantizeMode::RgbLevels(2))
        .expect("floyd-steinberg should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}
