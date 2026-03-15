use dithr::data::FLOYD_STEINBERG;
use dithr::diffusion::error_diffuse_in_place;
use dithr::{
    burkes_in_place, false_floyd_steinberg_in_place, floyd_steinberg_in_place,
    jarvis_judice_ninke_in_place, sierra_in_place, stucki_in_place, Buffer, Palette, PixelFormat,
    QuantizeMode,
};

#[test]
fn diffusion_engine_gray_binary_output_for_graybits1() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    error_diffuse_in_place(&mut buffer, QuantizeMode::GrayBits(1), &FLOYD_STEINBERG);

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn floyd_steinberg_gray_bits1_binary_only() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn false_floyd_steinberg_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    false_floyd_steinberg_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn jarvis_judice_ninke_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    jarvis_judice_ninke_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn stucki_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    stucki_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn burkes_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    burkes_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn sierra_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    sierra_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn diffusion_engine_rgb_palette_output_is_palette_member() {
    let mut data = vec![
        16, 24, 32, 48, 56, 64, 80, 88, 96, 112, 120, 128, 144, 152, 160, 176, 184, 192, 208, 216,
        224, 240, 248, 255, 12, 68, 124, 28, 84, 140, 44, 100, 156, 60, 116, 172, 76, 132, 188, 92,
        148, 204, 108, 164, 220, 124, 180, 236,
    ];
    let palette = Palette::new(vec![
        [0, 0, 0],
        [255, 255, 255],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ])
    .expect("palette should be valid");
    let mut buffer = Buffer {
        data: &mut data,
        width: 4,
        height: 4,
        stride: 12,
        format: PixelFormat::Rgb8,
    };

    error_diffuse_in_place(
        &mut buffer,
        QuantizeMode::Palette(&palette),
        &FLOYD_STEINBERG,
    );

    for chunk in data.chunks_exact(3) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        assert!(palette.as_slice().contains(&rgb));
    }
}

#[test]
fn diffusion_engine_preserves_alpha() {
    let mut data: Vec<u8> = (0_u16..120)
        .map(|value| ((value * 17 + 9) % 256) as u8)
        .collect();
    let before_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 6,
        height: 5,
        stride: 24,
        format: PixelFormat::Rgba8,
    };

    error_diffuse_in_place(&mut buffer, QuantizeMode::RgbBits(2), &FLOYD_STEINBERG);

    let after_alpha: Vec<u8> = data.iter().skip(3).step_by(4).copied().collect();
    assert_eq!(before_alpha, after_alpha);
}

#[test]
fn diffusion_engine_does_not_panic_on_1x1() {
    let mut data = vec![128_u8];
    let mut buffer = Buffer {
        data: &mut data,
        width: 1,
        height: 1,
        stride: 1,
        format: PixelFormat::Gray8,
    };

    error_diffuse_in_place(&mut buffer, QuantizeMode::GrayBits(1), &FLOYD_STEINBERG);

    assert_eq!(data.len(), 1);
}

#[test]
fn diffusion_engine_is_deterministic() {
    let seed_data: Vec<u8> = (0_u16..144)
        .map(|value| ((value * 29 + 3) % 256) as u8)
        .collect();
    let mut a = seed_data.clone();
    let mut b = seed_data;

    let mut buffer_a = Buffer {
        data: &mut a,
        width: 8,
        height: 6,
        stride: 24,
        format: PixelFormat::Rgb8,
    };
    let mut buffer_b = Buffer {
        data: &mut b,
        width: 8,
        height: 6,
        stride: 24,
        format: PixelFormat::Rgb8,
    };

    error_diffuse_in_place(&mut buffer_a, QuantizeMode::RgbBits(3), &FLOYD_STEINBERG);
    error_diffuse_in_place(&mut buffer_b, QuantizeMode::RgbBits(3), &FLOYD_STEINBERG);

    assert_eq!(a, b);
}
