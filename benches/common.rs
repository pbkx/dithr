use std::sync::OnceLock;

use dithr::{
    cga_palette, grayscale_16, grayscale_2, grayscale_4, Buffer, Palette, PixelFormat, QuantizeMode,
};

pub const CUSTOM_2X2_MAP: [u8; 4] = [0, 2, 3, 1];

pub fn touch_common() {
    let _ = gray_ramp(2, 2);
    let _ = gray_checker(2, 2, 1);
    let _ = gray_noise(2, 2, 1);
    let _ = rgb_gradient(2, 2);
    let _ = rgb_noise(2, 2, 1);

    let mut gray = vec![0_u8; 4];
    let mut rgb = vec![0_u8; 12];
    let _ = gray_buffer(&mut gray, 2, 2);
    let _ = rgb_buffer(&mut rgb, 2, 2);

    let _ = palette_bw();
    let _ = palette_grayscale_4();
    let _ = palette_cga();
    let _ = palette_16_gray();

    let _ = mode_gray_1();
    let _ = mode_gray_2();
    let _ = mode_rgb_332();
    let _ = mode_palette_bw();
    let _ = mode_palette_gray4();
    let _ = mode_palette_cga();

    let _ = CUSTOM_2X2_MAP;
}

pub fn gray_ramp(width: usize, height: usize) -> Vec<u8> {
    let len = width.saturating_mul(height);
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![0];
    }

    (0..len)
        .map(|i| ((i * 255) / (len - 1)) as u8)
        .collect::<Vec<_>>()
}

pub fn gray_checker(width: usize, height: usize, block: usize) -> Vec<u8> {
    let block = block.max(1);
    let mut out = vec![0_u8; width.saturating_mul(height)];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            let value = if ((x / block) + (y / block)) % 2 == 0 {
                0
            } else {
                255
            };
            out[idx] = value;
        }
    }
    out
}

pub fn gray_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut prng = XorShift64::new(seed);
    let len = width.saturating_mul(height);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push((prng.next_u64() & 0xff) as u8);
    }
    out
}

pub fn rgb_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0_u8; width.saturating_mul(height).saturating_mul(3)];
    let max_x = width.saturating_sub(1).max(1);
    let max_y = height.saturating_sub(1).max(1);

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            let r = ((x * 255) / max_x) as u8;
            let g = ((y * 255) / max_y) as u8;
            let b = (((x + y) * 255) / (max_x + max_y).max(1)) as u8;
            out[idx] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
        }
    }

    out
}

pub fn rgb_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut prng = XorShift64::new(seed);
    let len = width.saturating_mul(height).saturating_mul(3);
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        out.push((prng.next_u64() & 0xff) as u8);
    }
    out
}

pub fn gray_buffer<'a>(data: &'a mut [u8], width: usize, height: usize) -> Buffer<'a> {
    Buffer {
        data,
        width,
        height,
        stride: width,
        format: PixelFormat::Gray8,
    }
}

pub fn rgb_buffer<'a>(data: &'a mut [u8], width: usize, height: usize) -> Buffer<'a> {
    Buffer {
        data,
        width,
        height,
        stride: width.saturating_mul(3),
        format: PixelFormat::Rgb8,
    }
}

pub fn palette_bw() -> Palette {
    grayscale_2()
}

pub fn palette_grayscale_4() -> Palette {
    grayscale_4()
}

pub fn palette_cga() -> Palette {
    cga_palette()
}

pub fn palette_16_gray() -> Palette {
    grayscale_16()
}

pub fn mode_gray_1() -> QuantizeMode<'static> {
    QuantizeMode::GrayBits(1)
}

pub fn mode_gray_2() -> QuantizeMode<'static> {
    QuantizeMode::GrayBits(2)
}

pub fn mode_rgb_332() -> QuantizeMode<'static> {
    QuantizeMode::RgbBits(3)
}

pub fn mode_palette_bw() -> QuantizeMode<'static> {
    QuantizeMode::Palette(palette_bw_ref())
}

pub fn mode_palette_gray4() -> QuantizeMode<'static> {
    QuantizeMode::Palette(palette_gray4_ref())
}

pub fn mode_palette_cga() -> QuantizeMode<'static> {
    QuantizeMode::Palette(palette_cga_ref())
}

fn palette_bw_ref() -> &'static Palette {
    static PALETTE: OnceLock<Palette> = OnceLock::new();
    PALETTE.get_or_init(palette_bw)
}

fn palette_gray4_ref() -> &'static Palette {
    static PALETTE: OnceLock<Palette> = OnceLock::new();
    PALETTE.get_or_init(palette_grayscale_4)
}

fn palette_cga_ref() -> &'static Palette {
    static PALETTE: OnceLock<Palette> = OnceLock::new();
    PALETTE.get_or_init(palette_cga)
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9e37_79b9_7f4a_7c15_u64
        } else {
            seed
        };

        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
}
