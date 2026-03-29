use std::sync::OnceLock;

use criterion::{black_box, measurement::WallTime, BatchSize, BenchmarkGroup, Throughput};
use dithr::{
    cga_palette, gray_u16, gray_u8, grayscale_16, grayscale_2, grayscale_4, rgb_u8, GrayBuffer16,
    GrayBuffer8, Palette, QuantizeMode, Result as DithrResult, RgbBuffer8,
};

pub const CUSTOM_2X2_MAP: [u8; 4] = [0, 2, 3, 1];

pub const SEQUENTIAL_ALGORITHMS: [&str; 35] = [
    "threshold_binary_in_place",
    "random_binary_in_place",
    "threshold_in_place",
    "random_in_place",
    "bayer_2x2_in_place",
    "bayer_4x4_in_place",
    "bayer_8x8_in_place",
    "bayer_16x16_in_place",
    "cluster_dot_4x4_in_place",
    "cluster_dot_8x8_in_place",
    "custom_ordered_in_place",
    "yliluoma_1_in_place",
    "yliluoma_2_in_place",
    "yliluoma_3_in_place",
    "floyd_steinberg_in_place",
    "false_floyd_steinberg_in_place",
    "jarvis_judice_ninke_in_place",
    "stucki_in_place",
    "burkes_in_place",
    "sierra_in_place",
    "two_row_sierra_in_place",
    "sierra_lite_in_place",
    "stevenson_arce_in_place",
    "atkinson_in_place",
    "fan_in_place",
    "shiau_fan_in_place",
    "shiau_fan_2_in_place",
    "ostromoukhov_in_place",
    "zhou_fang_in_place",
    "gradient_based_error_diffusion_in_place",
    "riemersma_in_place",
    "knuth_dot_diffusion_in_place",
    "direct_binary_search_in_place",
    "lattice_boltzmann_in_place",
    "electrostatic_halftoning_in_place",
];

#[cfg(feature = "rayon")]
pub const PARALLEL_ALGORITHMS: [&str; 9] = [
    "bayer_2x2_in_place_par",
    "bayer_4x4_in_place_par",
    "bayer_8x8_in_place_par",
    "bayer_16x16_in_place_par",
    "cluster_dot_4x4_in_place_par",
    "cluster_dot_8x8_in_place_par",
    "custom_ordered_in_place_par",
    "threshold_binary_in_place_par",
    "random_binary_in_place_par",
];

pub fn coverage_count() -> usize {
    #[cfg(feature = "rayon")]
    let count = SEQUENTIAL_ALGORITHMS.len() + PARALLEL_ALGORITHMS.len();
    #[cfg(not(feature = "rayon"))]
    let count = SEQUENTIAL_ALGORITHMS.len();

    count
}

pub fn touch_common() {
    let _ = gray_ramp(2, 2);
    let _ = gray_ramp_u16(2, 2);
    let _ = gray_checker(2, 2, 1);
    let _ = gray_noise(2, 2, 1);
    let _ = rgb_gradient(2, 2);
    let _ = rgb_noise(2, 2, 1);

    let mut gray = vec![0_u8; 4];
    let mut gray16 = vec![0_u16; 4];
    let mut rgb = vec![0_u8; 12];
    let _ = gray_buffer(&mut gray, 2, 2);
    let _ = gray_buffer_u16(&mut gray16, 2, 2);
    let _ = rgb_buffer(&mut rgb, 2, 2);

    let _ = palette_bw();
    let _ = palette_grayscale_4();
    let _ = palette_cga();
    let _ = palette_16_gray();

    let _ = mode_gray_1();
    let _ = mode_gray_2();
    let _ = mode_rgb_bits3();
    let _ = mode_palette_bw();
    let _ = mode_palette_gray4();
    let _ = mode_palette_cga();

    let _ = CUSTOM_2X2_MAP;
    let _ = coverage_count();
}

#[allow(dead_code)]
pub fn set_gray_throughput(group: &mut BenchmarkGroup<'_, WallTime>, width: usize, height: usize) {
    group.throughput(Throughput::Bytes(width.saturating_mul(height) as u64));
}

pub fn set_rgb_throughput(group: &mut BenchmarkGroup<'_, WallTime>, width: usize, height: usize) {
    let bytes = width.saturating_mul(height).saturating_mul(3);
    group.throughput(Throughput::Bytes(bytes as u64));
}

#[allow(dead_code)]
pub fn bench_gray_case<F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    fixture: &[u8],
    width: usize,
    height: usize,
    mut run: F,
) where
    F: FnMut(&mut GrayBuffer8<'_>) -> DithrResult<()>,
{
    group.bench_function(name, |b| {
        b.iter_batched(
            || fixture.to_vec(),
            |mut data| {
                let mut buffer = gray_buffer(&mut data, width, height);
                run(&mut buffer).expect("benchmark case failed");
                black_box(data);
            },
            BatchSize::SmallInput,
        );
    });
}

pub fn bench_rgb_case<F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    fixture: &[u8],
    width: usize,
    height: usize,
    mut run: F,
) where
    F: FnMut(&mut RgbBuffer8<'_>) -> DithrResult<()>,
{
    group.bench_function(name, |b| {
        b.iter_batched(
            || fixture.to_vec(),
            |mut data| {
                let mut buffer = rgb_buffer(&mut data, width, height);
                run(&mut buffer).expect("benchmark case failed");
                black_box(data);
            },
            BatchSize::SmallInput,
        );
    });
}

#[cfg(feature = "rayon")]
#[allow(dead_code)]
pub fn assert_gray_seq_par_equal<F, G>(fixture: &[u8], width: usize, height: usize, seq: F, par: G)
where
    F: Fn(&mut GrayBuffer8<'_>) -> DithrResult<()>,
    G: Fn(&mut GrayBuffer8<'_>) -> DithrResult<()>,
{
    let mut seq_data = fixture.to_vec();
    let mut par_data = fixture.to_vec();

    let mut seq_buffer = gray_buffer(&mut seq_data, width, height);
    let mut par_buffer = gray_buffer(&mut par_data, width, height);

    seq(&mut seq_buffer).expect("sequential benchmark sanity check failed");
    par(&mut par_buffer).expect("parallel benchmark sanity check failed");
    assert_eq!(seq_data, par_data);
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
        .map(|index| ((index * 255) / (len - 1)) as u8)
        .collect()
}

pub fn gray_ramp_u16(width: usize, height: usize) -> Vec<u16> {
    let len = width.saturating_mul(height);
    if len == 0 {
        return Vec::new();
    }
    if len == 1 {
        return vec![0];
    }

    (0..len)
        .map(|index| ((index * 65_535) / (len - 1)) as u16)
        .collect()
}

pub fn gray_checker(width: usize, height: usize, block: usize) -> Vec<u8> {
    let block = block.max(1);
    let mut out = vec![0_u8; width.saturating_mul(height)];
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            out[idx] = if ((x / block) + (y / block)) % 2 == 0 {
                0
            } else {
                255
            };
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
            out[idx] = ((x * 255) / max_x) as u8;
            out[idx + 1] = ((y * 255) / max_y) as u8;
            out[idx + 2] = (((x + y) * 255) / (max_x + max_y).max(1)) as u8;
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

pub fn gray_buffer<'a>(data: &'a mut [u8], width: usize, height: usize) -> GrayBuffer8<'a> {
    gray_u8(data, width, height, width).expect("valid Gray8 benchmark buffer")
}

pub fn gray_buffer_u16<'a>(data: &'a mut [u16], width: usize, height: usize) -> GrayBuffer16<'a> {
    gray_u16(data, width, height, width).expect("valid Gray16 benchmark buffer")
}

pub fn rgb_buffer<'a>(data: &'a mut [u8], width: usize, height: usize) -> RgbBuffer8<'a> {
    rgb_u8(data, width, height, width.saturating_mul(3)).expect("valid Rgb8 benchmark buffer")
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

#[allow(dead_code)]
pub fn mode_gray_levels2_u16() -> QuantizeMode<'static, u16> {
    QuantizeMode::GrayLevels(2)
}

pub fn mode_rgb_bits3() -> QuantizeMode<'static> {
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

#[allow(dead_code)]
pub fn bench_gray_case_u16<F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    fixture: &[u16],
    width: usize,
    height: usize,
    mut run: F,
) where
    F: FnMut(&mut GrayBuffer16<'_>) -> DithrResult<()>,
{
    group.bench_function(name, |b| {
        b.iter_batched(
            || fixture.to_vec(),
            |mut data| {
                let mut buffer = gray_buffer_u16(&mut data, width, height);
                run(&mut buffer).expect("benchmark case failed");
                black_box(data);
            },
            BatchSize::SmallInput,
        );
    });
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
