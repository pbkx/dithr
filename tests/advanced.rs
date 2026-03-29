mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_8x8, rgb_cube_strip, rgb_gradient_8x8,
};
use dithr::{
    direct_binary_search_in_place, electrostatic_halftoning_in_place, knuth_dot_diffusion_in_place,
    lattice_boltzmann_in_place, riemersma_in_place, Buffer, PixelFormat, QuantizeMode,
};

#[test]
fn dot_diffusion_class_matrix_valid() {
    let matrix = [
        [0_u8, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ];
    let mut seen = [false; 16];

    for row in matrix {
        for class in row {
            let idx = usize::from(class);
            assert!(idx < 16);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.iter().all(|&value| value));
}

#[test]
fn knuth_dot_diffusion_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::<()>::Gray8,
    };

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayBits(1))
        .expect("knuth dot diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn dbs_runs_small_fixture() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };

    direct_binary_search_in_place(&mut buffer, 4).expect("direct binary search should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn dbs_objective_nonincreasing_over_iterations() {
    let target: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut data_0 = target.clone();
    let mut data_1 = target.clone();
    let mut data_2 = target.clone();

    let mut buffer_0 = Buffer {
        data: &mut data_0,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };
    let mut buffer_1 = Buffer {
        data: &mut data_1,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };
    let mut buffer_2 = Buffer {
        data: &mut data_2,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };

    direct_binary_search_in_place(&mut buffer_0, 0).expect("direct binary search should succeed");
    direct_binary_search_in_place(&mut buffer_1, 1).expect("direct binary search should succeed");
    direct_binary_search_in_place(&mut buffer_2, 2).expect("direct binary search should succeed");

    let objective_0 = dbs_objective_for_test(&target, &data_0, 8, 8);
    let objective_1 = dbs_objective_for_test(&target, &data_1, 8, 8);
    let objective_2 = dbs_objective_for_test(&target, &data_2, 8, 8);

    assert!(objective_1 <= objective_0);
    assert!(objective_2 <= objective_1);
}

#[test]
fn dbs_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..64)
        .map(|value| ((value * 1024) % 65_536) as u16)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray16,
    };

    direct_binary_search_in_place(&mut buffer, 4).expect("direct binary search should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn lattice_boltzmann_runs_small_fixture() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };

    lattice_boltzmann_in_place(&mut buffer, 6).expect("lattice-boltzmann should succeed");

    assert_eq!(data.len(), 64);
}

#[test]
fn lattice_boltzmann_binary_only_output() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };

    lattice_boltzmann_in_place(&mut buffer, 8).expect("lattice-boltzmann should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn lattice_boltzmann_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..64)
        .map(|value| ((value * 1024) % 65_536) as u16)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray16,
    };

    lattice_boltzmann_in_place(&mut buffer, 8).expect("lattice-boltzmann should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn electrostatic_halftoning_runs_small_fixture() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };

    electrostatic_halftoning_in_place(&mut buffer, 8)
        .expect("electrostatic halftoning should succeed");

    assert_eq!(data.len(), 64);
}

#[test]
fn electrostatic_halftoning_binary_only_output() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray8,
    };

    electrostatic_halftoning_in_place(&mut buffer, 10)
        .expect("electrostatic halftoning should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn electrostatic_halftoning_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..64)
        .map(|value| ((value * 1024) % 65_536) as u16)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 8,
        height: 8,
        stride: 8,
        format: PixelFormat::<()>::Gray16,
    };

    electrostatic_halftoning_in_place(&mut buffer, 10)
        .expect("electrostatic halftoning should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn riemersma_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::<()>::Gray8,
    };

    riemersma_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("riemersma should succeed");

    assert_eq!(data.len(), 256);
}

#[test]
fn riemersma_binary_only_for_graybits1() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::<()>::Gray8,
    };

    riemersma_in_place(&mut buffer, QuantizeMode::GrayBits(1)).expect("riemersma should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn riemersma_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::<()>::Gray16,
    };

    riemersma_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("riemersma should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn riemersma_f32_smoke() {
    let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
        .map(|value| (value % 256) as f32 / 255.0)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 48,
        format: PixelFormat::<()>::Rgb32F,
    };

    riemersma_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("riemersma should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

#[test]
fn dot_diffusion_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::<()>::Gray16,
    };

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("knuth dot diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn dot_diffusion_f32_smoke() {
    let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
        .map(|value| (value % 256) as f32 / 255.0)
        .collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 48,
        format: PixelFormat::<()>::Rgb32F,
    };

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("knuth dot diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

fn dbs_objective_for_test(target: &[u8], binary: &[u8], width: usize, height: usize) -> u64 {
    const KERNEL: [[u32; 3]; 3] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]];
    let mut total = 0_u64;

    for y in 0..height {
        for x in 0..width {
            let mut weighted_sum = 0_u32;
            let mut weight_total = 0_u32;
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            let x0 = x.saturating_sub(1);
            let x1 = (x + 1).min(width - 1);

            for ny in y0..=y1 {
                for nx in x0..=x1 {
                    let ky = ny + 1 - y;
                    let kx = nx + 1 - x;
                    let weight = KERNEL[ky][kx];
                    let idx = ny
                        .checked_mul(width)
                        .and_then(|base| base.checked_add(nx))
                        .expect("neighbor index overflow");
                    weighted_sum = weighted_sum
                        .checked_add(
                            u32::from(binary[idx])
                                .checked_mul(weight)
                                .expect("weighted sum overflow"),
                        )
                        .expect("weighted accumulation overflow");
                    weight_total = weight_total
                        .checked_add(weight)
                        .expect("weight accumulation overflow");
                }
            }

            let filtered = ((weighted_sum + (weight_total / 2)) / weight_total) as i32;
            let idx = y
                .checked_mul(width)
                .and_then(|base| base.checked_add(x))
                .expect("target index overflow");
            let diff = i32::from(target[idx]) - filtered;
            let sq = i64::from(diff) * i64::from(diff);
            total = total.checked_add(sq as u64).expect("objective overflow");
        }
    }

    total
}

#[test]
fn fixture_builders_are_deterministic() {
    let gray8 = gray_ramp_8x8();
    let gray16 = gray_ramp_16x16();
    let checker = checker_8x8();
    let gradient = rgb_gradient_8x8();
    let cube = rgb_cube_strip();

    assert_eq!(gray8.len(), 64);
    assert_eq!(gray16.len(), 256);
    assert_eq!(checker.len(), 64);
    assert_eq!(gradient.len(), 8 * 8 * 3);
    assert_eq!(cube.len(), 27 * 3);

    assert_eq!(checker.iter().filter(|&&value| value == 0).count(), 32);
    assert_eq!(checker.iter().filter(|&&value| value == 255).count(), 32);

    assert_eq!(fnv1a64(&gray8), fnv1a64(&gray_ramp_8x8()));
    assert_eq!(fnv1a64(&gray16), fnv1a64(&gray_ramp_16x16()));
    assert_eq!(fnv1a64(&checker), fnv1a64(&checker_8x8()));
    assert_eq!(fnv1a64(&gradient), fnv1a64(&rgb_gradient_8x8()));
    assert_eq!(fnv1a64(&cube), fnv1a64(&rgb_cube_strip()));
}
