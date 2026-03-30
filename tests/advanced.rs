mod common;

use common::{
    checker_8x8, fnv1a64, gray_ramp_16x16, gray_ramp_8x8, gray_ramp_8x8_u16, rgb_cube_strip,
    rgb_gradient_8x8, rgb_gradient_8x8_f32, rgb_gradient_8x8_u16,
};
use dithr::{
    direct_binary_search_in_place, electrostatic_halftoning_in_place, knuth_dot_diffusion_in_place,
    lattice_boltzmann_in_place, riemersma_in_place, QuantizeMode,
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
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::gray_bits(1))
        .expect("knuth dot diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn dbs_runs_small_fixture() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    direct_binary_search_in_place(&mut buffer, 4).expect("direct binary search should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn dbs_objective_nonincreasing_over_iterations() {
    let target: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut data_0 = target.clone();
    let mut data_1 = target.clone();
    let mut data_2 = target.clone();

    let mut buffer_0 = dithr::gray_u8(&mut data_0, 8, 8, 8).expect("valid buffer should construct");
    let mut buffer_1 = dithr::gray_u8(&mut data_1, 8, 8, 8).expect("valid buffer should construct");
    let mut buffer_2 = dithr::gray_u8(&mut data_2, 8, 8, 8).expect("valid buffer should construct");

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
    let mut buffer = dithr::gray_u16(&mut data, 8, 8, 8).expect("valid buffer should construct");

    direct_binary_search_in_place(&mut buffer, 4).expect("direct binary search should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn lattice_boltzmann_runs_small_fixture() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    lattice_boltzmann_in_place(&mut buffer, 6).expect("lattice-boltzmann should succeed");

    assert_eq!(data.len(), 64);
}

#[test]
fn lattice_boltzmann_binary_only_output() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    lattice_boltzmann_in_place(&mut buffer, 8).expect("lattice-boltzmann should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn lattice_boltzmann_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..64)
        .map(|value| ((value * 1024) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 8, 8, 8).expect("valid buffer should construct");

    lattice_boltzmann_in_place(&mut buffer, 8).expect("lattice-boltzmann should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn electrostatic_halftoning_runs_small_fixture() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    electrostatic_halftoning_in_place(&mut buffer, 8)
        .expect("electrostatic halftoning should succeed");

    assert_eq!(data.len(), 64);
}

#[test]
fn electrostatic_halftoning_binary_only_output() {
    let mut data: Vec<u8> = (0_u16..64).map(|value| (value * 4) as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 8, 8, 8).expect("valid buffer should construct");

    electrostatic_halftoning_in_place(&mut buffer, 10)
        .expect("electrostatic halftoning should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn electrostatic_halftoning_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..64)
        .map(|value| ((value * 1024) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 8, 8, 8).expect("valid buffer should construct");

    electrostatic_halftoning_in_place(&mut buffer, 10)
        .expect("electrostatic halftoning should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn riemersma_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    riemersma_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("riemersma should succeed");

    assert_eq!(data.len(), 256);
}

#[test]
fn riemersma_binary_only_for_graybits1() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = dithr::gray_u8(&mut data, 16, 16, 16).expect("valid buffer should construct");

    riemersma_in_place(&mut buffer, QuantizeMode::gray_bits(1)).expect("riemersma should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn riemersma_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 16, 16, 16).expect("valid buffer should construct");

    riemersma_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("riemersma should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn riemersma_f32_smoke() {
    let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
        .map(|value| (value % 256) as f32 / 255.0)
        .collect();
    let mut buffer = dithr::rgb_32f(&mut data, 16, 16, 48).expect("valid buffer should construct");

    riemersma_in_place(&mut buffer, QuantizeMode::GrayLevels(2)).expect("riemersma should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

#[test]
fn dot_diffusion_u16_smoke() {
    let mut data: Vec<u16> = (0_u32..256)
        .map(|value| ((value * 257) % 65_536) as u16)
        .collect();
    let mut buffer = dithr::gray_u16(&mut data, 16, 16, 16).expect("valid buffer should construct");

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("knuth dot diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0 || value == 65_535));
}

#[test]
fn dot_diffusion_f32_smoke() {
    let mut data: Vec<f32> = (0_u32..(16 * 16 * 3))
        .map(|value| (value % 256) as f32 / 255.0)
        .collect();
    let mut buffer = dithr::rgb_32f(&mut data, 16, 16, 48).expect("valid buffer should construct");

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayLevels(2))
        .expect("knuth dot diffusion should succeed");

    assert!(data.iter().all(|&value| value == 0.0 || value == 1.0));
}

fn dbs_objective_for_test(target: &[u8], binary: &[u8], width: usize, height: usize) -> f64 {
    const RADIUS: usize = 3;
    const SIZE: usize = RADIUS * 2 + 1;
    const SIGMA: f64 = 1.0;

    let mut kernel = [0.0_f32; SIZE * SIZE];
    let mut sum = 0.0_f64;
    for ky in 0..SIZE {
        for kx in 0..SIZE {
            let dx = kx as isize - RADIUS as isize;
            let dy = ky as isize - RADIUS as isize;
            let dist2 = (dx * dx + dy * dy) as f64;
            let value = (-dist2 / (2.0 * SIGMA * SIGMA)).exp();
            kernel[ky * SIZE + kx] = value as f32;
            sum += value;
        }
    }
    for value in &mut kernel {
        *value = (*value as f64 / sum) as f32;
    }

    let target_unit: Vec<f32> = target.iter().map(|&v| f32::from(v) / 255.0).collect();
    let binary_unit: Vec<f32> = binary
        .iter()
        .map(|&v| if v == 0 { 0.0_f32 } else { 1.0_f32 })
        .collect();

    let mut energy = 0.0_f64;
    for y in 0..height {
        for x in 0..width {
            let mut filtered = 0.0_f32;
            for ky in 0..SIZE {
                for kx in 0..SIZE {
                    let sx = x as isize + kx as isize - RADIUS as isize;
                    let sy = y as isize + ky as isize - RADIUS as isize;
                    if sx < 0 || sy < 0 || sx as usize >= width || sy as usize >= height {
                        continue;
                    }
                    let sidx = sy as usize * width + sx as usize;
                    filtered += kernel[ky * SIZE + kx] * (binary_unit[sidx] - target_unit[sidx]);
                }
            }
            energy += f64::from(filtered) * f64::from(filtered);
        }
    }

    energy
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
