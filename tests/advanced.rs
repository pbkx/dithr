use dithr::{knuth_dot_diffusion_in_place, riemersma_in_place, Buffer, PixelFormat, QuantizeMode};

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
        format: PixelFormat::Gray8,
    };

    knuth_dot_diffusion_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}

#[test]
fn riemersma_runs() {
    let mut data: Vec<u8> = (0_u16..256).map(|value| value as u8).collect();
    let mut buffer = Buffer {
        data: &mut data,
        width: 16,
        height: 16,
        stride: 16,
        format: PixelFormat::Gray8,
    };

    riemersma_in_place(&mut buffer, QuantizeMode::GrayBits(1));

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
        format: PixelFormat::Gray8,
    };

    riemersma_in_place(&mut buffer, QuantizeMode::GrayBits(1));

    assert!(data.iter().all(|&value| value == 0 || value == 255));
}
