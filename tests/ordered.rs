use dithr::data::{
    generate_bayer_16x16, BAYER_2X2, BAYER_4X4, BAYER_8X8, CLUSTER_DOT_4X4, CLUSTER_DOT_8X8,
};

#[test]
fn bayer_2x2_contains_unique_values() {
    assert_unique_square_coverage_2(BAYER_2X2);
}

#[test]
fn bayer_4x4_contains_unique_values() {
    assert_unique_square_coverage_4(BAYER_4X4);
}

#[test]
fn bayer_8x8_contains_unique_values() {
    assert_unique_square_coverage_8(BAYER_8X8);
}

#[test]
fn generated_bayer_16x16_contains_0_to_255_once() {
    let map = generate_bayer_16x16();
    let mut seen = [false; 256];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 256);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}

#[test]
fn cluster_maps_have_expected_dimensions() {
    assert_eq!(CLUSTER_DOT_4X4.len(), 4);
    for row in CLUSTER_DOT_4X4 {
        assert_eq!(row.len(), 4);
    }

    assert_eq!(CLUSTER_DOT_8X8.len(), 8);
    for row in CLUSTER_DOT_8X8 {
        assert_eq!(row.len(), 8);
    }
}

fn assert_unique_square_coverage_2(map: [[u8; 2]; 2]) {
    let mut seen = [false; 4];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 4);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}

fn assert_unique_square_coverage_4(map: [[u8; 4]; 4]) {
    let mut seen = [false; 16];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 16);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}

fn assert_unique_square_coverage_8(map: [[u8; 8]; 8]) {
    let mut seen = [false; 64];

    for row in map {
        for value in row {
            let idx = usize::from(value);
            assert!(idx < 64);
            assert!(!seen[idx]);
            seen[idx] = true;
        }
    }

    assert!(seen.into_iter().all(|entry| entry));
}
