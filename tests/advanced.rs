use dithr::{riemersma_in_place, Buffer, PixelFormat, QuantizeMode};

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
