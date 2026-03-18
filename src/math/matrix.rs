#[must_use]
pub fn tile_index(x: usize, y: usize, w: usize, h: usize) -> usize {
    if w == 0 || h == 0 {
        return 0;
    }

    let tile_x = x % w;
    let tile_y = y % h;

    tile_y.saturating_mul(w).saturating_add(tile_x)
}

#[must_use]
pub fn normalize_threshold_u8(value: u8, levels: u8) -> i16 {
    if levels <= 1 {
        return 0;
    }

    let levels_i32 = i32::from(levels);
    let scaled = (i32::from(value) * levels_i32) / 256;

    (scaled - (levels_i32 / 2)) as i16
}

#[cfg(test)]
mod tests {
    use super::tile_index;

    #[test]
    fn tile_index_wraps_correctly() {
        assert_eq!(tile_index(0, 0, 4, 4), 0);
        assert_eq!(tile_index(4, 0, 4, 4), 0);
        assert_eq!(tile_index(5, 1, 4, 4), 5);
        assert_eq!(tile_index(9, 10, 4, 4), 9);
    }
}
