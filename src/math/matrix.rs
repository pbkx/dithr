use crate::{Error, Result};

pub fn tile_index(x: usize, y: usize, w: usize, h: usize) -> Result<usize> {
    if w == 0 || h == 0 {
        return Err(Error::InvalidArgument(
            "tile dimensions must be greater than zero",
        ));
    }

    let tile_x = x % w;
    let tile_y = y % h;

    let idx = tile_y
        .checked_mul(w)
        .and_then(|row_start| row_start.checked_add(tile_x))
        .ok_or(Error::InvalidArgument("tile index overflow"))?;

    Ok(idx)
}

pub fn normalize_threshold_u8(value: u8, levels: u8) -> Result<i16> {
    if levels <= 1 {
        return Err(Error::InvalidArgument(
            "threshold levels must be greater than one",
        ));
    }

    let levels_i32 = i32::from(levels);
    let scaled = (i32::from(value) * levels_i32) / 256;

    Ok((scaled - (levels_i32 / 2)) as i16)
}

#[cfg(test)]
mod tests {
    use super::{normalize_threshold_u8, tile_index};
    use crate::Error;

    #[test]
    fn tile_index_wraps_correctly() {
        assert_eq!(tile_index(0, 0, 4, 4), Ok(0));
        assert_eq!(tile_index(4, 0, 4, 4), Ok(0));
        assert_eq!(tile_index(5, 1, 4, 4), Ok(5));
        assert_eq!(tile_index(9, 10, 4, 4), Ok(9));
    }

    #[test]
    fn tile_index_rejects_zero_dimensions() {
        assert_eq!(
            tile_index(0, 0, 0, 4),
            Err(Error::InvalidArgument(
                "tile dimensions must be greater than zero"
            ))
        );
        assert_eq!(
            tile_index(0, 0, 4, 0),
            Err(Error::InvalidArgument(
                "tile dimensions must be greater than zero"
            ))
        );
    }

    #[test]
    fn normalize_threshold_u8_rejects_invalid_levels() {
        assert_eq!(
            normalize_threshold_u8(128, 1),
            Err(Error::InvalidArgument(
                "threshold levels must be greater than one"
            ))
        );
    }

    #[test]
    fn normalize_threshold_u8_maps_valid_input() {
        assert_eq!(normalize_threshold_u8(0, 4), Ok(-2));
        assert_eq!(normalize_threshold_u8(255, 4), Ok(1));
    }
}
