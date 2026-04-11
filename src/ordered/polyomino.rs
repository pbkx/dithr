use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{layout::validate_layout_invariants, PixelLayout, Sample},
    Buffer, Error, QuantizeMode, Result,
};
use std::sync::OnceLock;

const POLYOMINO_SIDE: usize = 48;
const POLYOMINO_LEN: usize = POLYOMINO_SIDE * POLYOMINO_SIDE;
const POLYOMINO_TILE_SIZE: usize = 3;
const POLYOMINO_BLOCKS: usize = POLYOMINO_SIDE / POLYOMINO_TILE_SIZE;

static POLYOMINO_48X48_FLAT: OnceLock<[u16; POLYOMINO_LEN]> = OnceLock::new();

pub fn polyomino_ordered_dither_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    buffer.validate()?;
    validate_layout_invariants::<L>()?;

    if !L::IS_GRAY {
        return Err(Error::UnsupportedFormat(
            "polyomino ordered dithering supports Gray layouts only",
        ));
    }

    ordered_dither_in_place(
        buffer,
        mode,
        polyomino_48x48_flat(),
        POLYOMINO_SIDE,
        POLYOMINO_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn polyomino_48x48_flat() -> &'static [u16; POLYOMINO_LEN] {
    POLYOMINO_48X48_FLAT.get_or_init(generate_polyomino_48x48_flat)
}

fn generate_polyomino_48x48_flat() -> [u16; POLYOMINO_LEN] {
    let mut tiles = Vec::with_capacity(POLYOMINO_LEN / POLYOMINO_TILE_SIZE);

    for block_y in 0..POLYOMINO_BLOCKS {
        for block_x in 0..POLYOMINO_BLOCKS {
            let x0 = block_x * POLYOMINO_TILE_SIZE;
            let y0 = block_y * POLYOMINO_TILE_SIZE;
            let orientation_horizontal = block_orientation(block_x, block_y);

            if orientation_horizontal {
                for row in 0..POLYOMINO_TILE_SIZE {
                    let y = y0 + row;
                    let cells = [
                        y * POLYOMINO_SIDE + x0,
                        y * POLYOMINO_SIDE + x0 + 1,
                        y * POLYOMINO_SIDE + x0 + 2,
                    ];
                    let tile_key = tile_key(x0, y, true, row);
                    tiles.push(Tile {
                        cells,
                        key: tile_key,
                    });
                }
            } else {
                for column in 0..POLYOMINO_TILE_SIZE {
                    let x = x0 + column;
                    let cells = [
                        y0 * POLYOMINO_SIDE + x,
                        (y0 + 1) * POLYOMINO_SIDE + x,
                        (y0 + 2) * POLYOMINO_SIDE + x,
                    ];
                    let tile_key = tile_key(x, y0, false, column);
                    tiles.push(Tile {
                        cells,
                        key: tile_key,
                    });
                }
            }
        }
    }

    tiles.sort_by_key(|tile| (tile.key, tile.cells[0]));

    let mut ranked = [0_u16; POLYOMINO_LEN];
    for (tile_rank, tile) in tiles.iter().enumerate() {
        let local_order = local_cell_order(tile.key);
        let base = tile_rank * POLYOMINO_TILE_SIZE;
        ranked[tile.cells[local_order[0]]] = base as u16;
        ranked[tile.cells[local_order[1]]] = (base + 1) as u16;
        ranked[tile.cells[local_order[2]]] = (base + 2) as u16;
    }

    ranked
}

struct Tile {
    cells: [usize; 3],
    key: u64,
}

fn block_orientation(block_x: usize, block_y: usize) -> bool {
    let level0 = mix_u64(hash2d(block_x, block_y, 0));
    let level1 = mix_u64(hash2d(block_x >> 1, block_y >> 1, 1));
    let level2 = mix_u64(hash2d(block_x >> 2, block_y >> 2, 2));
    (level0 ^ level1.rotate_left(17) ^ level2.rotate_left(29)) & 1 == 0
}

fn tile_key(x: usize, y: usize, horizontal: bool, lane: usize) -> u64 {
    let orientation = u64::from(horizontal as u8);
    mix_u64(
        (x as u64).wrapping_mul(0x9e37_79b1_85eb_ca87_u64)
            ^ (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f_u64)
            ^ orientation.wrapping_mul(0x1656_67b1_9e37_79f9_u64)
            ^ (lane as u64).wrapping_mul(0x94d0_49bb_1331_11eb_u64),
    )
}

fn local_cell_order(key: u64) -> [usize; 3] {
    const PERMUTATIONS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let index = (key % (PERMUTATIONS.len() as u64)) as usize;
    PERMUTATIONS[index]
}

fn hash2d(x: usize, y: usize, salt: u64) -> u64 {
    mix_u64(
        (x as u64)
            .wrapping_mul(0x632b_e59b_d9b4_e019_u64)
            .wrapping_add((y as u64).wrapping_mul(0x8cb9_2baa_33a5_049f_u64))
            .wrapping_add(salt.wrapping_mul(0x9e37_79b1_85eb_ca87_u64)),
    )
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd_u64);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53_u64);
    value ^ (value >> 33)
}

#[cfg(test)]
mod tests {
    use super::generate_polyomino_48x48_flat;

    #[test]
    fn generated_polyomino_map_has_full_rank_coverage() {
        let map = generate_polyomino_48x48_flat();
        let mut seen = vec![false; map.len()];

        for &rank in &map {
            let idx = usize::from(rank);
            assert!(idx < map.len());
            assert!(!seen[idx]);
            seen[idx] = true;
        }

        assert!(seen.into_iter().all(|entry| entry));
    }

    #[test]
    fn generated_polyomino_map_is_deterministic() {
        assert_eq!(
            generate_polyomino_48x48_flat(),
            generate_polyomino_48x48_flat()
        );
    }
}
