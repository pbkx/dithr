use crate::Palette;

const CGA_COLORS: [[u8; 3]; 16] = [
    [0, 0, 0],
    [0, 0, 170],
    [0, 170, 0],
    [0, 170, 170],
    [170, 0, 0],
    [170, 0, 170],
    [170, 85, 0],
    [170, 170, 170],
    [85, 85, 85],
    [85, 85, 255],
    [85, 255, 85],
    [85, 255, 255],
    [255, 85, 85],
    [255, 85, 255],
    [255, 255, 85],
    [255, 255, 255],
];

#[must_use]
pub fn grayscale_2() -> Palette {
    grayscale_palette(2)
}

#[must_use]
pub fn grayscale_4() -> Palette {
    grayscale_palette(4)
}

#[must_use]
pub fn grayscale_16() -> Palette {
    grayscale_palette(16)
}

#[must_use]
pub fn cga_palette() -> Palette {
    palette_from_colors(CGA_COLORS.to_vec())
}

fn grayscale_palette(levels: usize) -> Palette {
    let last = levels - 1;
    let colors: Vec<[u8; 3]> = (0..levels)
        .map(|index| {
            let value = ((index * 255) + (last / 2)) / last;
            let value = value as u8;
            [value, value, value]
        })
        .collect();

    palette_from_colors(colors)
}

fn palette_from_colors(colors: Vec<[u8; 3]>) -> Palette {
    Palette::from_colors_trusted(colors)
}
