#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    colors: Vec<[u8; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteError {
    Empty,
    TooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexedImage {
    pub indices: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub palette: Palette,
}

impl Palette {
    pub const MIN_LEN: usize = 1;
    pub const MAX_LEN: usize = 256;

    pub fn new(colors: Vec<[u8; 3]>) -> Result<Self, PaletteError> {
        if colors.is_empty() {
            return Err(PaletteError::Empty);
        }

        if colors.len() > Self::MAX_LEN {
            return Err(PaletteError::TooLarge);
        }

        Ok(Self { colors })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }

    #[must_use]
    pub fn as_slice(&self) -> &[[u8; 3]] {
        &self.colors
    }

    #[must_use]
    pub fn get(&self, idx: usize) -> Option<[u8; 3]> {
        self.colors.get(idx).copied()
    }

    #[must_use]
    pub fn nearest_rgb(&self, rgb: [u8; 3]) -> usize {
        let mut best_idx = 0_usize;
        let mut best_dist = rgb_distance_sq(rgb, self.colors[0]);

        for (idx, &candidate) in self.colors.iter().enumerate().skip(1) {
            let dist = rgb_distance_sq(rgb, candidate);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        best_idx
    }
}

#[must_use]
fn rgb_distance_sq(a: [u8; 3], b: [u8; 3]) -> u32 {
    let dr = i32::from(a[0]) - i32::from(b[0]);
    let dg = i32::from(a[1]) - i32::from(b[1]);
    let db = i32::from(a[2]) - i32::from(b[2]);

    (dr * dr + dg * dg + db * db) as u32
}
