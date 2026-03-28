use crate::math::color::rgb_distance_sq;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    colors: Vec<[u8; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteError {
    Empty,
    TooLarge,
}

impl std::fmt::Display for PaletteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => f.write_str("palette must contain at least one color"),
            Self::TooLarge => f.write_str("palette cannot contain more than 256 colors"),
        }
    }
}

impl std::error::Error for PaletteError {}

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

    pub(crate) fn from_colors_trusted(colors: Vec<[u8; 3]>) -> Self {
        Self { colors }
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
    pub fn nearest_rgb_index(&self, rgb: [u8; 3]) -> usize {
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

    #[must_use]
    pub fn nearest_rgb_color(&self, rgb: [u8; 3]) -> [u8; 3] {
        self.colors[self.nearest_rgb_index(rgb)]
    }

    #[must_use]
    pub fn contains(&self, rgb: [u8; 3]) -> bool {
        self.colors.contains(&rgb)
    }

    #[deprecated(since = "0.1.0", note = "use nearest_rgb_index instead")]
    #[must_use]
    pub fn nearest_rgb(&self, rgb: [u8; 3]) -> usize {
        self.nearest_rgb_index(rgb)
    }
}
