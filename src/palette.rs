use crate::{core::Sample, math::color::rgb_distance_sq_unit, Error};

#[derive(Debug, Clone, PartialEq)]
pub struct Palette<S: Sample = u8> {
    colors: Vec<[S; 3]>,
}

impl<S: Sample + Eq> Eq for Palette<S> {}

pub type Palette8 = Palette<u8>;
pub type Palette16 = Palette<u16>;
pub type Palette32F = Palette<f32>;

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

#[derive(Debug, Clone, PartialEq)]
pub struct IndexedImage<S: Sample = u8> {
    pub(crate) indices: Vec<u8>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) palette: Palette<S>,
}

impl<S: Sample + Eq> Eq for IndexedImage<S> {}

pub type IndexedImage8 = IndexedImage<u8>;
pub type IndexedImage16 = IndexedImage<u16>;
pub type IndexedImage32F = IndexedImage<f32>;

impl<S: Sample> Palette<S> {
    pub const MIN_LEN: usize = 1;
    pub const MAX_LEN: usize = 256;

    pub fn new(colors: Vec<[S; 3]>) -> std::result::Result<Self, PaletteError> {
        if colors.is_empty() {
            return Err(PaletteError::Empty);
        }

        if colors.len() > Self::MAX_LEN {
            return Err(PaletteError::TooLarge);
        }

        Ok(Self { colors })
    }

    pub(crate) fn from_colors_trusted(colors: Vec<[S; 3]>) -> Self {
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
    pub fn as_slice(&self) -> &[[S; 3]] {
        &self.colors
    }

    #[must_use]
    pub fn get(&self, idx: usize) -> Option<[S; 3]> {
        self.colors.get(idx).copied()
    }

    #[must_use]
    pub fn nearest_rgb_index(&self, rgb: [S; 3]) -> usize {
        let mut best_idx = 0_usize;
        let mut best_dist = rgb_distance_sq_unit(rgb, self.colors[0]);

        for (idx, &candidate) in self.colors.iter().enumerate().skip(1) {
            let dist = rgb_distance_sq_unit(rgb, candidate);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }

        best_idx
    }

    #[must_use]
    pub fn nearest_rgb_color(&self, rgb: [S; 3]) -> [S; 3] {
        self.colors[self.nearest_rgb_index(rgb)]
    }
}

impl<S: Sample + PartialEq> Palette<S> {
    #[must_use]
    pub fn contains(&self, rgb: [S; 3]) -> bool {
        self.colors.contains(&rgb)
    }
}

impl<S: Sample> IndexedImage<S> {
    pub fn new(
        indices: Vec<u8>,
        width: usize,
        height: usize,
        palette: Palette<S>,
    ) -> crate::Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidArgument(
                "indexed image dimensions must be positive",
            ));
        }

        let expected_len = width
            .checked_mul(height)
            .ok_or(Error::InvalidArgument("indexed image dimensions overflow"))?;
        if indices.len() != expected_len {
            return Err(Error::InvalidArgument(
                "indexed image index count must match dimensions",
            ));
        }

        let palette_len = palette.len();
        if indices
            .iter()
            .any(|&index| usize::from(index) >= palette_len)
        {
            return Err(Error::InvalidArgument(
                "indexed image contains out-of-range palette indices",
            ));
        }

        Ok(Self {
            indices,
            width,
            height,
            palette,
        })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub fn indices(&self) -> &[u8] {
        &self.indices
    }

    #[must_use]
    pub fn palette(&self) -> &Palette<S> {
        &self.palette
    }

    #[must_use]
    pub fn color_at(&self, x: usize, y: usize) -> Option<[S; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = y.checked_mul(self.width)?.checked_add(x)?;
        let palette_idx = usize::from(*self.indices.get(idx)?);
        self.palette.get(palette_idx)
    }
}
