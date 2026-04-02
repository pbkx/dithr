use crate::{
    core::{alpha_index, read_unit_pixel, PixelLayout, Sample},
    math::color::luma_unit,
    Error, Palette, Result,
};

pub type QuantizeError = Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantizeMode<'a, S: Sample = u8> {
    GrayLevels(u16),
    RgbLevels(u16),
    Palette(&'a Palette<S>),
    SingleColor { fg: [S; 3], levels: u16 },
}

impl<'a, S: Sample> QuantizeMode<'a, S> {
    pub fn gray_levels(levels: u16) -> Result<Self> {
        validate_levels(levels)?;
        Ok(Self::GrayLevels(levels))
    }

    pub fn rgb_levels(levels: u16) -> Result<Self> {
        validate_levels(levels)?;
        Ok(Self::RgbLevels(levels))
    }

    #[must_use]
    pub const fn palette(palette: &'a Palette<S>) -> Self {
        Self::Palette(palette)
    }

    pub fn single_color(fg: [S; 3], levels: u16) -> Result<Self> {
        validate_levels(levels)?;
        Ok(Self::SingleColor { fg, levels })
    }
}

impl<'a> QuantizeMode<'a, u8> {
    pub fn gray_bits(bits: u8) -> Result<Self> {
        Ok(Self::GrayLevels(levels_from_bits(bits)?))
    }

    pub fn rgb_bits(bits: u8) -> Result<Self> {
        Ok(Self::RgbLevels(levels_from_bits(bits)?))
    }
}

pub fn levels_from_bits(bits: u8) -> std::result::Result<u16, QuantizeError> {
    if !(1..=8).contains(&bits) {
        return Err(Error::InvalidArgument("quantization bits must be in 1..=8"));
    }

    Ok(1_u16 << bits)
}

pub fn quantize_gray<S: Sample>(value: S, levels: u16) -> Result<S> {
    validate_levels(levels)?;

    let steps = f32::from(levels - 1);
    let unit = value.to_unit_f32().clamp(0.0, 1.0);
    let index = (unit * steps).round();
    let quantized = (index / steps).clamp(0.0, 1.0);

    Ok(S::from_unit_f32(quantized))
}

pub fn quantize_rgb<S: Sample>(rgb: [S; 3], levels: u16) -> Result<[S; 3]> {
    Ok([
        quantize_gray(rgb[0], levels)?,
        quantize_gray(rgb[1], levels)?,
        quantize_gray(rgb[2], levels)?,
    ])
}

pub fn quantize_pixel<S: Sample, L: PixelLayout>(
    pixel: &[S],
    mode: QuantizeMode<'_, S>,
) -> Result<[S; 4]> {
    if pixel.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }

    let rgba = read_unit_pixel::<S, L>(pixel)?;
    let rgb = [rgba[0], rgba[1], rgba[2]];

    let out = match mode {
        QuantizeMode::GrayLevels(levels) => {
            let g = quantize_gray(
                S::from_unit_f32(luma_unit([
                    S::from_unit_f32(rgb[0]),
                    S::from_unit_f32(rgb[1]),
                    S::from_unit_f32(rgb[2]),
                ])),
                levels,
            )?
            .to_unit_f32();
            [
                S::from_unit_f32(g),
                S::from_unit_f32(g),
                S::from_unit_f32(g),
                S::from_unit_f32(rgba[3]),
            ]
        }
        QuantizeMode::RgbLevels(levels) => {
            let q = quantize_rgb(
                [
                    S::from_unit_f32(rgb[0]),
                    S::from_unit_f32(rgb[1]),
                    S::from_unit_f32(rgb[2]),
                ],
                levels,
            )?;
            [q[0], q[1], q[2], S::from_unit_f32(rgba[3])]
        }
        QuantizeMode::Palette(palette) => {
            let nearest = palette.nearest_rgb_index([
                S::from_unit_f32(rgb[0]),
                S::from_unit_f32(rgb[1]),
                S::from_unit_f32(rgb[2]),
            ]);
            let q = palette.as_slice()[nearest];
            [q[0], q[1], q[2], S::from_unit_f32(rgba[3])]
        }
        QuantizeMode::SingleColor { fg, levels } => {
            let g = quantize_gray(
                S::from_unit_f32(luma_unit([
                    S::from_unit_f32(rgb[0]),
                    S::from_unit_f32(rgb[1]),
                    S::from_unit_f32(rgb[2]),
                ])),
                levels,
            )?
            .to_unit_f32();
            let fg_r = fg[0].to_unit_f32();
            let fg_g = fg[1].to_unit_f32();
            let fg_b = fg[2].to_unit_f32();
            [
                S::from_unit_f32(fg_r * g),
                S::from_unit_f32(fg_g * g),
                S::from_unit_f32(fg_b * g),
                S::from_unit_f32(rgba[3]),
            ]
        }
    };

    Ok(out)
}

pub fn quantize_error<S: Sample, L: PixelLayout>(
    original: &[S],
    quantized: &[S],
) -> Result<[f32; 4]> {
    if L::CHANNELS > 4 {
        return Err(Error::UnsupportedFormat(
            "quantize error supports layouts with up to 4 channels",
        ));
    }
    if original.len() != quantized.len() {
        return Err(Error::InvalidArgument(
            "original and quantized pixel lengths must match",
        ));
    }
    if original.len() != L::CHANNELS {
        return Err(Error::InvalidArgument(
            "pixel slice length does not match layout",
        ));
    }

    let mut out = [0.0_f32; 4];
    for (idx, (a, b)) in original.iter().zip(quantized.iter()).enumerate() {
        out[idx] = a.to_unit_f32() - b.to_unit_f32();
    }
    if alpha_index::<L>().is_none() {
        out[3] = 0.0;
    }

    Ok(out)
}

#[inline]
pub fn quantize_gray_u8(value: u8, bits: u8) -> Result<u8> {
    let levels = levels_from_bits(bits)?;
    quantize_gray(value, levels)
}

#[inline]
pub fn quantize_rgb_u8(rgb: [u8; 3], bits: u8) -> Result<[u8; 3]> {
    let levels = levels_from_bits(bits)?;
    quantize_rgb(rgb, levels)
}

fn validate_levels(levels: u16) -> Result<()> {
    if levels >= 2 {
        Ok(())
    } else {
        Err(Error::InvalidArgument(
            "quantization levels must be in 2..=65535",
        ))
    }
}
