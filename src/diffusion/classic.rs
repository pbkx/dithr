use super::core::error_diffuse_in_place;
use crate::{
    data::{
        ATKINSON, BURKES, FALSE_FLOYD_STEINBERG, FLOYD_STEINBERG, JARVIS_JUDICE_NINKE, SIERRA,
        SIERRA_LITE, STEVENSON_ARCE, STUCKI, TWO_ROW_SIERRA,
    },
    Buffer, QuantizeMode, Result,
};

pub fn floyd_steinberg_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &FLOYD_STEINBERG)
}

pub fn false_floyd_steinberg_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &FALSE_FLOYD_STEINBERG)
}

pub fn jarvis_judice_ninke_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &JARVIS_JUDICE_NINKE)
}

pub fn stucki_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &STUCKI)
}

pub fn burkes_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &BURKES)
}

pub fn sierra_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &SIERRA)
}

pub fn two_row_sierra_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &TWO_ROW_SIERRA)
}

pub fn sierra_lite_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &SIERRA_LITE)
}

pub fn stevenson_arce_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &STEVENSON_ARCE)
}

pub fn atkinson_in_place(buffer: &mut Buffer<'_>, mode: QuantizeMode<'_>) -> Result<()> {
    error_diffuse_in_place(buffer, mode, &ATKINSON)
}
