use super::{ordered_dither_in_place, DEFAULT_STRENGTH};
use crate::{
    core::{PixelLayout, Sample},
    data::maps::generate_void_and_cluster_64x64_flat,
    Buffer, QuantizeMode, Result,
};
use std::sync::OnceLock;

const VOID_CLUSTER_SIDE: usize = 64;
const VOID_CLUSTER_LEN: usize = VOID_CLUSTER_SIDE * VOID_CLUSTER_SIDE;

static VOID_CLUSTER_64X64_FLAT: OnceLock<[u16; VOID_CLUSTER_LEN]> = OnceLock::new();

pub fn void_and_cluster_in_place<S: Sample, L: PixelLayout>(
    buffer: &mut Buffer<'_, S, L>,
    mode: QuantizeMode<'_, S>,
) -> Result<()> {
    ordered_dither_in_place(
        buffer,
        mode,
        void_cluster_64x64_flat(),
        VOID_CLUSTER_SIDE,
        VOID_CLUSTER_SIDE,
        DEFAULT_STRENGTH,
    )
}

fn void_cluster_64x64_flat() -> &'static [u16; VOID_CLUSTER_LEN] {
    VOID_CLUSTER_64X64_FLAT.get_or_init(generate_void_and_cluster_64x64_flat)
}
