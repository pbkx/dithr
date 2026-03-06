pub(crate) mod core;

use crate::{data::ErrorKernel, Buffer, QuantizeMode};

#[doc(hidden)]
pub fn error_diffuse_in_place(
    buffer: &mut Buffer<'_>,
    mode: QuantizeMode<'_>,
    kernel: &ErrorKernel,
) {
    core::error_diffuse_in_place(buffer, mode, kernel);
}
