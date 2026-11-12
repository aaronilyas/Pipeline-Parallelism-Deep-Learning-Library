//! Elementwise square root for scalars, strided vectors, and batched diagonals.
//!
//! Accepts `f32` views of rank 0 through 8. A scalar is a rank-0 view (one element at
//! `offset`). A batched diagonal is a rank-2 view of shape `[batch, n]` whose strides are
//! `[batch_stride, ld + 1]`: element `(b, i)` is the diagonal entry `i` of matrix `b` when
//! each matrix is column-major with leading dimension `ld`. Any other stride pattern with
//! the same shape is also accepted, including a plain rank-1 strided vector.
//!
//! In-place is the same view passed as both source and destination (same buffer, offset,
//! shape, and strides). Any other byte-range overlap is [`LinalgError::Overlap`]. The
//! destination must be injective so parallel stores do not alias.

use crate::abi::{MetalContext, Result, TensorView};

/// Writes the square root of `src` into `dst` and waits until the GPU has finished.
///
/// See the [module docs](self) for the view contract. The reference value of each element
/// is `(x as f64).sqrt() as f32`.
pub fn sqrt(
    ctx: &MetalContext,
    src: &TensorView<'_, f32>,
    dst: &TensorView<'_, f32>,
) -> Result<()> {
    let _ = (ctx, src, dst);
    unimplemented!("sqrt");
}

/// Encodes the square root of `src` into `dst` on `cmd` without committing or waiting.
///
/// The encoder is created and ended inside this function. Host memory is not read or
/// written. Several primitives may encode into the same command buffer.
pub fn encode_sqrt(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, f32>,
    dst: &TensorView<'_, f32>,
) -> Result<()> {
    let _ = (ctx, cmd, src, dst);
    unimplemented!("encode_sqrt");
}
