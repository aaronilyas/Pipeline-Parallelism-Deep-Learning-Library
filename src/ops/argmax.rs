//! Index of the element of largest magnitude in a strided 1-D view (pivot selection).
//!
//! `k = argmaxᵢ |xᵢ|` over a rank-1 [`TensorView`], written as a 0-based `u32` into a
//! device-resident output slot. Real input is `f32`; complex input is `[f32; 2]` =
//! `(re, im)` ranked by a [`Magnitude`].
//!
//! Ordering (identical in every tier of the kernel, hence bitwise deterministic):
//! a NaN magnitude outranks every number; otherwise the larger magnitude wins; equal
//! magnitudes (including `-0` and `+0`, and equal infinities) go to the lowest index. Among
//! several NaNs the lowest index wins.

use crate::abi::{LinalgError, MetalContext, Result, TensorView};

/// Threads per threadgroup of every argmax kernel.
pub const THREADGROUP_SIZE: usize = 256;

/// Largest length handled by the single-dispatch, single-threadgroup fast path.
pub const SINGLE_PASS_MAX_LEN: usize = 4096;

/// Upper bound on threadgroups launched by the first pass of the two-pass path.
pub const MAX_GROUPS: usize = 1024;

/// How the magnitude of a complex element is ranked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Magnitude {
    /// `|re| + |im|`, computed in `f32` (saturates to `+inf` on overflow).
    L1,
    /// `max · sqrt(1 + (min/max)²)` with `max = max(|re|, |im|)`, computed in `f32`.
    Hypot,
}

/// Argmax of `|x|` over a real view; commits and waits. See [`encode_argmax`].
pub fn argmax(ctx: &MetalContext, x: &TensorView<f32>, out: &TensorView<u32>) -> Result<()> {
    let _ = (ctx, x, out, LinalgError::NoDevice);
    unimplemented!("argmax")
}

/// Encodes the argmax of `|x|` over a real view into `cmd` without committing.
pub fn encode_argmax(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    x: &TensorView<f32>,
    out: &TensorView<u32>,
) -> Result<()> {
    let _ = (ctx, cmd, x, out);
    unimplemented!("encode_argmax")
}

/// Argmax of the complex magnitude of `x`; commits and waits. See [`encode_argmax_complex`].
pub fn argmax_complex(
    ctx: &MetalContext,
    x: &TensorView<[f32; 2]>,
    magnitude: Magnitude,
    out: &TensorView<u32>,
) -> Result<()> {
    let _ = (ctx, x, magnitude, out);
    unimplemented!("argmax_complex")
}

/// Encodes the argmax of the complex magnitude of `x` into `cmd` without committing.
pub fn encode_argmax_complex(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    x: &TensorView<[f32; 2]>,
    magnitude: Magnitude,
    out: &TensorView<u32>,
) -> Result<()> {
    let _ = (ctx, cmd, x, magnitude, out);
    unimplemented!("encode_argmax_complex")
}
