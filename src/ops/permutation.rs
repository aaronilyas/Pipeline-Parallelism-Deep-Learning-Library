//! Axis permutation views, contiguous materialization, and complex adjoint.

use crate::abi::{MetalContext, Result, TensorView};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for i32 {}
    impl Sealed for [f32; 2] {}
}

/// Element types the permutation kernels move as raw bit patterns.
///
/// Four-byte types move as one 32-bit word and `[f32; 2]` (complex f32, `(re, im)`) as two.
/// The set is closed: it is sealed to the types the kernels are instantiated for.
pub trait Element: Copy + sealed::Sealed {
    /// True for complex types, whose adjoint and conjugate flip the imaginary sign bit.
    const COMPLEX: bool;
}

impl Element for f32 {
    const COMPLEX: bool = false;
}
impl Element for u32 {
    const COMPLEX: bool = false;
}
impl Element for i32 {
    const COMPLEX: bool = false;
}
impl Element for [f32; 2] {
    const COMPLEX: bool = true;
}

/// Reorders the axes of `view` without touching memory (O(rank), no GPU work).
///
/// Axis `i` of the result is axis `perm[i]` of `view`. The buffer, `host_ptr` and offset
/// are unchanged. Fails if `perm` is not a permutation of `0..view.rank()`.
pub fn permute_view<'a, T>(view: &TensorView<'a, T>, perm: &[usize]) -> Result<TensorView<'a, T>> {
    let _ = (view, perm);
    unimplemented!()
}

/// Copies `src` (any strided view) into the row-major view `dst` of the same shape and
/// waits for completion.
///
/// Fails on invalid metadata or if `dst` may overlap `src`.
pub fn materialize<T: Element>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let _ = (ctx, src, dst);
    unimplemented!()
}

/// Encodes [`materialize`] into `cmd` without committing or waiting.
pub fn encode_materialize<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let _ = (ctx, cmd, src, dst);
    unimplemented!()
}

/// Writes the conjugate of `src` into the row-major view `dst` and waits for completion.
///
/// Conjugation XORs the imaginary sign bit, so it is bitwise self-inverse; for real
/// element types it is a plain copy.
pub fn conjugate<T: Element>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let _ = (ctx, src, dst);
    unimplemented!()
}

/// Encodes [`conjugate`] into `cmd` without committing or waiting.
pub fn encode_conjugate<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let _ = (ctx, cmd, src, dst);
    unimplemented!()
}

/// Writes the batched adjoint of `src` (shape `[.., m, n]`) into the row-major view `dst`
/// (shape `[.., n, m]`) and waits for completion.
///
/// The adjoint transposes the last two axes and conjugates complex elements; for real
/// types it is the transpose. Fails if `dst` may overlap `src`.
pub fn adjoint<T: Element>(
    ctx: &MetalContext,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let _ = (ctx, src, dst);
    unimplemented!()
}

/// Encodes [`adjoint`] into `cmd` without committing or waiting.
pub fn encode_adjoint<T: Element>(
    ctx: &MetalContext,
    cmd: &metal::CommandBufferRef,
    src: &TensorView<'_, T>,
    dst: &TensorView<'_, T>,
) -> Result<()> {
    let _ = (ctx, cmd, src, dst);
    unimplemented!()
}
