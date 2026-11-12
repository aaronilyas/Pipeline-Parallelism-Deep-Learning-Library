//! Zero-copy slicing views plus GPU packing and concatenation.
//!
//! [`slice`] adjusts view metadata on the CPU. [`pack`] and [`concat`] are the GPU
//! entry points; this revision only freezes their signatures so the test suite can
//! compile. Both panic with `unimplemented!` until the kernels land.

use std::ops::Range;

use crate::abi::{MetalContext, Result, TensorView};

/// Maximum rank accepted by [`pack`] and [`concat`].
pub const MAX_RANK: usize = 8;

/// Maximum number of parts in one [`concat`] dispatch.
pub const MAX_CONCAT_PARTS: usize = 16;

/// Returns the sub-view of `view` selected by one range per axis.
///
/// `ranges` has one entry per axis (an empty slice for rank 0). The result borrows
/// the same buffer: `host_ptr` is the buffer base, the offset grows by `Σ start·stride`,
/// and the strides are unchanged. Empty ranges are legal; inverted or out-of-bounds
/// ranges return [`crate::abi::LinalgError::InvalidArgument`].
pub fn slice<'a, T>(
    _view: &TensorView<'a, T>,
    _ranges: &[Range<usize>],
) -> Result<TensorView<'a, T>> {
    unimplemented!("slice")
}

/// Copies `src` onto `dst` elementwise and waits until the GPU finishes.
///
/// Shapes must match. Strides may differ. Bits are copied for 2, 4, 8, and 16-byte
/// elements. See [`encode_pack`].
pub fn pack<T>(
    _ctx: &MetalContext,
    _src: &TensorView<'_, T>,
    _dst: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("pack")
}

/// Encodes [`pack`] into `cmd` without committing, waiting, or reading host memory.
pub fn encode_pack<T>(
    _ctx: &MetalContext,
    _cmd: &metal::CommandBufferRef,
    _src: &TensorView<'_, T>,
    _dst: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("encode_pack")
}

/// Concatenates `parts` along `axis` into `dst` and waits until the GPU finishes.
///
/// Parts are laid out in order. Empty parts are legal. See [`encode_concat`].
pub fn concat<T>(
    _ctx: &MetalContext,
    _parts: &[&TensorView<'_, T>],
    _axis: usize,
    _dst: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("concat")
}

/// Encodes [`concat`] into `cmd` without committing, waiting, or reading host memory.
pub fn encode_concat<T>(
    _ctx: &MetalContext,
    _cmd: &metal::CommandBufferRef,
    _parts: &[&TensorView<'_, T>],
    _axis: usize,
    _dst: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("encode_concat")
}
