//! Tensor contraction: `C = αAB + βC`, batched GEMM, and contraction over shared axes.

use crate::abi::{MetalContext, Result, TensorView};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
}

/// Element types the contraction kernels support: `f32` and `half::f16`.
pub trait Element: sealed::Sealed + Copy + 'static {}

impl Element for f32 {}
impl Element for half::f16 {}

/// A view paired with one label per axis, einsum style (`"ijk"` labels a rank-3 view).
pub type Operand<'v, 'b, T> = (&'v TensorView<'b, T>, &'v str);

/// Encodes `C = α·A·B + β·C` for labelled operands into `cmd`.
pub fn encode_contract<T: Element>(
    _ctx: &MetalContext,
    _cmd: &metal::CommandBufferRef,
    _alpha: f32,
    _a: Operand<'_, '_, T>,
    _b: Operand<'_, '_, T>,
    _beta: f32,
    _c: Operand<'_, '_, T>,
) -> Result<()> {
    unimplemented!("contraction::encode_contract")
}

/// Synchronous [`encode_contract`]: commits and waits for completion.
pub fn contract<T: Element>(
    _ctx: &MetalContext,
    _alpha: f32,
    _a: Operand<'_, '_, T>,
    _b: Operand<'_, '_, T>,
    _beta: f32,
    _c: Operand<'_, '_, T>,
) -> Result<()> {
    unimplemented!("contraction::contract")
}

/// Encodes the batched matrix product `C = α·A·B + β·C` into `cmd`.
pub fn encode_gemm<T: Element>(
    _ctx: &MetalContext,
    _cmd: &metal::CommandBufferRef,
    _alpha: f32,
    _a: &TensorView<'_, T>,
    _b: &TensorView<'_, T>,
    _beta: f32,
    _c: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("contraction::encode_gemm")
}

/// Synchronous [`encode_gemm`]: commits and waits for completion.
pub fn gemm<T: Element>(
    _ctx: &MetalContext,
    _alpha: f32,
    _a: &TensorView<'_, T>,
    _b: &TensorView<'_, T>,
    _beta: f32,
    _c: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("contraction::gemm")
}
