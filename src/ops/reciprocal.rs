//! Elementwise reciprocal over ℝ and ℂ for scalars, strided vectors, and batched diagonals.
//!
//! [`reciprocal`] maps every element `x` of a strided source view to `x⁻¹` in a destination
//! view of the same shape, either in place (source and destination are the same view) or out
//! of place (the views do not overlap). [`encode_reciprocal`] records the same work into a
//! caller-owned command buffer without committing it.

use crate::abi::{MetalContext, Result, TensorView};

/// Highest view rank the kernels address directly (one dispatch-grid axis per view axis).
pub const MAX_RANK: usize = 3;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for [f32; 2] {}
}

/// Element types the reciprocal kernels support: real `f32` and complex `[f32; 2]` = (re, im).
pub trait ReciprocalElement: sealed::Sealed + Copy + 'static {
    /// `[[host_name]]` of the kernel variant for this element type.
    const KERNEL: &'static str;
}

impl ReciprocalElement for f32 {
    const KERNEL: &'static str = "reciprocal_f32";
}

impl ReciprocalElement for [f32; 2] {
    const KERNEL: &'static str = "reciprocal_c32";
}

/// `MTLMathModeSafe`: no optimization may violate IEEE 754.
const MATH_MODE_SAFE: isize = 0;
/// `MTLMathFloatingPointFunctionsPrecise`: single-precision functions default to `precise::`.
const MATH_FUNCTIONS_PRECISE: isize = 1;

/// Compile options of the reciprocal shader: MSL 3.1, fast math off, and (where the OS
/// offers the selectors, macOS 15+) safe math mode with precise floating-point functions,
/// so that `/` is an IEEE division.
pub fn compile_options() -> metal::CompileOptions {
    use metal::foreign_types::ForeignType;
    use metal::objc::runtime::{Object, Sel};
    use metal::objc::Message;

    let options = metal::CompileOptions::new();
    options.set_language_version(metal::MTLLanguageVersion::V3_1);
    options.set_fast_math_enabled(false);
    let setters = [
        ("setMathMode:", MATH_MODE_SAFE),
        ("setMathFloatingPointFunctions:", MATH_FUNCTIONS_PRECISE),
    ];
    for (name, value) in setters {
        let selector = Sel::register(name);
        // SAFETY: `options` is a live `MTLCompileOptions`; each selector is sent only when
        // the object responds to it and takes one `NSInteger`-backed enum, returning void.
        unsafe {
            let object = &*(options.as_ptr() as *const Object);
            if object.class().instances_respond_to(selector) {
                let _ = options.send_message::<(isize,), ()>(selector, (value,));
            }
        }
    }
    options
}

/// Encodes `dst[i] = 1 / src[i]` for every multi-index `i` into `cmd`.
///
/// Creates and ends its own compute encoder; never commits, waits, or touches host memory.
pub fn encode_reciprocal<T: ReciprocalElement>(
    _ctx: &MetalContext,
    _cmd: &metal::CommandBufferRef,
    _src: &TensorView<'_, T>,
    _dst: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("reciprocal: RED stub")
}

/// Computes `dst[i] = 1 / src[i]` for every multi-index `i` and waits for completion.
pub fn reciprocal<T: ReciprocalElement>(
    _ctx: &MetalContext,
    _src: &TensorView<'_, T>,
    _dst: &TensorView<'_, T>,
) -> Result<()> {
    unimplemented!("reciprocal: RED stub")
}
