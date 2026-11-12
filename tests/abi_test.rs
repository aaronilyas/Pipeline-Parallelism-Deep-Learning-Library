//! Scaffold smoke tests: device access, shared allocation, view metadata, and the
//! pipeline cache running a trivial kernel end to end.

use linalg::abi::{
    alloc_shared, compute_pipeline, row_major_strides, LinalgError, MetalContext, TensorView,
    ALLOC_GRANULE,
};
use metal::foreign_types::ForeignType;

const ADD_ONE: &str = r#"
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] += 1.0f; }
"#;

#[test]
fn alloc_rounds_up_and_zeroes() {
    let ctx = MetalContext::new().unwrap();
    for (len, expected) in [(0usize, ALLOC_GRANULE), (1, ALLOC_GRANULE), (4097, 2 * ALLOC_GRANULE)]
    {
        let buf = alloc_shared::<f32>(&ctx, len);
        assert_eq!(buf.length() as usize, expected);
        let bytes = unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, expected) };
        assert!(bytes.iter().all(|&b| b == 0));
    }
}

#[test]
fn view_validation() {
    let ctx = MetalContext::new().unwrap();
    let buf = alloc_shared::<f32>(&ctx, 4096);
    let v = TensorView::<f32>::row_major(&buf, 0, vec![64, 64]);
    assert_eq!(v.strides, row_major_strides(&[64, 64]));
    assert_eq!(v.host_ptr, buf.contents() as *mut f32);
    assert!(v.validate().is_ok());
    assert_eq!(v.element_range(), Some(0..4096));
    let past_end = TensorView::<f32>::row_major(&buf, 1, vec![64, 64]);
    assert!(matches!(past_end.validate(), Err(LinalgError::InvalidArgument(_))));
    let overflow = TensorView::<f32>::new(&buf, 0, vec![2, 2], vec![usize::MAX, 1]);
    assert!(overflow.validate().is_err());
    let empty = TensorView::<f32>::new(&buf, 4096, vec![0, 7], vec![7, 1]);
    assert!(empty.validate().is_ok() && empty.element_range().is_none());
    let lo = TensorView::<f32>::row_major(&buf, 0, vec![10]);
    let hi = TensorView::<f32>::row_major(&buf, 10, vec![10]);
    assert!(
        !lo.may_overlap(&hi) && lo.may_overlap(&TensorView::<f32>::row_major(&buf, 9, vec![2]))
    );
}

#[test]
fn pipeline_cache_runs_kernel() {
    let ctx = MetalContext::new().unwrap();
    let buf = alloc_shared::<f32>(&ctx, 256);
    let pso =
        compute_pipeline(&ctx, "abi_test", ADD_ONE, metal::CompileOptions::new, "add_one").unwrap();
    let again =
        compute_pipeline(&ctx, "abi_test", ADD_ONE, metal::CompileOptions::new, "add_one").unwrap();
    assert_eq!(pso.as_ptr(), again.as_ptr());
    let cmd = ctx.queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&buf), 0);
    enc.dispatch_threads(metal::MTLSize::new(256, 1, 1), metal::MTLSize::new(64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let x = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, 256) };
    assert!(x.iter().all(|&v| v == 1.0));
    assert!(matches!(
        compute_pipeline(&ctx, "abi_test", ADD_ONE, metal::CompileOptions::new, "missing"),
        Err(LinalgError::ShaderCompile(_))
    ));
}
