//! Pipeline-parallel training of a stack of `gelu(x @ W)` layers.
//!
//! RED stubs: the public API compiles and returns well-formed but wrong values.
//!
//! Every rank is told the microbatch count: a middle rank receives neither
//! inputs nor dy, so it cannot infer it.

use linalg_core::{Backend, Error, Tensor, Transport};

/// `owners[layer]` is the rank that owns that layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub owners: Vec<usize>,
}

pub fn contiguous_placement(layers: usize, stages: usize) -> Result<Placement, Error> {
    let _ = (layers, stages);
    Ok(Placement { owners: Vec::new() })
}

pub struct TrainOutput<S> {
    pub y: Vec<Tensor<S>>,
    pub dw: Vec<Tensor<S>>,
}

pub fn train<B: Backend, T: Transport>(
    backend: &B,
    transport: &mut T,
    placement: &Placement,
    microbatches: usize,
    weights: &[Tensor<B::Store>],
    inputs: &[Tensor<B::Store>],
    dy: &[Tensor<B::Store>],
) -> Result<TrainOutput<B::Store>, Error> {
    let _ = (backend, transport, placement, microbatches, weights, inputs, dy);
    Ok(TrainOutput { y: Vec::new(), dw: Vec::new() })
}
