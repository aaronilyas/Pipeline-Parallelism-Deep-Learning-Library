//! Pipeline-parallel training of a stack of `gelu(x @ W)` layers.
//!
//! A layer is `gelu(x @ W)` with `W` row-major `[n_i, n_{i+1}]`, owned by one
//! rank. This crate evaluates no math: every product and activation goes
//! through the [`Backend`], and every boundary tensor through the
//! [`Transport`].
//!
//! Every rank is told the microbatch count: a middle rank receives neither
//! inputs nor dy, so it cannot infer it.

use std::time::Duration;

use linalg_core::{
    decode_wire, encode_wire, row_major_strides, transpose_matrix, Backend, DType, Error, Tensor,
    Transport,
};
use linalg_schedule::{local_order, Op};

const RECV_TIMEOUT: Duration = Duration::from_secs(5);

/// `owners[layer]` is the rank that owns that layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub owners: Vec<usize>,
}

/// Splits `layers` into `stages` contiguous runs. Rank `r` owns `⌊L/S⌋ + 1`
/// layers when `r < L mod S`, otherwise `⌊L/S⌋`.
pub fn contiguous_placement(layers: usize, stages: usize) -> Result<Placement, Error> {
    if stages == 0 || layers == 0 || stages > layers {
        return Err(Error::Schedule);
    }
    let (base, extra) = (layers / stages, layers % stages);
    let mut owners = Vec::with_capacity(layers);
    for rank in 0..stages {
        let count = base + usize::from(rank < extra);
        owners.extend(std::iter::repeat_n(rank, count));
    }
    Ok(Placement { owners })
}

pub struct TrainOutput<S> {
    /// Last rank: one activation per microbatch. Other ranks: empty.
    pub y: Vec<Tensor<S>>,
    /// One accumulated gradient per layer owned by this rank, in layer order.
    pub dw: Vec<Tensor<S>>,
}

/// What a forward pass keeps until its backward pass consumes it.
struct Saved<S> {
    /// The uploaded input when this rank received it (rank > 0).
    received: Option<Tensor<S>>,
    /// Pre-activation of every local layer.
    pre: Vec<Tensor<S>>,
    /// Activation of every local layer but the last, which is never read again.
    act: Vec<Tensor<S>>,
}

/// An f32 row-major contiguous matrix at offset 0.
fn check_matrix<S>(t: &Tensor<S>) -> Result<(), Error> {
    if t.dtype != DType::F32 {
        return Err(Error::Dtype);
    }
    if t.shape.len() != 2 || t.offset != 0 || t.strides != row_major_strides(&t.shape) {
        return Err(Error::Shape);
    }
    Ok(())
}

/// Downloads `t`, frames it, and sends it to `dest`.
fn send_tensor<B: Backend, T: Transport>(
    backend: &B,
    transport: &T,
    dest: usize,
    t: &Tensor<B::Store>,
) -> Result<(), Error> {
    let bytes = backend.download(t)?;
    transport.send(dest, &encode_wire(t.dtype, &t.shape, &bytes)?)
}

/// Receives a `[rows, cols]` f32 matrix from `src` and uploads it. `rows` is
/// checked when known.
fn recv_tensor<B: Backend, T: Transport>(
    backend: &B,
    transport: &T,
    src: usize,
    rows: Option<usize>,
    cols: usize,
) -> Result<Tensor<B::Store>, Error> {
    let (dtype, shape, bytes) = decode_wire(&transport.recv_timeout(src, RECV_TIMEOUT)?)?;
    if dtype != DType::F32 {
        return Err(Error::Dtype);
    }
    if shape.len() != 2 || shape[1] != cols || rows.is_some_and(|r| r != shape[0]) {
        return Err(Error::Shape);
    }
    backend.upload(dtype, &shape, &bytes)
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
    // ---- validation: nothing below touches the transport or the backend.
    let owners = &placement.owners;
    if owners.first() != Some(&0) || owners.windows(2).any(|w| w[1] != w[0] && w[1] != w[0] + 1) {
        return Err(Error::Schedule);
    }
    let stages = owners[owners.len() - 1] + 1;
    let rank = transport.rank();
    if transport.world() != stages || rank >= stages {
        return Err(Error::Schedule);
    }
    let order = local_order(stages, rank, microbatches)?;
    let count = owners.iter().filter(|&&o| o == rank).count();
    if weights.len() != count || count == 0 {
        return Err(Error::Schedule);
    }
    let is_first = rank == 0;
    let is_last = rank + 1 == stages;
    if inputs.len() != if is_first { microbatches } else { 0 }
        || dy.len() != if is_last { microbatches } else { 0 }
    {
        return Err(Error::Schedule);
    }
    for w in weights {
        check_matrix(w)?;
    }
    if weights.windows(2).any(|w| w[0].shape[1] != w[1].shape[0]) {
        return Err(Error::Shape);
    }
    let (n_in, n_out) = (weights[0].shape[0], weights[count - 1].shape[1]);
    for x in inputs {
        check_matrix(x)?;
        if x.shape[1] != n_in {
            return Err(Error::Shape);
        }
    }
    for g in dy {
        check_matrix(g)?;
        if g.shape[1] != n_out {
            return Err(Error::Shape);
        }
    }

    // ---- state
    // Alloc returns zeroed storage; dW accumulates into it with beta = 1.
    let mut dw = weights
        .iter()
        .map(|w| backend.alloc(DType::F32, &w.shape))
        .collect::<Result<Vec<_>, _>>()?;
    let mut y = Vec::with_capacity(if is_last { microbatches } else { 0 });
    let mut saved: Vec<Option<Saved<B::Store>>> = (0..microbatches).map(|_| None).collect();

    for op in order {
        match op {
            Op::Forward(m) => {
                let m = m as usize;
                let received = if is_first {
                    None
                } else {
                    Some(recv_tensor(backend, transport, rank - 1, None, n_in)?)
                };
                let batch = received.as_ref().unwrap_or_else(|| &inputs[m]).shape[0];
                if is_last && dy[m].shape[0] != batch {
                    return Err(Error::Shape);
                }
                let mut pre_all = Vec::with_capacity(count);
                let mut act_all: Vec<Tensor<B::Store>> = Vec::with_capacity(count);
                for (l, w) in weights.iter().enumerate() {
                    let x = match act_all.last() {
                        Some(a) => a,
                        None => received.as_ref().unwrap_or_else(|| &inputs[m]),
                    };
                    let n = w.shape[1];
                    let mut pre = backend.alloc(DType::F32, &[batch, n])?;
                    backend.gemm(x, w, &mut pre, 1.0, 0.0)?;
                    let mut act = backend.alloc(DType::F32, &[batch, n])?;
                    backend.gelu(&pre, &mut act)?;
                    pre_all.push(pre);
                    if l + 1 < count {
                        act_all.push(act);
                    } else if is_last {
                        y.push(act);
                    } else {
                        send_tensor(backend, transport, rank + 1, &act)?;
                    }
                }
                saved[m] = Some(Saved { received, pre: pre_all, act: act_all });
            }
            Op::Backward(m) => {
                let m = m as usize;
                let s = saved[m].take().ok_or(Error::Schedule)?;
                let batch = s.pre[0].shape[0];
                let mut owned = if is_last {
                    None
                } else {
                    Some(recv_tensor(backend, transport, rank + 1, Some(batch), n_out)?)
                };
                for l in (0..count).rev() {
                    let g = match owned.as_ref() {
                        Some(g) => g,
                        None => &dy[m],
                    };
                    let (k, n) = (weights[l].shape[0], weights[l].shape[1]);
                    let mut dpre = backend.alloc(DType::F32, &[batch, n])?;
                    backend.gelu_backward(&s.pre[l], g, &mut dpre)?;
                    let x = if l == 0 {
                        s.received.as_ref().unwrap_or_else(|| &inputs[m])
                    } else {
                        &s.act[l - 1]
                    };
                    backend.gemm(&transpose_matrix(x)?, &dpre, &mut dw[l], 1.0, 1.0)?;
                    let mut dx = backend.alloc(DType::F32, &[batch, k])?;
                    backend.gemm(&dpre, &transpose_matrix(&weights[l])?, &mut dx, 1.0, 0.0)?;
                    if l == 0 && !is_first {
                        send_tensor(backend, transport, rank - 1, &dx)?;
                    }
                    owned = Some(dx);
                }
            }
        }
    }
    Ok(TrainOutput { y, dw })
}
