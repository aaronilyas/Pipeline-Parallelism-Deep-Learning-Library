//! Pipeline invariants, written by the lead before the bodies.
//!
//! Every rank runs on its own thread, builds its own backend and tensors, and
//! returns only plain bytes. No `Store` crosses a thread.

use std::cell::{Cell, RefCell};
use std::sync::Mutex;
use std::time::Duration;

use linalg_core::{
    row_major_strides, Backend, DType, DeviceKind, Error, Identified, Tensor, Transport,
};
use linalg_cpu::CpuBackend;
use linalg_pipeline::{contiguous_placement, train, Placement};

// ---------------------------------------------------------------- fixture

const BATCH: usize = 2;
const WIDTHS: [usize; 4] = [3, 4, 2, 3];
const LAYERS: usize = 3;

const X0: [f32; 6] = [0.5, -1.0, 0.25, 1.0, 0.0, -0.5];
const X1: [f32; 6] = [-0.25, 0.5, 1.0, 0.0, -1.0, 0.25];
const DY0: [f32; 6] = [0.1, -0.2, 0.3, 0.0, 0.25, -0.1];
const DY1: [f32; 6] = [-0.3, 0.1, 0.0, 0.2, -0.2, 0.4];

fn weight(layer: usize) -> Vec<f32> {
    let (rows, cols) = (WIDTHS[layer], WIDTHS[layer + 1]);
    let mut w = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            w.push(0.1 * (r as f32 + 1.0) - 0.05 * (c as f32 + 1.0) + 0.02 * layer as f32);
        }
    }
    w
}

/// Microbatch inputs and output gradients. `order` picks X0/X1 per microbatch.
fn microbatches(order: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let xs = order.iter().map(|&i| if i == 0 { X0.to_vec() } else { X1.to_vec() }).collect();
    let dys = order.iter().map(|&i| if i == 0 { DY0.to_vec() } else { DY1.to_vec() }).collect();
    (xs, dys)
}

fn bytes_of(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32s_of(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn wire_len(width: usize) -> usize {
    16 + 4 * 2 + BATCH * width * 4
}

// ---------------------------------------------------------------- Counting

/// Delegates every method to `inner` and logs each success.
struct Counting<B: Backend> {
    inner: B,
    log: Mutex<Vec<&'static str>>,
}

impl<B: Backend> Counting<B> {
    fn new(inner: B) -> Self {
        Self { inner, log: Mutex::new(Vec::new()) }
    }

    fn note<T>(&self, name: &'static str, result: Result<T, Error>) -> Result<T, Error> {
        if result.is_ok() {
            self.log.lock().unwrap().push(name);
        }
        result
    }

    fn entries(&self) -> Vec<&'static str> {
        self.log.lock().unwrap().clone()
    }

    fn count(&self, name: &str) -> usize {
        self.entries().iter().filter(|e| **e == name).count()
    }
}

impl<B: Backend> Backend for Counting<B> {
    type Store = B::Store;

    fn kind(&self) -> DeviceKind {
        self.inner.kind()
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Self::Store>, Error> {
        self.note("alloc", self.inner.alloc(dtype, shape))
    }

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        row_major: &[u8],
    ) -> Result<Tensor<Self::Store>, Error> {
        self.note("upload", self.inner.upload(dtype, shape, row_major))
    }

    fn download(&self, tensor: &Tensor<Self::Store>) -> Result<Vec<u8>, Error> {
        self.note("download", self.inner.download(tensor))
    }

    fn gemm(
        &self,
        a: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        c: &mut Tensor<Self::Store>,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error> {
        self.note("gemm", self.inner.gemm(a, b, c, alpha, beta))
    }

    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        self.note("gelu", self.inner.gelu(x, y))
    }

    fn gelu_backward(
        &self,
        pre: &Tensor<Self::Store>,
        dy: &Tensor<Self::Store>,
        dx: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        self.note("gelu-backward", self.inner.gelu_backward(pre, dy, dx))
    }

    fn download_into(&self, tensor: &Tensor<Self::Store>, out: &mut [u8]) -> Result<(), Error> {
        self.note("download_into", self.inner.download_into(tensor, out))
    }

    fn upload_into(&self, bytes: &[u8], tensor: &mut Tensor<Self::Store>) -> Result<(), Error> {
        self.note("upload_into", self.inner.upload_into(bytes, tensor))
    }

    fn sync(&self) -> Result<(), Error> {
        self.note("sync", self.inner.sync())
    }

    fn add_bias(
        &self,
        x: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        y: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        self.note("add_bias", self.inner.add_bias(x, b, y))
    }

    fn bias_grad(
        &self,
        dpre: &Tensor<Self::Store>,
        db: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        self.note("bias_grad", self.inner.bias_grad(dpre, db))
    }

    fn mse_grad(
        &self,
        y: &Tensor<Self::Store>,
        target: &Tensor<Self::Store>,
        dy: &mut Tensor<Self::Store>,
        loss: &mut f32,
    ) -> Result<(), Error> {
        self.note("mse_grad", self.inner.mse_grad(y, target, dy, loss))
    }

    fn sgd_step(
        &self,
        w: &mut Tensor<Self::Store>,
        dw: &mut Tensor<Self::Store>,
        lr: f32,
    ) -> Result<(), Error> {
        self.note("sgd_step", self.inner.sgd_step(w, dw, lr))
    }

    fn convert(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        self.note("convert", self.inner.convert(x, y))
    }

    fn debug_allocs(&self) -> u64 {
        self.inner.debug_allocs()
    }

    fn debug_launches(&self) -> Vec<&'static str> {
        self.inner.debug_launches()
    }

    fn debug_downloads(&self) -> u64 {
        self.inner.debug_downloads()
    }

    fn debug_uploads(&self) -> u64 {
        self.inner.debug_uploads()
    }
}

// ---------------------------------------------------------------- transports

/// Records the length of every successful send and counts successful recvs.
struct Logged<T: Transport> {
    inner: T,
    sent: RefCell<Vec<usize>>,
    received: Cell<usize>,
}

impl<T: Transport> Logged<T> {
    fn new(inner: T) -> Self {
        Self { inner, sent: RefCell::new(Vec::new()), received: Cell::new(0) }
    }
}

impl<T: Transport> Transport for Logged<T> {
    fn rank(&self) -> usize {
        self.inner.rank()
    }

    fn world(&self) -> usize {
        self.inner.world()
    }

    fn send(&self, dest: usize, bytes: &[u8]) -> Result<(), Error> {
        self.inner.send(dest, bytes)?;
        self.sent.borrow_mut().push(bytes.len());
        Ok(())
    }

    fn recv_timeout(&self, src: usize, timeout: Duration) -> Result<Vec<u8>, Error> {
        let bytes = self.inner.recv_timeout(src, timeout)?;
        self.received.set(self.received.get() + 1);
        Ok(bytes)
    }

    fn send_parts(&self, dest: usize, parts: &[&[u8]]) -> Result<(), Error> {
        self.inner.send_parts(dest, parts)?;
        self.sent.borrow_mut().push(parts.iter().map(|part| part.len()).sum());
        Ok(())
    }

    fn recv_into(&self, src: usize, buf: &mut Vec<u8>, timeout: Duration) -> Result<(), Error> {
        self.inner.recv_into(src, buf, timeout)?;
        self.received.set(self.received.get() + 1);
        Ok(())
    }
}

/// World of one: every send or recv is a contract violation.
struct Solo;

impl Transport for Solo {
    fn rank(&self) -> usize {
        0
    }

    fn world(&self) -> usize {
        1
    }

    fn send(&self, _dest: usize, _bytes: &[u8]) -> Result<(), Error> {
        Err(Error::Transport)
    }

    fn recv_timeout(&self, _src: usize, _timeout: Duration) -> Result<Vec<u8>, Error> {
        Err(Error::Transport)
    }

    fn send_parts(&self, _dest: usize, _parts: &[&[u8]]) -> Result<(), Error> {
        Err(Error::Transport)
    }

    fn recv_into(&self, _src: usize, _buf: &mut Vec<u8>, _timeout: Duration) -> Result<(), Error> {
        Err(Error::Transport)
    }
}

// ---------------------------------------------------------------- rank driver

/// What one rank hands back to the test thread: bytes and logs only.
struct RankResult {
    y: Vec<Vec<u8>>,
    dw: Vec<Vec<u8>>,
    log: Vec<&'static str>,
    sent: Vec<usize>,
    received: usize,
    owned_layers: usize,
    weight_ids_before: Vec<u64>,
    weight_ids_after: Vec<u64>,
    weight_bytes_after: Vec<Vec<u8>>,
    dw_ids: Vec<u64>,
}

fn owned(placement: &Placement, rank: usize) -> Vec<usize> {
    (0..placement.owners.len()).filter(|&l| placement.owners[l] == rank).collect()
}

fn run_rank<T: Transport>(transport: T, stages: usize, order: &[usize]) -> RankResult {
    let rank = transport.rank();
    let mut transport = Logged::new(transport);
    let backend = Counting::new(CpuBackend::new());
    let cpu = &backend.inner;
    let placement = contiguous_placement(LAYERS, stages).expect("placement");
    let layers = owned(&placement, rank);
    let weights: Vec<_> = layers
        .iter()
        .map(|&l| {
            cpu.upload(DType::F32, &[WIDTHS[l], WIDTHS[l + 1]], &bytes_of(&weight(l))).unwrap()
        })
        .collect();
    let (xs, dys) = microbatches(order);
    let inputs: Vec<_> = if rank == 0 {
        xs.iter()
            .map(|x| cpu.upload(DType::F32, &[BATCH, WIDTHS[0]], &bytes_of(x)).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    let dy: Vec<_> = if rank + 1 == stages {
        dys.iter()
            .map(|d| cpu.upload(DType::F32, &[BATCH, WIDTHS[LAYERS]], &bytes_of(d)).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    let weight_ids_before = weights.iter().map(|w| w.storage.storage_id()).collect();

    let out = train(&backend, &mut transport, &placement, order.len(), &weights, &inputs, &dy)
        .expect("train succeeds");

    RankResult {
        y: out.y.iter().map(|t| cpu.download(t).unwrap()).collect(),
        dw: out.dw.iter().map(|t| cpu.download(t).unwrap()).collect(),
        log: backend.entries(),
        sent: transport.sent.take(),
        received: transport.received.get(),
        owned_layers: layers.len(),
        weight_ids_before,
        weight_ids_after: weights.iter().map(|w| w.storage.storage_id()).collect(),
        weight_bytes_after: weights.iter().map(|w| cpu.download(w).unwrap()).collect(),
        dw_ids: out.dw.iter().map(|t| t.storage.storage_id()).collect(),
    }
}

fn run_loopback(stages: usize, order: &[usize]) -> Vec<RankResult> {
    if stages == 1 {
        return vec![run_rank(Solo, 1, order)];
    }
    let endpoints = linalg_transport::loopback(stages).expect("loopback");
    std::thread::scope(|scope| {
        let handles: Vec<_> = endpoints
            .into_iter()
            .map(|ep| scope.spawn(move || run_rank(ep, stages, order)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn run_tcp(stages: usize, order: &[usize]) -> Vec<RankResult> {
    let endpoints = linalg_transport::localhost_mesh(stages).expect("mesh");
    std::thread::scope(|scope| {
        let handles: Vec<_> = endpoints
            .into_iter()
            .map(|ep| scope.spawn(move || run_rank(ep, stages, order)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// y from the last rank and every dW in layer order.
fn gather(ranks: &[RankResult]) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let y = ranks.last().unwrap().y.clone();
    let dw = ranks.iter().flat_map(|r| r.dw.clone()).collect();
    (y, dw)
}

fn assert_full_shape(y: &[Vec<u8>], dw: &[Vec<u8>], microbatches: usize) {
    assert_eq!(y.len(), microbatches, "one y per microbatch");
    for bytes in y {
        assert_eq!(bytes.len(), BATCH * WIDTHS[LAYERS] * 4, "y is [batch, n_L] f32");
    }
    assert_eq!(dw.len(), LAYERS, "one dW per layer");
    for (l, bytes) in dw.iter().enumerate() {
        assert_eq!(bytes.len(), WIDTHS[l] * WIDTHS[l + 1] * 4, "dW[{l}] shape");
    }
}

// ---------------------------------------------------------------- reference

/// The CPU gemm contract, as a test-local loop: start from `beta·C` (or 0 when
/// beta is 0), then add `alpha·a·b` over k in increasing order, in f32.
#[allow(clippy::too_many_arguments)]
fn ref_gemm(
    m: usize,
    k: usize,
    n: usize,
    a: impl Fn(usize, usize) -> f32,
    b: impl Fn(usize, usize) -> f32,
    c: &mut [f32],
    alpha: f32,
    beta: f32,
) {
    for row in 0..m {
        for col in 0..n {
            let mut acc = if beta == 0.0 { 0.0 } else { beta * c[row * n + col] };
            if alpha != 0.0 {
                for inner in 0..k {
                    acc += alpha * a(row, inner) * b(inner, col);
                }
            }
            c[row * n + col] = acc;
        }
    }
}

fn cpu_gelu(cpu: &CpuBackend, x: &[f32], shape: &[usize]) -> Vec<f32> {
    let xt = cpu.upload(DType::F32, shape, &bytes_of(x)).unwrap();
    let mut yt = cpu.alloc(DType::F32, shape).unwrap();
    cpu.gelu(&xt, &mut yt).unwrap();
    f32s_of(&cpu.download(&yt).unwrap())
}

fn cpu_gelu_backward(cpu: &CpuBackend, pre: &[f32], dy: &[f32], shape: &[usize]) -> Vec<f32> {
    let pt = cpu.upload(DType::F32, shape, &bytes_of(pre)).unwrap();
    let dt = cpu.upload(DType::F32, shape, &bytes_of(dy)).unwrap();
    let mut xt = cpu.alloc(DType::F32, shape).unwrap();
    cpu.gelu_backward(&pt, &dt, &mut xt).unwrap();
    f32s_of(&cpu.download(&xt).unwrap())
}

/// Sequential forward and backward over microbatches 0, 1, ... with dW
/// accumulated in microbatch order.
fn reference(order: &[usize]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let cpu = CpuBackend::new();
    let (xs, dys) = microbatches(order);
    let ws: Vec<Vec<f32>> = (0..LAYERS).map(weight).collect();
    let mut dws: Vec<Vec<f32>> =
        (0..LAYERS).map(|l| vec![0.0; WIDTHS[l] * WIDTHS[l + 1]]).collect();
    let mut ys = Vec::new();
    for (x, dy) in xs.iter().zip(&dys) {
        let mut acts = vec![x.clone()];
        let mut pres = Vec::new();
        for l in 0..LAYERS {
            let (k, n) = (WIDTHS[l], WIDTHS[l + 1]);
            let a = &acts[l];
            let w = &ws[l];
            let mut pre = vec![0.0; BATCH * n];
            ref_gemm(BATCH, k, n, |r, i| a[r * k + i], |i, c| w[i * n + c], &mut pre, 1.0, 0.0);
            let act = cpu_gelu(&cpu, &pre, &[BATCH, n]);
            pres.push(pre);
            acts.push(act);
        }
        ys.push(acts[LAYERS].clone());
        let mut g = dy.clone();
        for l in (0..LAYERS).rev() {
            let (k, n) = (WIDTHS[l], WIDTHS[l + 1]);
            let dpre = cpu_gelu_backward(&cpu, &pres[l], &g, &[BATCH, n]);
            let input = &acts[l];
            // dW[k, n] += input^T [k, batch] @ dpre [batch, n]
            ref_gemm(
                k,
                BATCH,
                n,
                |r, i| input[i * k + r],
                |i, c| dpre[i * n + c],
                &mut dws[l],
                1.0,
                1.0,
            );
            // dX[batch, k] = dpre [batch, n] @ W^T [n, k]
            let w = &ws[l];
            let mut dx = vec![0.0; BATCH * k];
            ref_gemm(BATCH, n, k, |r, i| dpre[r * n + i], |i, c| w[c * n + i], &mut dx, 1.0, 0.0);
            g = dx;
        }
    }
    (ys, dws)
}

// ---------------------------------------------------------------- tests

#[test]
fn placement_examples_and_rejections() {
    let owners = |l, s| contiguous_placement(l, s).map(|p| p.owners);
    assert_eq!(owners(5, 3), Ok(vec![0, 0, 1, 1, 2]));
    assert_eq!(owners(3, 3), Ok(vec![0, 1, 2]));
    assert_eq!(owners(3, 1), Ok(vec![0, 0, 0]));
    assert_eq!(owners(3, 2), Ok(vec![0, 0, 1]));
    assert_eq!(owners(7, 3), Ok(vec![0, 0, 0, 1, 1, 2, 2]));
    assert_eq!(owners(4, 2), Ok(vec![0, 0, 1, 1]));
    assert_eq!(owners(0, 1), Err(Error::Schedule));
    assert_eq!(owners(1, 0), Err(Error::Schedule));
    assert_eq!(owners(0, 0), Err(Error::Schedule));
    assert_eq!(owners(2, 3), Err(Error::Schedule));
    // Rank r owns floor(L/S)+1 layers when r < L mod S, otherwise floor(L/S),
    // and owners are contiguous (non-decreasing, every rank present).
    for stages in 1..=6 {
        for layers in stages..=12 {
            let o = owners(layers, stages).unwrap();
            assert_eq!(o.len(), layers);
            assert!(o.windows(2).all(|w| w[0] <= w[1] && w[1] - w[0] <= 1));
            for r in 0..stages {
                let expect = layers / stages + usize::from(r < layers % stages);
                assert_eq!(
                    o.iter().filter(|&&x| x == r).count(),
                    expect,
                    "L{layers} S{stages} r{r}"
                );
            }
        }
    }
}

#[test]
fn single_rank_cpu_matches_an_independent_sequential_summation_within_0_ulps() {
    let order = [0, 1];
    let ranks = run_loopback(1, &order);
    let (y, dw) = gather(&ranks);
    assert_full_shape(&y, &dw, order.len());
    let (ry, rdw) = reference(&order);
    for (m, (got, want)) in y.iter().zip(&ry).enumerate() {
        assert_eq!(
            f32s_of(got).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "y[{m}] bits"
        );
    }
    for (l, (got, want)) in dw.iter().zip(&rdw).enumerate() {
        assert_eq!(
            f32s_of(got).iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "dW[{l}] bits"
        );
    }
}

#[test]
fn three_ranks_loopback_bitwise_equal_single_rank() {
    let order = [0, 1];
    let (sy, sdw) = gather(&run_loopback(1, &order));
    assert_full_shape(&sy, &sdw, order.len());
    let (y, dw) = gather(&run_loopback(3, &order));
    assert_full_shape(&y, &dw, order.len());
    assert_eq!(y, sy, "y bits");
    assert_eq!(dw, sdw, "dW bits");
}

#[test]
fn two_ranks_tcp_mesh_bitwise_equal_single_rank() {
    let order = [0, 1];
    let (sy, sdw) = gather(&run_loopback(1, &order));
    assert_full_shape(&sy, &sdw, order.len());
    let (y, dw) = gather(&run_tcp(2, &order));
    assert_full_shape(&y, &dw, order.len());
    assert_eq!(y, sy, "y bits");
    assert_eq!(dw, sdw, "dW bits");
}

#[test]
fn launch_counts_and_download_counts_for_1_stage_2_stages_and_3_stages() {
    let order = [0, 1];
    let m = order.len();
    for stages in 1..=3 {
        let ranks = run_loopback(stages, &order);
        assert_eq!(ranks.len(), stages);
        let total_layers: usize = ranks.iter().map(|r| r.owned_layers).sum();
        assert_eq!(total_layers, LAYERS, "stages {stages}: every layer owned once");
        for (rank, r) in ranks.iter().enumerate() {
            let l = r.owned_layers;
            let count = |name: &str| r.log.iter().filter(|e| **e == name).count();
            let ctx = format!("stages {stages} rank {rank}");
            assert!(l > 0, "{ctx}: owns a layer");
            assert_eq!(count("gemm"), 3 * l * m, "{ctx}: gemm");
            assert_eq!(count("gelu"), l * m, "{ctx}: gelu");
            assert_eq!(count("gelu-backward"), l * m, "{ctx}: gelu_backward");
            let compute =
                r.log.iter().filter(|e| !matches!(**e, "alloc" | "upload" | "download")).count();
            assert_eq!(compute, 5 * l * m, "{ctx}: no other compute");

            let links = usize::from(rank > 0) + usize::from(rank + 1 < stages);
            assert_eq!(r.sent.len(), links * m, "{ctx}: sends");
            assert_eq!(r.received, links * m, "{ctx}: recvs");
            assert_eq!(count("download"), r.sent.len(), "{ctx}: downloads equal sends");
            assert_eq!(count("upload"), r.received, "{ctx}: uploads equal recvs");
            if stages == 1 {
                assert_eq!(count("download"), 0, "{ctx}: single stage downloads");
                assert_eq!(count("upload"), 0, "{ctx}: single stage uploads");
            }
        }
    }
}

#[test]
fn boundary_payload_length_is_the_activation_wire_size() {
    let order = [0, 1];
    let m = order.len();
    let ranks = run_loopback(3, &order);
    // Placement [0, 1, 2]: forward boundaries carry widths n1 = 4 and n2 = 2;
    // backward boundaries carry the input widths of layers 1 and 2: 4 and 2.
    let expected: [Vec<usize>; 3] = [
        vec![wire_len(WIDTHS[1]); m],
        [vec![wire_len(WIDTHS[2]); m], vec![wire_len(WIDTHS[1]); m]].concat(),
        vec![wire_len(WIDTHS[2]); m],
    ];
    for (rank, (r, want)) in ranks.iter().zip(&expected).enumerate() {
        let mut got = r.sent.clone();
        got.sort_unstable();
        let mut want = want.clone();
        want.sort_unstable();
        assert_eq!(got, want, "rank {rank} payload lengths");
    }
}

#[test]
fn weight_storage_ids_survive_train() {
    let order = [0, 1];
    for stages in [1, 3] {
        let ranks = run_loopback(stages, &order);
        for (rank, r) in ranks.iter().enumerate() {
            let ctx = format!("stages {stages} rank {rank}");
            assert!(r.owned_layers > 0, "{ctx}: owns a layer");
            assert_eq!(r.weight_ids_after, r.weight_ids_before, "{ctx}: ids");
            assert_eq!(r.dw_ids.len(), r.owned_layers, "{ctx}: one dW per owned layer");
            for id in &r.dw_ids {
                assert!(!r.weight_ids_before.contains(id), "{ctx}: dW is not a weight");
            }
            let mut ids = r.dw_ids.clone();
            ids.sort_unstable();
            ids.dedup();
            assert_eq!(ids.len(), r.dw_ids.len(), "{ctx}: dW buffers distinct");
            let first = (0..LAYERS)
                .find(|&l| contiguous_placement(LAYERS, stages).unwrap().owners[l] == rank)
                .unwrap();
            for (i, bytes) in r.weight_bytes_after.iter().enumerate() {
                assert_eq!(*bytes, bytes_of(&weight(first + i)), "{ctx}: weight values unchanged");
            }
        }
    }
}

#[test]
fn schedule_used_is_one_f_one_b() {
    let order = [0, 1, 1, 0];
    let ranks = run_loopback(3, &order);
    let log = &ranks[0].log;
    let last_gelu = log.iter().rposition(|e| *e == "gelu");
    let first_back = log.iter().position(|e| *e == "gelu-backward");
    assert!(
        matches!((first_back, last_gelu), (Some(b), Some(g)) if b < g),
        "rank 0 runs a gelu-backward before its last gelu: {log:?}"
    );
    assert_eq!(ranks[2].y.len(), order.len(), "last rank produced every microbatch");
}

#[test]
fn counting_wrapper_logs_each_success_once() {
    // Guards the instrument the other tests rely on.
    let backend = Counting::new(CpuBackend::new());
    let a = backend.upload(DType::F32, &[1, 1], &bytes_of(&[2.0])).unwrap();
    let b = backend.upload(DType::F32, &[1, 1], &bytes_of(&[3.0])).unwrap();
    let mut c = backend.alloc(DType::F32, &[1, 1]).unwrap();
    backend.gemm(&a, &b, &mut c, 1.0, 0.0).unwrap();
    assert_eq!(backend.gemm(&a, &a, &mut c, 1.0, 0.0), Err(Error::Alias));
    assert_eq!(f32s_of(&backend.download(&c).unwrap()), vec![6.0]);
    assert_eq!(backend.entries(), vec!["upload", "upload", "alloc", "gemm", "download"]);
    assert_eq!(backend.count("gemm"), 1);
    assert_eq!(c.strides, row_major_strides(&[1, 1]));
}

#[test]
fn microbatch_count_must_match_inputs_and_dy() {
    // Amendment: every rank is told M. Rank 0 needs inputs.len() == M and the
    // last rank needs dy.len() == M; M == 0 is rejected.
    let cpu = CpuBackend::new();
    let placement = Placement { owners: vec![0, 0, 0] };
    let weights: Vec<_> = (0..LAYERS)
        .map(|l| {
            cpu.upload(DType::F32, &[WIDTHS[l], WIDTHS[l + 1]], &bytes_of(&weight(l))).unwrap()
        })
        .collect();
    let x = |v: &[f32]| cpu.upload(DType::F32, &[BATCH, WIDTHS[0]], &bytes_of(v)).unwrap();
    let d = |v: &[f32]| cpu.upload(DType::F32, &[BATCH, WIDTHS[LAYERS]], &bytes_of(v)).unwrap();
    let inputs = vec![x(&X0), x(&X1)];
    let dy = vec![d(&DY0), d(&DY1)];
    let run = |m: usize, inputs: &[Tensor<_>], dy: &[Tensor<_>]| {
        train(&cpu, &mut Solo, &placement, m, &weights, inputs, dy).map(|_| ())
    };
    assert_eq!(run(3, &inputs, &dy), Err(Error::Schedule), "M above inputs");
    assert_eq!(run(1, &inputs, &dy), Err(Error::Schedule), "M below inputs");
    assert_eq!(run(2, &inputs, &dy[..1]), Err(Error::Schedule), "dy short");
    assert_eq!(run(0, &[], &[]), Err(Error::Schedule), "M zero");
    assert_eq!(run(2, &inputs, &dy), Ok(()), "matching M");
}
