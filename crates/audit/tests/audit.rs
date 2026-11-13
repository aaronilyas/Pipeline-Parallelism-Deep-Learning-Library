//! Workspace audit: drives the other crates and checks their invariants. It adds no math.
//!
//! Every rank runs on its own thread, builds its own backend and tensors, and hands back only
//! plain bytes and logs. No `Store` crosses a thread.
//!
//! # Gating of host-specific tests
//!
//! * `heterogeneous_cpu_metal_matches_within_1e_3` is `cfg(target_os = "macos")`. Off macOS it
//!   does not exist: it is "not run" there.
//! * The audit crate has no `cuda` feature (the feature belongs to `linalg-cuda` and is switched
//!   on with `--features linalg-cuda/cuda`), so a `cfg(feature = ...)` here cannot observe it.
//!   Two runtime gates replace it:
//!   - `stubs_return_unsupported_device` (cuda half) detects the feature-off stub through
//!     `size_of::<CudaBackend>() == 0` (the stub is a unit struct; the real backend holds a
//!     stream, a cuBLAS handle and kernels). With the stub present, `new()` must be
//!     `Err(UnsupportedDevice)`; with the real backend present the half is skipped, because
//!     `new()` is then allowed to succeed.
//!   - `heterogeneous_cpu_cuda_matches_within_1e_3` runs when `CudaBackend::new()` succeeds. When
//!     the lead exports `LINALG_AUDIT_EXPECT_CUDA=1`, a failing `new()` is a test failure, not a
//!     skip. Without the variable and without a device the test prints "not run" and returns.
//! * `workspace_default_test_does_not_build_legacy_linalg` and `release_cpu_builds_without_counters`
//!   spawn cargo. `LINALG_AUDIT_INNER=1` is set on those children so the audit tests inside them
//!   return at once instead of recursing.

use std::cell::{Cell, RefCell};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::Duration;

use linalg_core::{Backend, DType, DeviceKind, Error, Tensor, Transport};
use linalg_cpu::CpuBackend;
use linalg_cuda::CudaBackend;
use linalg_pipeline::{contiguous_placement, train, Placement};

// ---------------------------------------------------------------- contract constants

/// The one default member that must not exist: the legacy root package.
const LEGACY_DEFAULT_MEMBER: &str = ".";
/// A dependency that metal and cuda must not have.
const BANNED_BACKEND_DEP: &str = "linalg-cpu";
/// Whether the feature-off / off-macOS stubs report `UnsupportedDevice`.
const STUB_REPORTS_UNSUPPORTED: bool = true;
/// Bytes before the shape words of a wire frame: magic, version, dtype tag, ndim.
const WIRE_HEADER_BYTES: usize = 16;
/// The only compute methods a CPU rank may call.
const EXPECTED_CPU_COMPUTE: [&str; 3] = ["gemm", "gelu", "gelu_backward"];
/// The guard attribute the cpu crate must put on every Mutex and atomic.
const CPU_GUARD: &str = "#[cfg(test)]";

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

/// Wire size of one `[BATCH, width]` f32 activation: header, two shape words, payload.
fn wire_len(width: usize) -> usize {
    WIRE_HEADER_BYTES + 4 * 2 + BATCH * width * 4
}

// ---------------------------------------------------------------- Counting

/// One successful backend call.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Call {
    method: &'static str,
    /// Every tensor dtype the call touched.
    dtypes: Vec<DType>,
    kind: DeviceKind,
}

/// Delegates every method to `inner` and logs `(method, dtype, kind)` per success.
struct Counting<B: Backend> {
    inner: B,
    log: Mutex<Vec<Call>>,
}

impl<B: Backend> Counting<B> {
    fn new(inner: B) -> Self {
        Self { inner, log: Mutex::new(Vec::new()) }
    }

    fn note<T>(
        &self,
        method: &'static str,
        dtypes: &[DType],
        result: Result<T, Error>,
    ) -> Result<T, Error> {
        if result.is_ok() {
            let call = Call { method, dtypes: dtypes.to_vec(), kind: self.inner.kind() };
            self.log.lock().unwrap().push(call);
        }
        result
    }

    fn entries(&self) -> Vec<Call> {
        self.log.lock().unwrap().clone()
    }
}

impl<B: Backend> Backend for Counting<B> {
    type Store = B::Store;

    fn kind(&self) -> DeviceKind {
        self.inner.kind()
    }

    fn alloc(&self, dtype: DType, shape: &[usize]) -> Result<Tensor<Self::Store>, Error> {
        self.note("alloc", &[dtype], self.inner.alloc(dtype, shape))
    }

    fn upload(
        &self,
        dtype: DType,
        shape: &[usize],
        row_major: &[u8],
    ) -> Result<Tensor<Self::Store>, Error> {
        self.note("upload", &[dtype], self.inner.upload(dtype, shape, row_major))
    }

    fn download(&self, tensor: &Tensor<Self::Store>) -> Result<Vec<u8>, Error> {
        self.note("download", &[tensor.dtype], self.inner.download(tensor))
    }

    fn gemm(
        &self,
        a: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        c: &mut Tensor<Self::Store>,
        alpha: f32,
        beta: f32,
    ) -> Result<(), Error> {
        let dtypes = [a.dtype, b.dtype, c.dtype];
        self.note("gemm", &dtypes, self.inner.gemm(a, b, c, alpha, beta))
    }

    fn gelu(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let dtypes = [x.dtype, y.dtype];
        self.note("gelu", &dtypes, self.inner.gelu(x, y))
    }

    fn gelu_backward(
        &self,
        pre: &Tensor<Self::Store>,
        dy: &Tensor<Self::Store>,
        dx: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        let dtypes = [pre.dtype, dy.dtype, dx.dtype];
        self.note("gelu_backward", &dtypes, self.inner.gelu_backward(pre, dy, dx))
    }

    fn download_into(&self, tensor: &Tensor<Self::Store>, out: &mut [u8]) -> Result<(), Error> {
        self.note("download_into", &[tensor.dtype], self.inner.download_into(tensor, out))
    }

    fn upload_into(&self, bytes: &[u8], tensor: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let dtypes = [tensor.dtype];
        self.note("upload_into", &dtypes, self.inner.upload_into(bytes, tensor))
    }

    fn sync(&self) -> Result<(), Error> {
        self.note("sync", &[], self.inner.sync())
    }

    fn add_bias(
        &self,
        x: &Tensor<Self::Store>,
        b: &Tensor<Self::Store>,
        y: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        let dtypes = [x.dtype, b.dtype, y.dtype];
        self.note("add_bias", &dtypes, self.inner.add_bias(x, b, y))
    }

    fn bias_grad(
        &self,
        dpre: &Tensor<Self::Store>,
        db: &mut Tensor<Self::Store>,
    ) -> Result<(), Error> {
        let dtypes = [dpre.dtype, db.dtype];
        self.note("bias_grad", &dtypes, self.inner.bias_grad(dpre, db))
    }

    fn mse_grad(
        &self,
        y: &Tensor<Self::Store>,
        target: &Tensor<Self::Store>,
        dy: &mut Tensor<Self::Store>,
        loss: &mut f32,
    ) -> Result<(), Error> {
        let dtypes = [y.dtype, target.dtype, dy.dtype];
        self.note("mse_grad", &dtypes, self.inner.mse_grad(y, target, dy, loss))
    }

    fn sgd_step(
        &self,
        w: &mut Tensor<Self::Store>,
        dw: &mut Tensor<Self::Store>,
        lr: f32,
    ) -> Result<(), Error> {
        let dtypes = [w.dtype, dw.dtype];
        self.note("sgd_step", &dtypes, self.inner.sgd_step(w, dw, lr))
    }

    fn convert(&self, x: &Tensor<Self::Store>, y: &mut Tensor<Self::Store>) -> Result<(), Error> {
        let dtypes = [x.dtype, y.dtype];
        self.note("convert", &dtypes, self.inner.convert(x, y))
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
    /// The Counting log, snapshotted right after `train` (before results are downloaded).
    calls: Vec<Call>,
    sent: Vec<usize>,
    received: usize,
    owned_layers: usize,
    /// The backend's own counters, read right after `train`.
    launches: Vec<&'static str>,
    downloads: u64,
    allocs: u64,
}

fn owned(placement: &Placement, rank: usize) -> Vec<usize> {
    (0..placement.owners.len()).filter(|&l| placement.owners[l] == rank).collect()
}

/// Builds the rank's backend on the current thread and calls `train` generically over the
/// `Counting` wrapper and the transport. No trait object is involved.
fn run_rank<B: Backend, T: Transport>(
    make: impl FnOnce() -> B,
    transport: T,
    stages: usize,
    order: &[usize],
) -> RankResult {
    let rank = transport.rank();
    let mut transport = Logged::new(transport);
    let backend = Counting::new(make());
    let placement = contiguous_placement(LAYERS, stages).expect("placement");
    let layers = owned(&placement, rank);
    let weights: Vec<_> = layers
        .iter()
        .map(|&l| {
            backend.upload(DType::F32, &[WIDTHS[l], WIDTHS[l + 1]], &bytes_of(&weight(l))).unwrap()
        })
        .collect();
    let (xs, dys) = microbatches(order);
    let inputs: Vec<_> = if rank == 0 {
        xs.iter()
            .map(|x| backend.upload(DType::F32, &[BATCH, WIDTHS[0]], &bytes_of(x)).unwrap())
            .collect()
    } else {
        Vec::new()
    };
    let dy: Vec<_> = if rank + 1 == stages {
        dys.iter()
            .map(|d| backend.upload(DType::F32, &[BATCH, WIDTHS[LAYERS]], &bytes_of(d)).unwrap())
            .collect()
    } else {
        Vec::new()
    };

    let out = train(&backend, &mut transport, &placement, order.len(), &weights, &inputs, &dy)
        .expect("train succeeds");

    let calls = backend.entries();
    let launches = backend.inner.debug_launches();
    let downloads = backend.inner.debug_downloads();
    let allocs = backend.inner.debug_allocs();
    RankResult {
        y: out.y.iter().map(|t| backend.inner.download(t).unwrap()).collect(),
        dw: out.dw.iter().map(|t| backend.inner.download(t).unwrap()).collect(),
        calls,
        sent: transport.sent.take(),
        received: transport.received.get(),
        owned_layers: layers.len(),
        launches,
        downloads,
        allocs,
    }
}

fn run_single(order: &[usize]) -> RankResult {
    run_rank(CpuBackend::new, Solo, 1, order)
}

fn run_cpu_loopback(stages: usize, order: &[usize]) -> Vec<RankResult> {
    let endpoints = linalg_transport::loopback(stages).expect("loopback");
    std::thread::scope(|scope| {
        let handles: Vec<_> = endpoints
            .into_iter()
            .map(|ep| scope.spawn(move || run_rank(CpuBackend::new, ep, stages, order)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

fn run_cpu_tcp(stages: usize, order: &[usize]) -> Vec<RankResult> {
    let endpoints = linalg_transport::localhost_mesh(stages).expect("mesh");
    std::thread::scope(|scope| {
        let handles: Vec<_> = endpoints
            .into_iter()
            .map(|ep| scope.spawn(move || run_rank(CpuBackend::new, ep, stages, order)))
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Rank 0 on a CPU, rank 1 on another backend, over a two-rank localhost mesh.
fn run_cpu_then<B: Backend>(
    make_rank1: impl FnOnce() -> B + Send,
    order: &[usize],
) -> Vec<RankResult> {
    let mut endpoints = linalg_transport::localhost_mesh(2).expect("mesh").into_iter();
    let (e0, e1) = (endpoints.next().unwrap(), endpoints.next().unwrap());
    std::thread::scope(|scope| {
        let h0 = scope.spawn(move || run_rank(CpuBackend::new, e0, 2, order));
        let h1 = scope.spawn(move || run_rank(make_rank1, e1, 2, order));
        vec![h0.join().unwrap(), h1.join().unwrap()]
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

fn assert_close(got: &[Vec<u8>], want: &[Vec<u8>], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: count");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let (g, w) = (f32s_of(g), f32s_of(w));
        assert_eq!(g.len(), w.len(), "{what}[{i}]: length");
        for (j, (a, b)) in g.iter().zip(&w).enumerate() {
            assert!((a - b).abs() <= tol, "{what}[{i}][{j}]: {a} vs {b} exceeds {tol}");
        }
    }
}

/// The expected payload lengths each rank of a 3-stage run sends, any order.
fn expected_three_stage_sends(microbatches: usize) -> [Vec<usize>; 3] {
    let m = microbatches;
    [
        vec![wire_len(WIDTHS[1]); m],
        [vec![wire_len(WIDTHS[2]); m], vec![wire_len(WIDTHS[1]); m]].concat(),
        vec![wire_len(WIDTHS[2]); m],
    ]
}

fn assert_payloads(ranks: &[RankResult], microbatches: usize, what: &str) {
    for (rank, (r, want)) in ranks.iter().zip(&expected_three_stage_sends(microbatches)).enumerate()
    {
        let mut got = r.sent.clone();
        got.sort_unstable();
        let mut want = want.clone();
        want.sort_unstable();
        assert_eq!(got, want, "{what}: rank {rank} payload lengths = activation wire size");
        // Every payload sent to a neighbour is received there: rank r receives what its
        // neighbours sent it, and the run is closed (no unread frames are left to count).
        let neighbours = usize::from(rank > 0) + usize::from(rank + 1 < ranks.len());
        assert_eq!(r.received, neighbours * microbatches, "{what}: rank {rank} recvs");
    }
}

// ---------------------------------------------------------------- workspace helpers

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().expect("workspace root")
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn inner_run() -> bool {
    std::env::var_os("LINALG_AUDIT_INNER").is_some_and(|v| v == "1")
}

/// A cargo child with its own target dir (so the outer build lock is not contended), the
/// recursion guard set, and the CUDA expectation cleared.
fn cargo(target_subdir: &str) -> Command {
    let root = workspace_root();
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let mut cmd = Command::new(cargo);
    cmd.current_dir(&root)
        .env("CARGO_TARGET_DIR", root.join("target").join(target_subdir))
        .env("LINALG_AUDIT_INNER", "1")
        // The child builds with default features, so a CUDA device cannot be expected of it.
        .env_remove("LINALG_AUDIT_EXPECT_CUDA");
    cmd
}

fn tail(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(60)..].join("\n")
}

// ---------------------------------------------------------------- workspace_default

/// The quoted entries of `default-members = [ ... ]` in a manifest.
fn default_members(manifest: &str) -> Vec<String> {
    let start = manifest.find("default-members").expect("default-members key");
    let rest = &manifest[start..];
    let open = rest.find('[').expect("default-members opens");
    let close = rest.find(']').expect("default-members closes");
    rest[open + 1..close]
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// True when a `cargo --message-format=json` line is a build artifact of the legacy root
/// package: its manifest is the root manifest, or its target is named exactly `linalg`.
fn is_legacy_artifact(line: &str, root: &Path) -> bool {
    if !line.contains("\"reason\":\"compiler-artifact\"") {
        return false;
    }
    let root_manifest = format!("\"manifest_path\":\"{}/Cargo.toml\"", root.display());
    line.contains(&root_manifest) || line.contains("\"name\":\"linalg\",")
}

#[test]
fn workspace_default_test_does_not_build_legacy_linalg() {
    if inner_run() {
        return;
    }
    let root = workspace_root();
    let manifest = read(&root.join("Cargo.toml"));
    let members = default_members(&manifest);
    assert!(!members.is_empty(), "default-members is non-empty");
    assert!(
        !members.iter().any(|m| m == LEGACY_DEFAULT_MEMBER),
        "default-members {members:?} must not contain the legacy root package"
    );
    assert!(
        members.iter().all(|m| m.starts_with("crates/")),
        "every default member lives under crates/: {members:?}"
    );

    // Running the root `cargo test` is Linux-only: the legacy package builds Metal code, and
    // on other hosts the audit still checks default-members above but does not spawn cargo.
    if !cfg!(target_os = "linux") {
        eprintln!("not run: root `cargo test` is only spawned on Linux");
        return;
    }
    let output = cargo("audit-default")
        .args(["test", "--message-format=json"])
        .output()
        .expect("spawn cargo test");
    assert!(
        output.status.success(),
        "inner `cargo test` failed ({}):\n{}",
        output.status,
        tail(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let artifacts: Vec<&str> =
        stdout.lines().filter(|l| l.contains("\"reason\":\"compiler-artifact\"")).collect();
    assert!(
        artifacts.iter().any(|l| l.contains("\"name\":\"linalg_core\"")),
        "the inner run built the workspace crates (scan is not vacuous)"
    );
    let legacy: Vec<&&str> = artifacts.iter().filter(|l| is_legacy_artifact(l, &root)).collect();
    assert!(legacy.is_empty(), "the default `cargo test` compiled the legacy package: {legacy:?}");
}

// ---------------------------------------------------------------- manifests

/// Dependency names declared in any `*dependencies*` table of a manifest, plus the `path` values.
fn manifest_deps(manifest: &str) -> (BTreeSet<String>, Vec<String>) {
    let mut names = BTreeSet::new();
    let mut paths = Vec::new();
    let mut in_deps = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if let Some(header) = line.strip_prefix('[') {
            let header = header.trim_end_matches(']').trim_start_matches('[');
            in_deps = header.contains("dependencies");
            // `[dependencies.name]` and `[target.x.dependencies.name]` name the dep in the header.
            if let Some(at) = header.find("dependencies.") {
                names.insert(header[at + "dependencies.".len()..].trim_matches('"').to_string());
            }
            continue;
        }
        if !in_deps || line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            names.insert(key.trim().trim_matches('"').to_string());
            if let Some(p) = value.split("path").nth(1) {
                if let Some(q) = p.split('"').nth(1) {
                    paths.push(q.to_string());
                }
            }
        }
    }
    (names, paths)
}

/// Transitive dependency closure of `start` in Cargo.lock (every target's edges, so a superset).
fn lock_closure(lock: &str, start: &str) -> BTreeSet<String> {
    let mut graph: Vec<(String, Vec<String>)> = Vec::new();
    for block in lock.split("[[package]]").skip(1) {
        let mut name = String::new();
        let mut deps = Vec::new();
        let mut in_deps = false;
        for line in block.lines().map(str::trim) {
            if let Some(v) = line.strip_prefix("name = ") {
                name = v.trim_matches('"').to_string();
            } else if line.starts_with("dependencies = [") {
                in_deps = true;
            } else if in_deps && line == "]" {
                in_deps = false;
            } else if in_deps {
                let dep = line.trim_matches(|c| c == '"' || c == ',');
                deps.push(dep.split(' ').next().unwrap_or("").to_string());
            }
        }
        graph.push((name, deps));
    }
    let mut seen = BTreeSet::new();
    let mut stack = vec![start.to_string()];
    while let Some(next) = stack.pop() {
        if let Some((_, deps)) = graph.iter().find(|(n, _)| *n == next) {
            for d in deps {
                if seen.insert(d.clone()) {
                    stack.push(d.clone());
                }
            }
        }
    }
    seen
}

#[test]
fn backend_manifests_do_not_depend_on_the_cpu_crate() {
    let root = workspace_root();
    let lock = read(&root.join("Cargo.lock"));
    for (crate_dir, name, other) in
        [("metal", "linalg-metal", "linalg-cuda"), ("cuda", "linalg-cuda", "linalg-metal")]
    {
        let manifest = read(&root.join("crates").join(crate_dir).join("Cargo.toml"));
        let (deps, paths) = manifest_deps(&manifest);
        for banned in ["linalg-cpu", other, BANNED_BACKEND_DEP] {
            assert!(!deps.contains(banned), "{name} declares a dependency on {banned}: {deps:?}");
        }
        for p in &paths {
            let target = p.trim_end_matches('/');
            assert!(
                !target.ends_with("/cpu")
                    && !target.ends_with("/metal")
                    && !target.ends_with("/cuda"),
                "{name} has a path dependency on a backend crate: {p}"
            );
        }
        // Transitively, through every package the lock file wires up.
        let closure = lock_closure(&lock, name);
        for banned in ["linalg-cpu", other] {
            assert!(!closure.contains(banned), "{name} reaches {banned} through {closure:?}");
        }
        // The lock must actually describe this crate, or the closure check is vacuous.
        assert!(lock.contains(&format!("name = \"{name}\"")), "{name} is in Cargo.lock");
    }
}

// ---------------------------------------------------------------- stubs

#[test]
fn stubs_return_unsupported_device() {
    // CUDA: with the `cuda` feature off the crate exports a unit-struct stub, which is the only
    // case where `CudaBackend` has size zero. A real backend may succeed, so that case is skipped.
    if std::mem::size_of::<CudaBackend>() == 0 {
        let unsupported = matches!(CudaBackend::new(), Err(Error::UnsupportedDevice));
        assert_eq!(unsupported, STUB_REPORTS_UNSUPPORTED, "feature-off cuda stub: new()");
    } else {
        eprintln!("not run (cuda half): linalg-cuda was built with its `cuda` feature");
    }

    // Metal: off macOS only.
    #[cfg(not(target_os = "macos"))]
    {
        let unsupported = matches!(
            linalg_metal::MetalBackend::new(),
            Err(linalg_metal::Error::UnsupportedDevice)
        );
        assert_eq!(unsupported, STUB_REPORTS_UNSUPPORTED, "off-macOS metal stub: new()");
    }
}

// ---------------------------------------------------------------- cpu + tcp + loopback

#[test]
fn cpu_tcp_three_ranks_match_single_rank_bits() {
    let order = [0, 1, 1, 0];
    let m = order.len();
    let single = run_single(&order);
    let (sy, sdw) = gather(std::slice::from_ref(&single));
    assert_full_shape(&sy, &sdw, m);

    let loopback = run_cpu_loopback(3, &order);
    let (y, dw) = gather(&loopback);
    assert_full_shape(&y, &dw, m);
    assert_eq!(y, sy, "loopback y bits");
    assert_eq!(dw, sdw, "loopback dW bits");
    assert_payloads(&loopback, m, "loopback");

    let tcp = run_cpu_tcp(3, &order);
    let (y, dw) = gather(&tcp);
    assert_full_shape(&y, &dw, m);
    assert_eq!(y, sy, "tcp y bits");
    assert_eq!(dw, sdw, "tcp dW bits");
    assert_payloads(&tcp, m, "tcp");
}

/// Whether `code` (comments already stripped) contains `dyn` as a whole identifier token, so
/// `&dyn Tr`, `Box<dyn\nTr>` and `dyn(Fn())` match but `dynamic` and `my_dyn` do not.
fn has_dyn_token(code: &str) -> bool {
    code.split(|c: char| !(c.is_alphanumeric() || c == '_')).any(|tok| tok == "dyn")
}

/// No `dyn` in the pipeline source: `train` is generic over `Backend` and `Transport`. Walks
/// `dir` recursively and returns how many files it read (it panics if that is none).
fn assert_no_trait_objects(dir: &Path) -> usize {
    let files = rust_files(dir);
    assert!(!files.is_empty(), "no sources under {}", dir.display());
    for path in &files {
        let code = strip_comments(&read(path)).join("\n");
        assert!(!has_dyn_token(&code), "{} uses a trait object", path.display());
    }
    files.len()
}

fn assert_cpu_calls(r: &RankResult, ctx: &str) {
    assert!(!r.calls.is_empty(), "{ctx}: the rank called the backend");
    for call in &r.calls {
        assert_eq!(call.kind, DeviceKind::Cpu, "{ctx}: {call:?} kind");
        assert!(call.dtypes.iter().all(|d| *d == DType::F32), "{ctx}: {call:?} dtype");
    }
    let compute: BTreeSet<&str> = r
        .calls
        .iter()
        .map(|c| c.method)
        .filter(|m| !matches!(*m, "alloc" | "upload" | "download"))
        .collect();
    let want: BTreeSet<&str> = EXPECTED_CPU_COMPUTE.into_iter().collect();
    assert_eq!(compute, want, "{ctx}: compute methods");
    // The CPU crate is a non-test dependency here, so it keeps no counters.
    assert!(r.launches.is_empty(), "{ctx}: cpu crate recorded launches {:?}", r.launches);
    assert_eq!(r.allocs, 0, "{ctx}: cpu crate counted allocs");
    assert_eq!(r.downloads, 0, "{ctx}: cpu crate counted downloads");
}

#[test]
fn cpu_launch_names_stay_on_the_cpu_path() {
    let scanned = assert_no_trait_objects(&workspace_root().join("crates/pipeline/src"));
    assert!(scanned > 0, "the trait-object scan read no files");
    let order = [0, 1];
    // Both transports, one generic `run_rank::<Counting<CpuBackend>, _>` instantiation each.
    let single = run_single(&order);
    assert_cpu_calls(&single, "single");
    for (name, ranks) in
        [("loopback", run_cpu_loopback(3, &order)), ("tcp", run_cpu_tcp(3, &order))]
    {
        assert_eq!(ranks.len(), 3);
        let layers: usize = ranks.iter().map(|r| r.owned_layers).sum();
        assert_eq!(layers, LAYERS, "{name}: every layer owned once");
        for (rank, r) in ranks.iter().enumerate() {
            assert_cpu_calls(r, &format!("{name} rank {rank}"));
        }
    }
}

// ---------------------------------------------------------------- source scan

/// Source lines with comments removed. Understands what a plain scan would trip over: `//` and
/// `/*` inside string, byte-string and raw-string literals (`r"…"`, `br#"…"#`, any hash count)
/// are not comments; a quote inside a char literal (`'"'`, `'\\''`) opens no string; lifetimes
/// (`'a`) are not char literals; quotes inside comments open nothing; block comments nest, as in
/// Rust. Literal text is kept verbatim (so a `Mutex` named in a string is still reported: the
/// scan errs towards flagging). Line numbering is preserved: every `\n` in the input ends a line.
fn strip_comments(src: &str) -> Vec<String> {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        Line,
        Block(usize),
        Str,
        Raw(usize),
    }
    let chars: Vec<char> = src.chars().collect();
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut state = State::Code;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if c == '\n' {
            out.push(std::mem::take(&mut cur));
            if matches!(state, State::Line) {
                state = State::Code;
            }
            i += 1;
            continue;
        }
        match state {
            State::Line => i += 1,
            State::Block(depth) => {
                if c == '*' && next == Some('/') {
                    state = if depth == 1 { State::Code } else { State::Block(depth - 1) };
                    i += 2;
                } else if c == '/' && next == Some('*') {
                    state = State::Block(depth + 1);
                    i += 2;
                } else {
                    i += 1;
                }
            }
            State::Str => {
                cur.push(c);
                if c == '\\' {
                    // An escaped newline still ends the line; any other escaped char is literal.
                    match next {
                        Some('\n') | None => i += 1,
                        Some(n) => {
                            cur.push(n);
                            i += 2;
                        }
                    }
                } else {
                    if c == '"' {
                        state = State::Code;
                    }
                    i += 1;
                }
            }
            State::Raw(hashes) => {
                cur.push(c);
                i += 1;
                if c == '"' && (0..hashes).all(|k| chars.get(i + k) == Some(&'#')) {
                    for _ in 0..hashes {
                        cur.push('#');
                    }
                    i += hashes;
                    state = State::Code;
                }
            }
            State::Code => {
                if c == '/' && next == Some('/') {
                    state = State::Line;
                    i += 2;
                } else if c == '/' && next == Some('*') {
                    state = State::Block(1);
                    i += 2;
                } else if c == '"' {
                    cur.push(c);
                    state = State::Str;
                    i += 1;
                } else if (c == 'r' || c == 'b') && (i == 0 || !is_ident(chars[i - 1])) {
                    // Possible literal prefix: b"…", r"…", r#"…"#, br#"…"#.
                    let mut j = i;
                    if chars[j] == 'b' {
                        j += 1;
                    }
                    let raw = chars.get(j) == Some(&'r');
                    if raw {
                        j += 1;
                    }
                    let mut hashes = 0;
                    while raw && chars.get(j + hashes) == Some(&'#') {
                        hashes += 1;
                    }
                    if chars.get(j + hashes) == Some(&'"') && (raw || j > i) {
                        cur.extend(&chars[i..=j + hashes]);
                        state = if raw { State::Raw(hashes) } else { State::Str };
                        i = j + hashes + 1;
                    } else {
                        cur.push(c);
                        i += 1;
                    }
                } else if c == '\'' {
                    // Char literal ('x', '\n', '\'', '\u{…}') or lifetime ('a).
                    let end = if next == Some('\\') {
                        let mut j = i + 3;
                        while j < chars.len() && chars[j] != '\'' && chars[j] != '\n' {
                            j += 1;
                        }
                        (chars.get(j) == Some(&'\'')).then_some(j)
                    } else {
                        (chars.get(i + 2) == Some(&'\'') && next != Some('\n')).then_some(i + 2)
                    };
                    match end {
                        Some(j) => {
                            cur.extend(&chars[i..=j]);
                            i = j + 1;
                        }
                        None => {
                            cur.push(c);
                            i += 1;
                        }
                    }
                } else {
                    cur.push(c);
                    i += 1;
                }
            }
        }
    }
    if !(src.is_empty() || src.ends_with('\n')) {
        out.push(cur);
    }
    out
}

fn squash(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// Net bracket depth change of the characters in `s`.
fn depth_delta(s: &str) -> i32 {
    s.chars()
        .map(|c| match c {
            '{' | '(' | '[' => 1,
            '}' | ')' | ']' => -1,
            _ => 0,
        })
        .sum()
}

/// Which lines sit inside an item, field, statement or block preceded by an accepted `cfg`
/// attribute, and which `mod name;` declarations carry one.
///
/// Rules (deliberately conservative: when in doubt a line is reported as unguarded):
/// * An attribute is accepted when its whitespace-free text equals one of `allowed`. Attributes
///   with `not(...)`, `all(...)`, or any other spelling are not accepted.
/// * Stacked attributes guard the next item when any one of them is accepted.
/// * The guarded region starts at the first code line after the attributes and ends at the first
///   of: a `;` or `,` at bracket depth 0, or a `}` that returns the depth to 0. So a `struct`,
///   `fn`, `impl`, `mod x { }`, `{ block }`, `let`, `use`, or a field each end where they end.
///   An `else` branch after a guarded `if` block is therefore NOT covered.
/// * An inner `#![cfg(...)]` that is accepted guards the whole file.
struct Guards {
    guarded: Vec<bool>,
    whole_file: bool,
    guarded_mods: Vec<String>,
}

fn find_guards(lines: &[String], allowed: &[String]) -> Guards {
    let n = lines.len();
    let mut guards = Guards { guarded: vec![false; n], whole_file: false, guarded_mods: vec![] };
    let mut i = 0;
    while i < n {
        let trimmed = lines[i].trim();
        if !trimmed.starts_with("#[") && !trimmed.starts_with("#![") {
            i += 1;
            continue;
        }
        // Gather consecutive attributes (each may span lines).
        let mut accepted = false;
        let mut j = i;
        while j < n {
            let t = lines[j].trim();
            if t.is_empty() {
                j += 1;
                continue;
            }
            if !t.starts_with("#[") && !t.starts_with("#![") {
                break;
            }
            let inner = t.starts_with("#![");
            let mut text = String::new();
            let mut depth = 0;
            loop {
                text.push_str(lines[j].trim());
                depth += depth_delta(lines[j].trim());
                j += 1;
                if depth <= 0 || j >= n {
                    break;
                }
            }
            let norm = squash(&text).replacen("#![", "#[", 1);
            if allowed.contains(&norm) {
                if inner {
                    guards.whole_file = true;
                } else {
                    accepted = true;
                }
            }
        }
        if accepted && j < n {
            // Region: from line j to the end of the first complete item.
            let mut depth = 0;
            let mut end = j;
            'scan: for (k, line) in lines.iter().enumerate().skip(j) {
                end = k;
                for c in line.chars() {
                    match c {
                        '{' | '(' | '[' => depth += 1,
                        '}' | ')' | ']' => {
                            depth -= 1;
                            if c == '}' && depth <= 0 {
                                break 'scan;
                            }
                        }
                        ';' | ',' if depth == 0 => break 'scan,
                        _ => {}
                    }
                }
            }
            for flag in &mut guards.guarded[j..=end] {
                *flag = true;
            }
            let head = lines[j].trim();
            let head = head.strip_prefix("pub ").unwrap_or(head);
            if let Some(rest) = head.strip_prefix("mod ") {
                if let Some(name) = rest.trim().strip_suffix(';') {
                    guards.guarded_mods.push(name.trim().to_string());
                }
            }
        }
        i = j.max(i + 1);
    }
    guards
}

fn allowed_guards(attrs: &[&str]) -> Vec<String> {
    attrs.iter().map(|a| squash(a)).collect()
}

/// Lines (1-based, with text) that mention `Mutex`, `Atomic`, or `atomic` outside an accepted
/// guard, in one source text. `file_guarded` exempts the whole text.
fn unguarded_sync(
    src: &str,
    allowed: &[String],
    file_guarded: bool,
) -> (Vec<(usize, String)>, usize) {
    let lines = strip_comments(src);
    let guards = find_guards(&lines, allowed);
    let mut bad = Vec::new();
    let mut guarded_hits = 0;
    for (idx, line) in lines.iter().enumerate() {
        let hit = line.contains("Mutex") || line.contains("Atomic") || line.contains("atomic");
        if !hit {
            continue;
        }
        if file_guarded || guards.whole_file || guards.guarded[idx] {
            guarded_hits += 1;
        } else {
            bad.push((idx + 1, line.trim().to_string()));
        }
    }
    (bad, guarded_hits)
}

/// Every `.rs` file under `dir`, recursively, sorted.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![dir.to_path_buf()];
    while let Some(d) = pending.pop() {
        for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read {}: {e}", d.display())) {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

/// Scans every `.rs` under `dir`, nested directories included. A `mod name;` in `lib.rs` that
/// carries an accepted guard (e.g. `#[cfg(test)] mod invariants;`) wholly guards `name.rs` and
/// everything under `name/`. Returns violations, the number of guarded hits (proof the scan
/// saw the counters) and the number of files scanned (so it cannot pass vacuously).
fn scan_crate(dir: &Path, allowed: &[String]) -> (Vec<String>, usize, usize) {
    let lib = read(&dir.join("lib.rs"));
    let guarded_mods = find_guards(&strip_comments(&lib), allowed).guarded_mods;
    let mut violations = Vec::new();
    let mut guarded_hits = 0;
    let files = rust_files(dir);
    assert!(!files.is_empty(), "no sources under {}", dir.display());
    for file in &files {
        let rel = file.strip_prefix(dir).expect("under dir");
        let top = rel.components().next().expect("non-empty path").as_os_str();
        let top = Path::new(top).file_stem().unwrap().to_string_lossy().to_string();
        let file_guarded = guarded_mods.contains(&top);
        let (bad, hits) = unguarded_sync(&read(file), allowed, file_guarded);
        guarded_hits += hits;
        for (line, text) in bad {
            violations.push(format!("{}:{line}: {text}", file.display()));
        }
    }
    (violations, guarded_hits, files.len())
}

/// A throwaway crate `src/` under the cargo target tmpdir with the given `(relative path,
/// contents)` files; returns the `src/` directory.
fn scratch_crate(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("audit-scan").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    for (rel, text) in files {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).expect("create scratch dir");
        std::fs::write(&path, text).expect("write scratch file");
    }
    dir
}

#[test]
fn source_scan_rules_self_check() {
    let test_only = allowed_guards(&["#[cfg(test)]"]);
    let counters = allowed_guards(&[
        "#[cfg(test)]",
        "#[cfg(any(test, feature = \"debug-counters\"))]",
        "#[cfg(any(feature = \"debug-counters\", test))]",
    ]);
    let flagged = |src: &str, allowed: &[String]| unguarded_sync(src, allowed, false).0.len();

    assert_eq!(flagged("use std::sync::Mutex;\n", &test_only), 1, "bare use");
    assert_eq!(flagged("#[cfg(test)]\nuse std::sync::Mutex;\n", &test_only), 0, "guarded use");
    assert_eq!(flagged("let c = AtomicU64::new(0);\n", &counters), 1, "bare atomic");
    assert_eq!(flagged("use std::sync::atomic::Ordering;\n", &counters), 1, "bare atomic path");
    assert_eq!(flagged("// a Mutex in a comment\n/* an atomic\n in a block */\n", &test_only), 0);
    assert_eq!(flagged("/// docs say Mutex\nfn f() {}\n", &test_only), 0, "doc comment");

    let counters_src = "#[cfg(any(test, feature = \"debug-counters\"))]\nuse std::sync::Mutex;\n";
    assert_eq!(flagged(counters_src, &test_only), 1, "debug-counters guard is not enough on cpu");
    assert_eq!(flagged(counters_src, &counters), 0, "debug-counters guard on metal/cuda");
    let wrong = "#[cfg(not(test))]\nuse std::sync::Mutex;\n";
    assert_eq!(flagged(wrong, &counters), 1, "not(test) is no guard");

    let fields = "struct S {\n    #[cfg(test)]\n    stats: std::sync::Mutex<Stats>,\n    other: Mutex<u8>,\n}\n";
    assert_eq!(unguarded_sync(fields, &test_only, false).0, vec![(4, "other: Mutex<u8>,".into())]);

    let ctor = "fn new() -> Self {\n    Self {\n        #[cfg(test)]\n        stats: Mutex::new(Stats {\n            a: 0,\n        }),\n        b: 1,\n    }\n}\n";
    assert_eq!(flagged(ctor, &test_only), 0, "multi-line guarded field initializer");

    let block = "fn f() {\n    #[cfg(test)]\n    {\n        self.stats.lock().unwrap();\n        let _ = Mutex::new(0);\n    }\n    let late = Mutex::new(1);\n}\n";
    assert_eq!(unguarded_sync(block, &test_only, false).0.len(), 1, "only the block is guarded");

    let module = "#[cfg(test)]\nmod tests {\n    use std::sync::Mutex;\n    fn g() {\n        let _m = Mutex::new(0);\n    }\n}\nstatic X: Mutex<u8> = todo();\n";
    assert_eq!(unguarded_sync(module, &test_only, false).0.len(), 1, "after the test module");

    // Review round 1, finding 1: comment stripping must understand literals and nesting.
    let m = "let m = Mutex::new(0);";
    let hidden = |prefix: &str| format!("{prefix}{m}\n");
    assert_eq!(flagged(&hidden("let _ = \"//\"; "), &test_only), 1, "// in a string");
    assert_eq!(flagged(&hidden("let _ = r\"//\"; "), &test_only), 1, "// in a raw string");
    assert_eq!(flagged(&hidden("let _ = r#\"a\"//b\"#; "), &test_only), 1, "// in r#\"..\"#");
    assert_eq!(flagged(&hidden("let _ = br##\"//\"#\"##; "), &test_only), 1, "// in br##\"..\"##");
    assert_eq!(flagged(&hidden("let _ = b\"//\"; "), &test_only), 1, "// in a byte string");
    assert_eq!(flagged(&hidden("let _ = \"a\\\"//\"; "), &test_only), 1, "escaped quote in string");
    assert_eq!(flagged(&format!("let s = \"/*\";\n{m}\n"), &test_only), 1, "/* in a string");
    assert_eq!(
        flagged(&format!("let s = \"a\n// b\";\n{m}\n"), &test_only),
        1,
        "multi-line string"
    );
    assert_eq!(flagged(&hidden("let q = '\"'; "), &test_only), 1, "quote char literal");
    assert_eq!(flagged(&hidden("let q = '\\''; "), &test_only), 1, "escaped quote char literal");
    assert_eq!(
        flagged(&hidden("let q = '/'; let r = '/'; "), &test_only),
        1,
        "slash char literals"
    );
    assert_eq!(flagged(&format!("fn f<'a>(x: &'a str) {{ {m} }}\n"), &test_only), 1, "lifetimes");
    assert_eq!(flagged(&format!("/* \" */ {m}\n"), &test_only), 1, "quote inside a block comment");
    assert_eq!(
        flagged(&format!("// \" not a string\n{m}\n"), &test_only),
        1,
        "quote in a line comment"
    );
    assert_eq!(flagged("/* a /* nested Mutex */ still Mutex */\n", &test_only), 0, "nested block");
    assert_eq!(flagged(&format!("/* a /* b */ c */ {m}\n"), &test_only), 1, "code after nesting");
    assert_eq!(
        flagged("let s = \"// Mutex\";\n", &test_only),
        1,
        "Mutex text in a string stays flagged"
    );
    let spread = strip_comments("let a = 1; /* x\n y */ let b = 2;\n");
    assert_eq!(spread.len(), 2, "line numbering survives block comments");
    assert_eq!(spread[1].trim(), "let b = 2;");

    // Review round 1, finding 2: nested source files are scanned.
    let nested = scratch_crate(
        "nested",
        &[
            ("lib.rs", "mod a;\n"),
            ("a/mod.rs", "mod deep;\n"),
            ("a/deep.rs", "fn f() {\n    let _m = Mutex::new(0);\n}\n"),
        ],
    );
    let (bad, _, seen) = scan_crate(&nested, &test_only);
    assert_eq!(seen, 3, "lib.rs, a/mod.rs and a/deep.rs are all read");
    assert_eq!(bad.len(), 1, "unguarded Mutex in src/a/deep.rs: {bad:?}");
    assert!(bad[0].contains("deep.rs:2:"), "{bad:?}");

    // A guarded `mod a;` guards the whole subtree; an unrelated nested file of the same stem
    // as a guarded top-level module is not exempted by it.
    let guarded = scratch_crate(
        "nested-guarded",
        &[
            ("lib.rs", "#[cfg(test)]\nmod a;\n#[cfg(test)]\nmod inv;\nmod b;\n"),
            ("a/mod.rs", "mod deep;\n"),
            ("a/deep.rs", "fn f() { let _m = Mutex::new(0); }\n"),
            ("inv.rs", "fn g() { let _m = Mutex::new(0); }\n"),
            ("b/mod.rs", "mod inv;\n"),
            ("b/inv.rs", "fn h() { let _m = Mutex::new(0); }\n"),
        ],
    );
    let (bad, hits, seen) = scan_crate(&guarded, &test_only);
    assert_eq!(seen, 6);
    assert_eq!(hits, 2, "a/deep.rs and inv.rs are guarded hits");
    assert_eq!(bad.len(), 1, "only b/inv.rs is unguarded: {bad:?}");
    assert!(bad[0].contains("b/inv.rs:1:"), "{bad:?}");

    let lines =
        strip_comments("#[cfg(test)]\nmod invariants;\n#[cfg(feature = \"cuda\")]\nmod device;\n");
    assert_eq!(find_guards(&lines, &test_only).guarded_mods, vec!["invariants".to_string()]);
}

/// Whether `assert_no_trait_objects` panics on a scratch source tree.
fn trait_scan_rejects(name: &str, files: &[(&str, &str)]) -> bool {
    let dir = scratch_crate(name, files);
    std::panic::catch_unwind(|| assert_no_trait_objects(&dir)).is_err()
}

#[test]
fn trait_object_scan_self_check() {
    // Review round 2: the scan walks nested files and matches `dyn` as a token.
    assert!(trait_scan_rejects(
        "dyn-nested",
        &[("lib.rs", "mod a;\n"), ("a/b/c.rs", "type T = Box<dyn Tr>;\n")]
    ));
    assert!(trait_scan_rejects("dyn-ref", &[("lib.rs", "fn f(x: &dyn Tr) {}\n")]));
    assert!(trait_scan_rejects("dyn-newline", &[("lib.rs", "type T = Box<dyn\nTr>;\n")]));
    assert!(trait_scan_rejects("dyn-paren", &[("lib.rs", "type T = Box<dyn(Fn())>;\n")]));
    assert!(trait_scan_rejects("dyn-tab", &[("lib.rs", "type T = Box<dyn\tTr>;\n")]));
    assert!(!trait_scan_rejects(
        "dyn-ident",
        &[("lib.rs", "let dynamic = 1; let my_dyn = 2; let dyn_x = 3; fn dyna() {}\n")]
    ));
    assert!(!trait_scan_rejects("dyn-comment", &[("lib.rs", "// Box<dyn Tr>\n/* dyn Tr */\n")]));
    assert!(!trait_scan_rejects("dyn-clean", &[("lib.rs", "mod a;\n"), ("a/b.rs", "fn f() {}\n")]));
    // No sources at all must fail loudly rather than pass vacuously.
    assert!(trait_scan_rejects("dyn-empty", &[("notes.txt", "dyn Tr\n")]));
}

#[test]
fn release_cpu_builds_without_counters() {
    if inner_run() {
        return;
    }
    let root = workspace_root();

    // Source scan first: cpu keeps every Mutex and atomic behind `cfg(test)`; metal and cuda
    // behind `cfg(any(test, feature = "debug-counters"))` (or `cfg(test)`).
    let cpu_allowed = vec![squash(CPU_GUARD)];
    let accel_allowed =
        allowed_guards(&["#[cfg(test)]", "#[cfg(any(test, feature = \"debug-counters\"))]"]);
    let mut violations = Vec::new();
    let mut vacuous = Vec::new();
    let mut empty = Vec::new();
    for (crate_dir, allowed, label) in [
        ("cpu", &cpu_allowed, "cfg(test)"),
        ("metal", &accel_allowed, "cfg(any(test, feature = \"debug-counters\"))"),
        ("cuda", &accel_allowed, "cfg(any(test, feature = \"debug-counters\"))"),
    ] {
        let src = root.join("crates").join(crate_dir).join("src");
        let (bad, guarded, files) = scan_crate(&src, allowed);
        if files == 0 {
            empty.push(crate_dir);
        }
        if guarded == 0 {
            vacuous.push(crate_dir);
        }
        violations.extend(bad.into_iter().map(|v| format!("[{crate_dir}, needs {label}] {v}")));
    }
    assert!(violations.is_empty(), "Mutex/atomic outside their guard:\n{}", violations.join("\n"));
    assert!(empty.is_empty(), "the scan read no source files in {empty:?} (vacuous)");
    assert!(vacuous.is_empty(), "the scan saw no guarded counters in {vacuous:?} (vacuous)");

    // The release build itself.
    let output = cargo("audit-release")
        .args(["build", "--release", "-p", "linalg-cpu"])
        .output()
        .expect("spawn cargo build");
    assert!(
        output.status.success(),
        "`cargo build --release -p linalg-cpu` failed ({}):\n{}",
        output.status,
        tail(&output.stderr)
    );
}

// ---------------------------------------------------------------- heterogeneous

/// Shared checks for a heterogeneous 2-rank run against single-rank CPU: y and every dW within
/// 1e-3, the CPU rank on the CPU path, and the accelerator rank on exactly `names`.
fn assert_heterogeneous(
    ranks: &[RankResult],
    order: &[usize],
    names: [&str; 3],
    forbidden: &[&str],
) {
    const TOL: f32 = 1e-3;
    let m = order.len();
    assert_eq!(ranks.len(), 2);
    let (sy, sdw) = gather(std::slice::from_ref(&run_single(order)));
    let (y, dw) = gather(ranks);
    assert_full_shape(&y, &dw, m);
    assert_close(&y, &sy, TOL, "y");
    assert_close(&dw, &sdw, TOL, "dW");

    // Rank 0: the CPU path only.
    for call in &ranks[0].calls {
        assert_eq!(call.kind, DeviceKind::Cpu, "rank 0 call {call:?}");
        assert!(call.dtypes.iter().all(|d| *d == DType::F32), "rank 0 call {call:?} dtype");
    }

    // Rank 1: only the accelerator's three launch names, in the expected counts.
    let accel = &ranks[1];
    let l = accel.owned_layers;
    assert_eq!(l, 1, "placement(3, 2) gives the last rank one layer");
    let count = |name: &str| accel.launches.iter().filter(|e| **e == name).count();
    for name in names {
        assert!(
            accel.launches.contains(&name),
            "rank 1 never launched {name}: {:?}",
            accel.launches
        );
    }
    for launch in &accel.launches {
        assert!(names.contains(launch), "rank 1 launched {launch}, outside {names:?}");
        assert!(!forbidden.contains(launch), "rank 1 launched forbidden {launch}");
    }
    assert_eq!(count(names[0]), 3 * l * m, "rank 1 {}", names[0]);
    assert_eq!(count(names[1]), l * m, "rank 1 {}", names[1]);
    assert_eq!(count(names[2]), l * m, "rank 1 {}", names[2]);
    assert_eq!(accel.launches.len(), 5 * l * m, "rank 1 launches in total");

    // Downloads equal sends: the only bytes the accelerator rank reads back are its boundary.
    assert_eq!(accel.sent.len(), m, "rank 1 sends one dx per microbatch");
    assert_eq!(accel.downloads, accel.sent.len() as u64, "rank 1 downloads = sends");
}

#[cfg(target_os = "macos")]
#[test]
fn heterogeneous_cpu_metal_matches_within_1e_3() {
    let order = [0, 1, 1, 0];
    let ranks = run_cpu_then(
        || linalg_metal::MetalBackend::new().expect("a Metal device on this Mac"),
        &order,
    );
    assert_heterogeneous(
        &ranks,
        &order,
        ["gemm-f32-metal", "gelu-metal", "gelu-backward-metal"],
        &["gemm-f32", "contract_general_f16"],
    );
}

#[test]
fn heterogeneous_cpu_cuda_matches_within_1e_3() {
    let expected = std::env::var("LINALG_AUDIT_EXPECT_CUDA").is_ok_and(|v| v == "1");
    match CudaBackend::new() {
        Ok(_) => {}
        Err(e) if expected => {
            panic!("LINALG_AUDIT_EXPECT_CUDA=1 but CudaBackend::new() failed: {e:?}")
        }
        Err(e) => {
            eprintln!("not run: no usable CUDA backend here ({e:?}); set LINALG_AUDIT_EXPECT_CUDA=1 to require one");
            return;
        }
    }
    let order = [0, 1, 1, 0];
    let ranks = run_cpu_then(|| CudaBackend::new().expect("cuda backend"), &order);
    assert_heterogeneous(
        &ranks,
        &order,
        ["gemm-f32-cublas", "gelu-cuda", "gelu-backward-cuda"],
        &["gemm-f32", "contract_general_f16"],
    );
}
