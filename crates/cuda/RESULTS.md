# linalg-cuda measurements

Commit db9e02c, 2026-11-13.
Command: `LINALG_AUDIT_EXPECT_CUDA=1 cargo test --release -p linalg-cuda --features cuda -- --nocapture`

| host | GPU, driver | convert 256x4096 f32->bf16 + `download_into` | `upload_into` + convert bf16->f32 (+ sync) | bound |
|---|---|---|---|---|
| `ubuntu` (Linux, CUDA 12.6 user-space) | GTX 1080 sm_61, 580.178.04 | 0.444 ms | 0.430 ms | 2 ms |
| `windows` via WSL2 Ubuntu-26.04 (CUDA 12.6 user-space) | RTX 3070 Ti Laptop sm_86, 621.14 | 0.406 ms | 0.479 ms | 2 ms |

Each is the median of 20 timed rounds after 5 warm-ups (`boundary_convert_and_transfer_take_at_most_2_ms`).

Native Windows was not run. Smart App Control (enforce mode, kept on by the owner) blocks
every cargo build-script binary (os error 4551, Code Integrity event 3077), including cudarc's
and zerocopy's. The Windows-only DLL search path (`%CUDA_PATH%\bin`) passes
`cargo clippy --target x86_64-pc-windows-msvc -p linalg-cuda --features cuda --all-targets -- -D warnings`
on the hub. The CUDA 12.6 cuBLAS 12.6.4.1 and NVRTC 12.6.85 redistributables (SHA-256 checked) are
installed at `C:\Users\aaron\cuda-12.6\bin`, with user `CUDA_PATH` set to that root and `PATH`
left as it was.
