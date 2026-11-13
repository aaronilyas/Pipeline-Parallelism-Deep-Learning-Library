# linalg-metal measurements

Commit db9e02c, host `macmini` (Apple M2, 10-core GPU, macOS 27.2), 2026-11-13.
Command: `cargo test --release -p linalg-metal -- --nocapture --test-threads=1`

| test | result | bound | verdict |
|---|---|---|---|
| `boundary_convert_and_transfer_take_at_most_2_ms`: convert 256x4096 f32->bf16 + `download_into` | 0.305 ms | 2 ms | pass |
| same test: `upload_into` + convert bf16->f32 (+ sync) | 0.261 ms | 2 ms | pass |
| `twelve_layer_forward_without_waits_is_at_most_0_85x_and_bitwise_equal`: enqueue-all / wait-each | 65.442 / 71.890 ms = 0.910 | 0.85 | **fail** (bits equal) |
| `gemm_1024_cubed_is_within_1_30x_of_legacy` | 0.004 | 1.30 | pass, see note |

Why the 12-layer forward misses 0.85: each wait costs about 0.18 ms (36 waits = 6.4 ms), and
the enqueue-all path already runs at the GPU's speed. The 12 gemms alone take 61.6 ms enqueued
(about 5.1 ms each, 1.7 TFLOP/s, legacy `contract_tiled_f32`). To reach 0.85 with 36 waits, the
GPU work would need to drop to about 36 ms, so about 2.9 ms per gemm. That takes a faster gemm
kernel; this task may not change the legacy kernel.

Note on the 1.30 test: `gemm` no longer waits, so that unchanged test now times the enqueue
(0.013 ms) against the legacy waiting gemm (3.260 ms). It passes, but it no longer compares GPU
work.
