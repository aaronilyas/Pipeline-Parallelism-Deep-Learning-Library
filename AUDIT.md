# `linalg` compliance re-audit, 2026-11-12

Re-run of audit `audit/20261112-0658` after the four fix branches. The previous score on `main` @ `f9bafa4` was 33 PASS, 7 PARTIAL, 1 FAIL. This run scores **40 PASS, 1 PARTIAL, 0 FAIL**.

## Audited refs

| Ref | SHA | Status |
|---|---|---|
| Combined tree (this branch before this commit) | `48d5e26` octopus merge of the four fixes onto `f9bafa4` | the tree that was built and run |
| `fix/argmax-20261112` | `9a58568` | ARG3, argmax G7 |
| `fix/contraction-20261112` | `5f29d04` | CON3 (partial), CON4, contraction G7 |
| `fix/permutation-20261112` | `82ca8e5` | PER4 |
| `fix/slicing-20261112` | `af1f050` | SLI1, G6, slicing G7 |
| `main` | `f9bafa4` | previous audit target; not modified |
| `feat/slicing-tests` | was `1bf37ee` | branch deleted |

`src/abi.rs` sha256 is `f5968dd3f653eac5bc6443115ae93f7d56890408fa6328ea18bd138931261da2`, the same as `f9bafa4`. `Cargo.toml`, `src/lib.rs`, and `src/ops/mod.rs` are unchanged. Each fix branch differs from `f9bafa4` only in its own `src/ops/<name>.rs`, `shaders/<name>.metal`, and `tests/<name>_test.rs`.

Remediation prompts stay on `audit/20261112-0658`.

## Host

Apple M2, 8 GB, via `ssh macmini 'zsh -lc "..."'`. Tree synced to `/tmp/linalg-audit-rerun`, target `/tmp/linalg-audit-rerun-target`, `CARGO_BUILD_JOBS=2`.

- `cargo fmt --check` exit 0 (Mac and the Linux worktree).
- `cargo clippy --release --all-targets -- -D warnings` exit 0. The `block v0.1.6` future-incompatibility warning remains.
- Default suite, three times, through `/tmp/linalg-tools/gpu-test`: exit 0 each time, same counts. lib 1, abi 3, argmax 35 passed + 3 ignored, composition 2, contraction 51 passed + 7 ignored, permutation 29 passed + 2 ignored, reciprocal 34 passed + 4 ignored, slicing 16 passed + 1 ignored, sqrt 12 passed + 1 ignored, doctests 0.
- Every `#[ignore]` test, one `flock /tmp/linalg-gpu.lock cargo test --release -- --ignored --nocapture --test-threads=1`: exit 0.

## Overall verdict

**One requirement is still open.** CON3's audited failures (a uniform subnormal GEMM flushed to zero, and `K = 0`, `β = 1` wiping a subnormal C) now pass at the original gate. What remains is a product more than about `2^200` below the largest product of the same contraction: `simdgroup_multiply_accumulate` flushes it, the behavior is documented and pinned, and the audit regression tests were not weakened.

Everything else that this audit scores is PASS, including ARG3, CON4, PER4, SLI1, G5, G6, and G7.

## Requirement table

| ID | Verdict | Severity | Evidence |
|---|---|---|---|
| G1 | PASS | — | Unchanged from `f9bafa4`: crate `linalg`, edition 2021, pinned deps, no `build.rs`, `src/lib.rs` and `src/ops/mod.rs` as before. |
| G2 | PASS | — | `src/abi.rs` hash matches `f9bafa4`. `Cargo.toml`, `lib.rs`, and `ops/mod.rs` match as well. |
| G3 | PASS | — | `git diff --name-only f9bafa4..<fix>` lists only that primitive's three files, for all four fix branches. |
| G4 | PASS | — | `ops::<primitive>` references in `src/ops/*.rs` and `tests/*_test.rs` are self-references, plus `tests/composition_test.rs`, which is the allowed cross-primitive test. |
| G5 | PASS | — | `feat/slicing-tests` is gone (`git branch --list` is empty; it was `1bf37ee`). The combined tree builds, the default suite is green 3/3, and clippy `-D warnings` on all targets exits 0. |
| G6 | PASS | — | Buffers stay bound at offset 0. Params structs are `#[repr(C)]` and passed with `set_bytes`. Shaders compile from `include_str!` through `compute_pipeline` with fast math off. `encode_*` plus a sync wrapper, `Result` on bad metadata, no blit encoders, seeded `ChaCha8Rng`. `encode_concat` passes `ConcatParams` with `set_bytes` at buffer 17 (`src/ops/slicing.rs`). The part table stays a device buffer because it can exceed the 4 KB `set_bytes` limit. |
| G7 | PASS | — | No `todo!`, `unimplemented!`, or `dbg!`. `cargo fmt --check` is clean. Contraction kernels are `contraction_<variant>` and each `kernel` entry point has its own doc comment. The argmax params struct is `ArgmaxParams` in Rust and MSL. The slicing kernel takes `constant ConcatParams&` and reads fields by name. |
| G8 | PASS | — | `tests/composition_test.rs` is unchanged and passed on all three suite runs (2 tests). It still uses only `encode_argmax`, `encode_gemm`, `encode_materialize`, `encode_reciprocal`, `slice`, and `encode_sqrt`. |
| CON1 | PASS | — | Axis grouping and the four-stride merge are unchanged in role. Rank-4 and invalid-metadata tests pass in the default suite. |
| CON2 | PASS | — | `store_result` loads C only when `β` is non-zero (`shaders/contraction.metal`). `K = 0` or `α = 0` with `β = 1` encodes nothing (`Plan::Empty` in `src/ops/contraction.rs`), so C keeps its bits. `β = 0` cases still NaN-fill C. `audit_k0_keeps_subnormal_c` and `audit_k0_keeps_subnormal_c_f16` pass. |
| CON3 | PARTIAL | major | The audited failures pass. `audit_f32_subnormal_results_meet_the_spec_gate` and `audit_k0_keeps_subnormal_c` use the gate `(K+2)·u·\|C\| + u·\|C\| + η/2` against the unrounded f64 reference. The f16 audit gate is the same with `u_out = 2⁻¹¹` (`exact/2048`) plus half the smallest f16 subnormal. `check` in `tests/contraction_test.rs` uses that formula, and f16 sweeps log a nonzero `err/(√K·u)` (shapes 17446.69, Split-K 8075.84). **Still open:** `subnormal_product_beyond_rescale_is_flushed` gets `C = [1e37, 0, 0, 0]` against f64 `[1e37, 0, 0, 1e-40]`. The scale that keeps `1e37` finite leaves `1e-20·1e-20` subnormal inside `simdgroup_multiply_accumulate`, which flushes it. A probe pins the same flush for scalar `fma` and for MMA. Module docs state the limit. The audit tests were not weakened. |
| CON4 | PASS | — | f16 uses `contraction_tiled_f16` / `contraction_partial_f16` / `contraction_general_f16`: `simdgroup_matrix`, threadgroup staging, f32 accumulation, guarded edges, Split-K summed in order `s = 0..splits`, `β` applied once. `bench_square_f16` on this tree: 1024 f16 1374.8 / 1336.0 vs f32 919.4 / 712.2; 2048 f16 1591.1 / 1577.4 vs f32 1487.1 / 1481.4; 4096 f16 1663.1 / 1661.5 vs f32 1512.7 / 1511.7 GFLOP/s (best / median). The previous f16 figure was 4.6 GFLOP/s at 2048. |
| CON5 | PASS | — | The typed suite still covers both element types and passed inside the 51 contraction tests. The gate is the CON3 formula, so the CON3 caveat applies to the wide-range flush only. |
| CON6 | PASS | — | `bench_square_f32_vs_mps`, medians: 1024 = 1277.7 vs 2071.2 (60.6%), 2048 = 1472.6 vs 2517.8 (56.6%), **4096 = 1514.6 vs 2615.3 (57.6%)**, target ≥ 50%. |
| PER1 | PASS | — | View path is unchanged. Its tests passed in the 29 permutation tests. |
| PER2 | PASS | — | Materialize tests still compare against a Rust gather. They passed. |
| PER3 | PASS | — | Conjugation is still a sign-bit XOR. Adjoint tests passed. |
| PER4 | PASS | — | `fastest_axis` picks the smallest non-zero source stride, last axis on a tie, never a broadcast. The `[32][33]` tile runs when that axis is not last, and the kernel loads it at `src_stride_a`. Otherwise the coalesced copy runs. Stride-2 transpose on this tree: 55.68 GB/s, 98.2% of the stride-2 copy (56.68 GB/s). Before the fix that case was 15.66 GB/s. |
| PER5 | PASS | — | The factorial permutation tests passed in the default suite. |
| PER6 | PASS | — | `bench_transpose_8191x4099_vs_copy`: copy 89.97 GB/s, transpose 88.40 GB/s = **98.3%** (target ≥ 70%). |
| SLI1 | PASS | — | `slice` documents and implements the rule: a non-empty offset is `offset + Σ start·stride`; an in-bounds empty range returns `Ok`; an empty view's offset is clamped to capacity when the exact sum would not be a valid empty view. `audit_empty_slice_at_extent_end_is_legal` passed. Slicing is 16 passed + 1 ignored. |
| SLI2 | PASS | — | Pack still copies integer words and rejects overlap. Those tests passed. |
| SLI3 | PASS | — | Quadrant round-trip passed in the default suite. |
| SLI4 | PASS | — | Pack and concat dispatch shape is unchanged aside from `ConcatParams` moving to `set_bytes`. Tests passed. `encode_concat` documents that `cmd` must retain its resources. |
| SLI5 | PASS | — | `pack_sweep` and the special-bit-pattern tests passed. |
| SLI6 | PASS | — | `bench_strided_pack_vs_copy`: pack 87.91 GB/s, copy 88.47 GB/s = **99.4%** (target ≥ 80%). |
| REC1 | PASS | — | `sweep_real_exhaustive`: all 4,294,967,296 inputs at 0 ULP, 9.6 s. |
| REC2 | PASS | — | Real special and subnormal tests passed in the default suite. The exhaustive sweep is 0 ULP. |
| REC3 | PASS | — | `sweep_complex_random`: worst normwise error 2.4927u, 0 violations of the 4u gate, across five regimes of 2^28 inputs each. |
| REC4 | PASS | — | `compile_options` still turns fast math off. Unchanged file. The exhaustive sweep is the check. |
| REC5 | PASS | — | Batched diagonal, in-place, and overlap tests passed. |
| SQR1 | PASS | — | `sweep_nonnegative_patterns`: all 2,147,483,648 patterns at 0 ULP, 3.4 s. |
| SQR2 | PASS | — | Domain specials and the subnormal test passed in the default suite. |
| SQR3 | PASS | — | Fast math stays off. Unchanged file. |
| SQR4 | PASS | — | Stride, in-place, and overlap tests passed. |
| ARG1 | PASS | — | Empty and oversized inputs still error. Those tests passed (argmax 35 passed). |
| ARG2 | PASS | — | Tie, NaN, and ±0 tests passed. The comparator was not changed. |
| ARG3 | PASS | — | `audit_hypot_pick_is_within_one_ulp_of_f64_max_on_near_ties` passed unchanged apart from rustfmt. `sweep_hypot_key_bits`: 21,440,599 inputs, 9,440,599 exact halfway ties, kernel = host reference = 128-bit oracle, 0 mismatches, 107.4 s. Hypot at 1e7 is 86.3 GB/s next to L1 at 89.8 GB/s on the same run. |
| ARG4 | PASS | — | Reduction structure is unchanged. `results_are_bitwise_deterministic` passed, and the three suite runs matched. |
| ARG5 | PASS | — | Tie, NaN, boundary, and 1e7 tests passed inside the 35 argmax tests. |
| ARG6 | PASS | — | `bench_latency_and_bandwidth`, median GPU time: 3.2 µs at N = 64, 4.8 µs at 1024, 10.2 µs at 1e5, 455.9 µs at 1e7. Bandwidth at 1e7 is **87.7 GB/s**. Complex at 1e7: L1 89.8 GB/s, Hypot 86.3 GB/s. |

## Ignored tests

All exited 0 under `flock /tmp/linalg-gpu.lock`.

| Test | Result |
|---|---|
| contraction `bench_square_f32_vs_mps` | 4096: 57.6% of MPS (median) |
| contraction `bench_square_f16` | 1024 / 2048 / 4096: 1374.8 / 1591.1 / 1663.1 GFLOP/s best |
| contraction `bench_other_paths` | f16 square 2048: 1652.4 GFLOP/s; f32 square 512: 440.3; transposed 2048: 983.2; batched 4096×32³: 121.7; dot K = 16M: 0.5; 32×32 K = 1M: 131.8; rank-1 4096²: 17.1; rank-4 48⁶: 1453.8 |
| contraction `sweep_shapes_f32` / `_f16` | 2016 cases; worst err/(√K·u) 20.56 / 17446.69; worst gate ratio 0.447 / 0.999 |
| contraction `sweep_split_k_f32` / `_f16` | 100 cases; worst 20.70 / 8075.84; gate ratio 0.001 / 0.480 |
| permutation `bench_transpose_8191x4099_vs_copy` | 98.3% of copy |
| permutation `bench_transpose_stride2_vs_copy` | 98.2% of the stride-2 copy |
| slicing `bench_strided_pack_vs_copy` | 99.4% of copy |
| reciprocal `sweep_real_exhaustive` | 2³² values, all 0 ULP, 9.6 s |
| reciprocal `sweep_complex_random` | worst normwise 2.4927u, 0 gate violations |
| reciprocal `bench_real_throughput` / `bench_complex_throughput` | 566.9 / 288.3 M elements/s at 2²⁴ |
| sqrt `sweep_nonnegative_patterns` | 2³¹ values, all 0 ULP, 3.4 s |
| argmax `bench_latency_and_bandwidth` / `bench_complex_and_strided_1e7` | see ARG6 |
| argmax `sweep_hypot_key_bits` | 21,440,599 inputs, 0 mismatches |

## CON3 data for the spec owner

`subnormal_product_beyond_rescale_is_flushed` is a 2×2 product. One entry is `1e18·1e19` (about `1e37`). The other is `1e-20·1e-20` (`1e-40`, a representable f32 subnormal). The kernel's power-of-two rescale is limited by the large product, so the small product stays subnormal in `simdgroup_multiply_accumulate` and becomes 0. The same inputs flush in a scalar `fma` probe and in an MMA probe. f16 is outside this case: its products sit at or above `2⁻⁴⁸`.

Uniform subnormal inputs, `β·C`, and an epilogue whose operands are normal meet the gate. Those are what the original CON3 failure measured.
