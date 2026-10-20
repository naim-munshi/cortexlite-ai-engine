# Measured benchmark results — CortexLite v1.0

For the independent **real macOS ARM64 / Apple Clang 17** evidence collected with the CMake-free native build, see [macOS benchmark results](results-macos.md) and [raw Mac CSV](../reports/benchmarks-macos-arm64-20261020.csv). Cross-platform absolute timing values are **not** controlled hardware comparisons.

These are **observed, not estimated** results of five repetitions on a Linux x86_64 shared container, collected on 2026-10-20 UTC. CPU model was not provided by `platform.processor()`; reported logical CPUs: **5**. Compiler: `c++ (Debian 14.2.0-19) 14.2.0`. OS: Linux 6.18.55. No dedicated CPU pinning or isolated host. Figures are **median of five individual runs**, not peak scores or production SLA.

## Scalar vs SIMD vs threads vs INT8

All configurations use the same 2-input / 8-hidden / 3-output toy MLP, 18-item reference index, 12 evaluation samples, 5,000 operations per run and Release build. p95 is each run's operation-level percentile, then median of those five p95 values.

| Configuration | Median wall throughput (ops/s) | Median operation p95 (us) | Stored model bytes |
|---|---:|---:|---:|
| FP32 scalar, 1 thread | 1,736,958 | 0.481 | 204 |
| FP32 SSE2, 1 thread | 1,606,620 | 0.481 | 204 |
| FP32 scalar, 4 threads | 3,627,834 | 0.761 | 204 |
| INT8 weights, 1 thread | 1,515,203 | 0.501 | 128 |

- Four threads: **2.09x** measured wall throughput relative to one-thread scalar on this host (not guaranteed on another machine).
- SIMD: **0.92x** throughput relative to scalar. This tiny 2x8x3 network does **not** benefit from vectorization; do not advertise a SIMD speedup here.
- INT8 stored weights/scales/biases: **37.3%** fewer model bytes (204 vs 128). The FP32-activation INT8 path measured **0.87x** scalar baseline throughput; memory improvement is *not* equivalent to speed improvement.

## Exact vs graph search — 1000 32D embeddings, k=10, seed 42

| ef | Mean Recall@10 (median of five runs) | Graph p95 (us) | Exact p95 (us) |
|---|---:|---:|---:|
| 16 | 0.793 | 27.42 | 120.60 |
| 64 | 0.952 | 47.13 | 104.28 |
| 128 | 0.987 | 90.38 | 61.17 |

Recall@10 = (number of candidate IDs also in the exact top-10) / 10, averaged over the same 100 deterministic synthetic queries. Larger ef generally improves recall but costs more search time. This implementation's graph build is quadratic and **not intended for large-scale retrieval**.

## Evidence and limitations

- [Per-run JSON](../reports/benchmarks-linux-container.json) contains 35 measurement records with commands and host metadata.
- [Per-run CSV](../reports/benchmarks-linux-container.csv) contains the same raw metrics suitable for plotting and external analysis.
- CPU cache state, OS scheduler, noisy shared-host effects and 18-vector toy workload can distort timing. This experiment is a reproducible engineering demonstration, not external hardware performance evidence.
- Synthetic labels and samples cannot validate deployed model accuracy, ANN robustness, or large-model quantization quality. FP32/INT8 prediction agreement and ASan/UBSan are covered by automated regression tests.
