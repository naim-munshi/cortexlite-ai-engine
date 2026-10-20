# CortexLite benchmark protocol (v1.0)

This benchmark is a **CPU educational microbenchmark**, not a production inference server. Timed operations include MLP inference and vector search over the 18-example toy reference set. CSV loading and index building are outside the latency interval. Multi-thread throughput includes worker start and join in the wall-clock denominator.

## Reproduce

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python3 scripts/run_benchmarks.py --repetitions 5 --iterations 5000 --vectors 1000 --output reports/benchmarks-local.json
```

- Seed: **42**; vector benchmark: **1000 vectors x 32 dimensions**, 100 perturbed queries, k=10, ef in {16, 64, 128}, same queries for exact and graph.
- Repeat every configuration **five times**; raw JSON includes each command, host, compiler, sample and timestamp. CSV contains all individual observations; no numbers are simulated.
- FP32 scalar is the reference. SSE2/NEON SIMD kernels use FP32 vector arithmetic, so expect small floating-point differences. Unsupported architectures reject `--kernel simd`.
- INT8 uses symmetric **per-row weight-only quantization** (`int8` weights; FP32 biases/scales/activations). Weight bytes are reported separately from index and process RSS. FP32 weights are freed after quantization in the executable.
- Thread count 4 uses **parallel independent read-only inference/search operations**. It is not a network server or batch matrix-multiplication kernel. Report measured wall throughput and p50/p95 operation latency; these are distinct statistics.
- Graph search is **not HNSW** and graph build is O(n² d). Report exact-vs-approximate Recall@K and graph p50/p95 search latency. The synthetic Gaussian dataset is not real-world retrieval quality.
- Benchmark hardware may be noisy or shared. Pin CPU/compiler, model, seed, sample counts and compiler flags. Do not compare absolute microseconds across unpinned hosts.
- Data should be evaluated beyond synthetic points before any real-world accuracy claim.

See [measured summary](results.md) and [raw repeat CSV](../reports/benchmarks-linux-container.csv). GitHub Actions also uploads short-retention raw evidence for each successful Linux CI run.
