# Implementation roadmap

The architecture and language ownership remain fixed across all stages.

| Phase | Scope | Exit condition | Status |
|---|---|---|---|
| 1 | C11/C++17/CMake foundation | Native build, CI and tests | Implemented |
| 2 | Core C++ DSA | Custom hash map, top-k heap, graph traversal | Implemented |
| 3 | C tensor/MLP inference | Load weights, run ReLU/softmax predictions | Implemented |
| 4 | Python/PyTorch model handoff | Train/export, validate numerical parity | Implemented |
| 5 | Vector search | Exact and educational approximate graph modes | Implemented |
| 6 | Performance optimization | SSE2/NEON optional kernels, read-only C++ multithreaded batches, per-row INT8 weights, numerical regression tests | Implemented |
| 7 | Portfolio-grade validation | Seeded 1,000x32 vector study, five-run repetitions, recall/latency evidence, ASan/UBSan, cross-platform CI | Implemented for educational scope |

## Project completion criteria

- All build/test workflows pass on the target developer environment.
- Prediction results match PyTorch reference within explicit numerical tolerance.
- Exact k-nearest searches match brute-force expected ordering and class-label retrieval measurements are reproducible.
- Approximate graph search has a documented recall/latency trade-off against exact top-k on a meaningful vector collection.
- Implement and compare at least one CPU computation optimization (SIMD or multithreading) and one weight/memory optimization (e.g., INT8) without silently losing model accuracy.
- Include reproducible commands, benchmark machine specifications and honest limitations.

All initial educational portfolio completion criteria are implemented in v1.0. This is **not** a production-grade AI serving engine or vector database; see measured trade-offs and scaling limitations in docs/results.md.
