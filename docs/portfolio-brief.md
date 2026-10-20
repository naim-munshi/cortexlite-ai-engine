# CortexLite — technical reviewer brief

> **Scope:** A C11 / C++17 CPU reference for a two-layer PyTorch-trained classifier and vector retrieval. Reproducible educational implementation, not a production LLM runtime, vector database, or deployed web service.

## The 30-second explanation

I implemented a lightweight AI inference and search pipeline to explore where machine-learning math meets low-level systems work. Python trains and exports a 2-input, 8-hidden, 3-output network. C11 owns the weight loader, tensor primitives, FP32/SIMD inference and optional weight-only INT8 inference. C++17 builds a custom ID hash map, top-K heap and neighbor graph over C-produced embeddings, then performs exact or approximate cosine search. The CLI tests predictions and measures latency, recall and throughput.

The project demonstrates **engineering ownership** of low-level AI execution and algorithmic tradeoffs. It is intentionally small enough to inspect, compile and reproduce.

## Review this project in three minutes

| Question | Source of evidence |
|---|---|
| How does inference work? | [C inference implementation](../src/c/inference.c), [tensor operations](../src/c/tensor.c), [model format](model_format.md) |
| Where is the DSA work? | [Custom C++ hash map, top-K min-heap, graph search](../src/cpp/vector_index.cpp), [algorithm complexity](architecture.md) |
| Is Python required at runtime? | No. [Training and export](../python/train_model.py) are separate from the [native CLI](../src/cpp/main.cpp) |
| What are the optimization tradeoffs? | [Measured results](results.md) and [35 raw runs](../reports/benchmarks-linux-container.csv) |
| Do builds/tests pass? | [Linux/macOS and ASan/UBSan CI](../.github/workflows/ci.yml) |
| Can I run it? | `bash scripts/showcase.sh` from the repository root; [script](../scripts/showcase.sh) |

## Three engineering decisions worth discussing

**1. Why separate C from C++?** The C ABI keeps numeric kernels independent of search infrastructure; the C++ layer uses RAII/STL as needed around custom data structures. This boundary makes memory ownership and execution cost visible.

**2. Why a custom heap and graph?** The top-K heap uses O(k) selection storage and O(n log k) heap work over n scanned vectors. The graph offers an approximate search/recall tradeoff; its O(n²d) construction and unbounded node degree limit scalability. It is **not HNSW**.

**3. Why show a slower SIMD/INT8 result?** The sample model is tiny. SSE2 achieved ~0.92× scalar throughput; weight-only INT8 ~0.87×, while shrinking stored model parameters by 37.3%. Presenting negative results avoids conflating an optimization technique with a measured speedup.

## Benchmarks: read the denominator

All data in [results.md](results.md) were measured on a noisy shared Linux x86_64 container, with five repetitions per configuration, 5,000 repeated inference/search operations per run, and the same 18-vector toy index. Four threads measured **2.09× wall throughput versus one thread** on that host. This does not mean 2.09× real-world web service throughput.

The graph study uses 1,000 random 32-dimensional vectors and 100 synthetic queries: Recall@10 is **0.952** at ef=64 relative to exact top-10, not a measure of semantic search relevance. Compare [raw measurements](../reports/benchmarks-linux-container.json) and the [reproduction protocol](benchmarks.md) before using any numbers elsewhere.

## Suggested CV description (conservative)

**CortexLite — C/C++ AI Inference & Vector Search Engine**

Built a CPU-based AI inference and nearest-neighbor search pipeline using **C11, C++17, and Python/PyTorch**, implementing tensor operations, model-weight loading, custom hash tables/top-K heaps/graph traversal, and reproducible testing.

Evaluated **SIMD, weight-only INT8, and 4-thread execution** on a small synthetic workload, observing **2.09× multi-threaded throughput** and **37.3% lower model parameter storage** versus FP32 baselines, with documented environment and limitations.

**Note:** These measurements are from one shared Linux container and a small toy model; they are not deployed production results.

## Reproduction and scope

- [One-command local walkthrough](../scripts/showcase.sh)
- [Build and CLI usage](../README.md#run-it-yourself)
- [Performance methodology](benchmarks.md)
- [Verified macOS ARM64 benchmark and cross-platform context](results-macos.md)
- [Limitations](../README.md#scope-and-limitations)
- [MIT license](../LICENSE)

No hosted demo is advertised. Reviewers can inspect the code, run the CLI locally, or check CI logs without creating accounts, obtaining API keys, or provisioning a GPU.
