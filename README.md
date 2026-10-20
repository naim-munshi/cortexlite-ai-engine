# CortexLite

**CortexLite is an educational C/C++ AI inference and vector search engine.** It runs a small PyTorch-trained neural network using a dependency-free C inference core, produces hidden-layer embeddings, and searches them using custom C++ data structures and algorithms.

The purpose of the project is to demonstrate **AI inference, data structures and algorithms (DSA), systems programming, and measurable performance** in one coherent engineering project. It is a CPU reference implementation, not a production LLM engine or a full vector database.

## Language responsibilities

| Language | Responsibilities | Demonstrated skill |
|---|---|---|
| **C11** | Row-major matrix operations, ReLU, stable softmax, cosine similarity, two-layer MLP, explicit memory lifecycle and model loading | Low-level programming and ML execution |
| **C++17** | Open-addressing hash map, custom top-K min-heap, graph adjacency index, exact and approximate nearest-neighbor searches, CLI application | DSA, interfaces, algorithms and integration |
| **Python / PyTorch** | Reproducible model training, portable weight export and independent numerical parity checks | Model development and inference validation |
| **CMake / GitHub Actions** | Builds, regression tests, cross-platform CI | Engineering discipline |

### End-to-end flow

```
Python/PyTorch: train 2D/3-class MLP -> export CLMODEL1 weights
                                         |
                                         v
C11: load weights -> matrix operations -> hidden embedding -> predictions
                                         |
                                         v
C++17: hash IDs -> build vector graph -> exact / graph neighbor search
                                         |
                                         v
                 Unified CLI: demo, evaluate, benchmark
```

## Requirements

- C11 compiler, C++17 compiler, and CMake 3.16+
- Python 3.9+ with PyTorch (only needed to retrain or run the independent parity check)
- Linux or macOS; CPU only

## Quick start

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/cortexlite demo
./build/cortexlite evaluate
./build/cortexlite benchmark --iterations 5000 --search exact
./build/cortexlite benchmark --iterations 5000 --search graph --ef 32
```

Default demo query is `(1.8, -2.0)`. To run another:

```bash
./build/cortexlite demo --x -2.1 --y -1.9 --k 3 --search exact
```

Optional PyTorch training and verification:

```bash
python3 -m pip install -r python/requirements.txt
python3 python/train_model.py --output data/demo_model.clm
python3 python/verify_parity.py --binary ./build/cortexlite
```

If you retrain, rebuild is unnecessary because model weights are loaded from disk.

## What is measured

- **Classification accuracy** on 12 held-out, hand-picked synthetic evaluation points.
- **Top-1 retrieved-label accuracy** against an index of 18 synthetic reference embeddings; this is *not* Recall@K over a real corpus.
- **p50/p95 end-to-end operation latency**, plus timed-section throughput for the MLP inference and search call.
- **Exact versus neighbor-graph search** under the same model and dataset.

The checked-in model and dataset are a small deterministic teaching example. Their accuracy and microsecond-scale latency must **not** be extrapolated to real industrial systems. Collect larger datasets, multiple runs and realistic queries for a portfolio-grade performance study.

## Repository structure

```
include/cortexlite/
  tensor.h             C tensor/math API
  inference.h          C model API
  vector_index.hpp     C++ DSA/vector index API
src/c/
  tensor.c             Matrix-vector, ReLU, softmax, cosine
  inference.c          Model loader, two-layer MLP inference
src/cpp/
  vector_index.cpp     Hash map, top-K heap, graph index and search
  main.cpp             CLI: demo/evaluate/benchmark
data/
  demo_model.clm       Exported example model weights
  samples.csv          Reference vectors (via C hidden activations)
  eval.csv             Separate evaluation inputs
python/
  train_model.py       PyTorch training and model export
  verify_parity.py     PyTorch vs C inference consistency
  requirements.txt
tests/
  test_core.cpp        C tensor, MLP, DSA, and graph tests
 docs/
  architecture.md
  roadmap.md
  benchmarks.md
  model_format.md
.github/workflows/
  ci.yml
CMakeLists.txt
```

## Algorithms

- **Open-addressing hash map:** average O(1) lookup/insert; worst-case O(n) with collisions.
- **Fixed-capacity min-heap:** O(n log k) exact top-k selection over n vectors, O(k) auxiliary heap memory.
- **Exact cosine search:** O(n d + n log k), where d is embedding dimension.
- **Neighbor graph:** insertion scans prior vectors, O(n² d) total construction; best-first query visits at most `ef` expanded nodes but can discover more than `ef` neighbors. Approximate results are **not guaranteed** to match exact search.
- **MLP inference:** O(i h + h o) arithmetic, where i/h/o are input/hidden/output dimensions.

See [architecture](docs/architecture.md), [roadmap](docs/roadmap.md) and [benchmark guidance](docs/benchmarks.md).

## Limitations

The example is a 2D toy classifier with 8-dimensional hidden embeddings and three output classes. `CLMODEL1` supports general layer dimensions, but the CLI CSV adapter expects two input features. The neighbor graph is a simple educational approximate-search method, **not HNSW**. SIMD, quantized inference, multithreaded execution and production-scale benchmarking are documented future milestones, not implemented features.

## License

MIT. See [LICENSE](LICENSE).
