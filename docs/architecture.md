# Architecture

## Purpose

CortexLite is a single native executable that demonstrates how a trained neural model can be run in C and paired with a C++ similarity-search index. The demonstration uses synthetic 2D input coordinates representing three labeled clusters; this controlled dataset is intentionally small enough to inspect and test.

## Boundaries

1. `python/train_model.py` generates a deterministic training dataset and trains a PyTorch `Linear(2,8) -> ReLU -> Linear(8,3)` model. It exports weights without requiring a Python runtime in the compiled binary.
2. `src/c/inference.c` parses those weights into heap-allocated C arrays, computes the hidden vector and returns softmax class probabilities. `src/c/tensor.c` provides the mathematical primitives; C exposes a stable C ABI through `include/cortexlite/*.h`.
3. `src/cpp/vector_index.cpp` accepts C-produced 8D embeddings, inserts unique IDs into its own hash map, selects a bounded number of nearest existing vectors using a custom heap and links neighbors into a bidirectional graph.
4. `src/cpp/main.cpp` loads records, invokes the C inference API, builds the C++ index and returns predictions, nearest-neighbor results and basic performance measurements.

## DSA implementation details

**ID map:** FNV-1a hashing, linear probing and dynamic doubling near 70% load factor. Keys are immutable; deletion/tombstones are not implemented. Expected O(1) lookup, worst-case O(n).

**Top-k min-heap:** Maintains the k highest cosine scores, replacing the root whenever a better candidate arrives. This reduces selection memory to O(k) while exact search still performs O(nd) similarity work.

**Graph:** Every new vector connects to up to M nearest existing vectors. Links are bidirectional; existing nodes are not degree-pruned. A max-priority frontier performs best-first traversal with an expansion budget `ef`. This implementation demonstrates graph traversal and retrieval approximation; it is not HNSW, and its construction is O(n²d).

**Cosine similarity:** returns zero for a zero-length vector; used consistently by exact and graph search. The example embeds points with nonnegative ReLU outputs.

## Explicit non-goals in v0.1

No serving HTTP API, GPU kernel, LLM token generation, persistent index format, production vector database, external cloud infrastructure or UI. Those features would distract from the core C/C++/DSA learning objective.
