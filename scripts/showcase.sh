#!/usr/bin/env bash
# CortexLite: reproducible, CPU-only portfolio walkthrough.
# Run from repository root: bash scripts/showcase.sh
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure

printf '\n=== Native C inference + C++ nearest-neighbor demo ===\n'
./build/cortexlite demo --x 1.8 --y -2.0 --k 3 --search exact
printf '\n=== Evaluation (synthetic held-out examples) ===\n'
./build/cortexlite evaluate
printf '\n=== Fixed-seed C++ exact-vs-graph retrieval ===\n'
./build/cortexlite_vector_benchmark --vectors 1000 --dimensions 32 --queries 100 --k 10 --ef 64 --seed 42

printf '\nEvidence: docs/results.md and reports/benchmarks-linux-container.csv\n'
printf 'This is a teaching benchmark, not production inference throughput.\n'
