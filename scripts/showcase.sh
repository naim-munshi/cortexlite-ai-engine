#!/usr/bin/env bash
# Reproducible CPU-only showcase. Works without Homebrew or CMake on macOS.
# Force the compiler-only build even when CMake exists:
#   CORTEXLITE_FORCE_NATIVE=1 bash scripts/showcase.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${CORTEXLITE_FORCE_NATIVE:-0}" == "1" ]] || ! command -v cmake >/dev/null 2>&1; then
  echo "Using system C/C++ compiler directly (CMake not required)"
  bash scripts/build-native.sh
  bash scripts/test-native.sh
  BUILD_DIR="${CORTEXLITE_NATIVE_DIR:-build-native}"
else
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build --parallel 2
  ctest --test-dir build --output-on-failure
  python3 python/verify_native_reference.py --binary ./build/cortexlite
  BUILD_DIR="build"
fi

printf '\n=== Native C inference + C++ nearest-neighbor demo ===\n'
"./$BUILD_DIR/cortexlite" demo --x 1.8 --y -2.0 --k 3 --search exact
printf '\n=== Evaluation (synthetic held-out examples) ===\n'
"./$BUILD_DIR/cortexlite" evaluate
printf '\n=== Fixed-seed C++ exact-vs-graph retrieval ===\n'
"./$BUILD_DIR/cortexlite_vector_benchmark" --vectors 1000 --dimensions 32 --queries 100 --k 10 --ef 64 --seed 42

printf '\nEvidence: docs/results.md and reports/benchmarks-linux-container.csv\n'
printf 'This is a teaching benchmark, not production inference throughput.\n'
