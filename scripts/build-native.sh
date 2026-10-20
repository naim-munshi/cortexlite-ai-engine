#!/usr/bin/env bash
# Build CortexLite with the system C/C++ compilers; CMake and Homebrew not required.
# Usage: bash scripts/build-native.sh
# Override compilers using CC and CXX, or output directory using CORTEXLITE_NATIVE_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."

CC_BIN="${CC:-cc}"
CXX_BIN="${CXX:-c++}"
OUT="${CORTEXLITE_NATIVE_DIR:-build-native}"

if ! command -v "$CC_BIN" >/dev/null 2>&1; then
  echo "Missing C compiler: $CC_BIN (on macOS: xcode-select --install)" >&2
  exit 1
fi
if ! command -v "$CXX_BIN" >/dev/null 2>&1; then
  echo "Missing C++ compiler: $CXX_BIN (on macOS: xcode-select --install)" >&2
  exit 1
fi

mkdir -p "$OUT"
CFLAGS=(-std=c11 -O2 -Wall -Wextra -Wpedantic -Iinclude)
CXXFLAGS=(-std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude)
"$CC_BIN" "${CFLAGS[@]}" -c src/c/tensor.c -o "$OUT/tensor.o"
"$CC_BIN" "${CFLAGS[@]}" -c src/c/inference.c -o "$OUT/inference.o"
"$CXX_BIN" "${CXXFLAGS[@]}" src/cpp/main.cpp src/cpp/vector_index.cpp \
    "$OUT/tensor.o" "$OUT/inference.o" -pthread -lm -o "$OUT/cortexlite"
"$CXX_BIN" "${CXXFLAGS[@]}" benchmarks/vector_benchmark.cpp src/cpp/vector_index.cpp \
    "$OUT/tensor.o" "$OUT/inference.o" -pthread -lm -o "$OUT/cortexlite_vector_benchmark"
"$CXX_BIN" "${CXXFLAGS[@]}" tests/test_core.cpp src/cpp/vector_index.cpp \
    "$OUT/tensor.o" "$OUT/inference.o" -pthread -lm -o "$OUT/cortexlite_tests"
echo "Native build complete: $OUT (C: $CC_BIN, C++: $CXX_BIN)"
