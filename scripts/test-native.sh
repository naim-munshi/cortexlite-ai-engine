#!/usr/bin/env bash
# Exercise all 7 CTest-equivalent checks using a compiler-only native build.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${CORTEXLITE_NATIVE_DIR:-build-native}"
APP="$OUT/cortexlite"
VECTOR="$OUT/cortexlite_vector_benchmark"
UNIT="$OUT/cortexlite_tests"

for binary in "$APP" "$VECTOR" "$UNIT"; do
  if [[ ! -x "$binary" ]]; then
    echo "Missing $binary. First run: bash scripts/build-native.sh" >&2
    exit 1
  fi
done

pass=0
run_test() {
  printf '\n[%d/7] %s\n' "$((pass + 1))" "$1"
  shift
  "$@"
  pass=$((pass + 1))
}

run_test "Unit tests" "$UNIT" data/demo_model.clm
run_test "FP32 evaluation" "$APP" evaluate
run_test "INT8 evaluation" "$APP" evaluate --precision int8
architecture="$(uname -m)"
case "$architecture" in
  x86_64|amd64|arm64|aarch64)
    run_test "SIMD evaluation" "$APP" evaluate --kernel simd ;;
  *)
    echo "SIMD unavailable on this architecture; verifying explicit unsupported error"
    if "$APP" evaluate --kernel simd; then
      echo "Unsupported SIMD request unexpectedly succeeded" >&2
      exit 1
    fi
    pass=$((pass + 1)) ;;
esac
run_test "Four-thread benchmark" "$APP" benchmark --threads 4 --iterations 128
run_test "Vector-search smoke benchmark" "$VECTOR" --vectors 128 --dimensions 16 --queries 16 --k 5 --ef 32
run_test "Graph-search evaluation" "$APP" evaluate --search graph --ef 32

python3 python/check_metrics.py --binary "$APP"
python3 python/verify_native_reference.py --binary "$APP"
printf '\nAll %d native tests and Python reference checks passed.\n' "$pass"
