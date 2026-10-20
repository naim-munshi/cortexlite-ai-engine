# Benchmark protocol

The built-in `benchmark` command runs C MLP inference, then C++ vector search, reporting timing distributions. Evaluation uses held-out synthetic points (`data/eval.csv`) and a separately indexed reference set (`data/samples.csv`).

```bash
./build/cortexlite benchmark --iterations 5000 --search exact
./build/cortexlite benchmark --iterations 5000 --search graph --ef 8
./build/cortexlite benchmark --iterations 5000 --search graph --ef 32
```

## Interpret results correctly

- End-to-end operation latency includes inference and search in the timed region but not CSV parsing or index construction.
- `summed_operation_throughput_per_sec` is calculated from the **sum of operation durations**, not a wall-clock throughput measurement of a deployed service. Do not present it as requests per second under production load.
- `classification_accuracy` and `retrieval_top1_label_accuracy` are measured only on a 12-sample synthetic evaluation set. They are not representative of real-world accuracy or retrieval recall.
- `--ef` limits graph node expansions. Compare approximate top-k set overlap (Recall@K) against exact search when larger test data becomes available.
- Optimization phases must be compared with pinned compiler flags, CPU details, dataset, search mode, repetitions, and model checkpoint. Collect at least 5 independent runs and median/p95 variability.

## Suggested results table for future measured runs

| Mode | Dataset size | p50 (us) | p95 (us) | Recall@K versus exact | Peak RSS |
|---|---:|---:|---:|---:|---:|
| Exact | TBD | TBD | TBD | 1.00 | TBD |
| Graph ef=8 | TBD | TBD | TBD | TBD | TBD |
| Graph ef=32 | TBD | TBD | TBD | TBD | TBD |

No benchmark results should be invented. Fill this table only with repeatable observed measurements.
