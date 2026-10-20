#!/usr/bin/env python3
"""Verify that a committed raw JSON/CSV benchmark pair agrees, run by run.

Standard-library-only. Checks are value-based, not timestamp- or host-based,
so both Linux and macOS evidence can be independently verified.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


EXPECTED = {
    "fp32_scalar_t1", "fp32_simd_t1", "fp32_scalar_t4",
    "int8_scalar_t1", "graph_ef16", "graph_ef64", "graph_ef128",
}


def check(json_path: Path, csv_path: Path) -> None:
    document = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    runs = document["runs"]
    if document.get("schema_version") != 1 or len(runs) != 35 or len(rows) != 35:
        raise AssertionError("Expected schema v1 with 35 JSON and CSV run records")

    seen = set()
    groups = {}
    for run, row in zip(runs, rows):
        name, iteration = run["test"], int(run["run"])
        if row["test"] != name or int(row["run"]) != iteration:
            raise AssertionError(f"Mismatched test/run identity: {name} #{iteration}")
        if (name, iteration) in seen or name not in EXPECTED or not 1 <= iteration <= 5:
            raise AssertionError(f"Unexpected or repeated run: {name} #{iteration}")
        seen.add((name, iteration))
        groups.setdefault(name, []).append(run["metrics"])
        metric_fields = set(run["metrics"])
        if not metric_fields.issubset(row):
            raise AssertionError(f"CSV missing metric columns for {name}")
        for key, value in run["metrics"].items():
            raw = row[key]
            if isinstance(value, (int, float)):
                actual = float(raw)
                if not math.isfinite(actual) or not math.isclose(
                    float(value), actual, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AssertionError(f"Mismatch in {name} #{iteration}, {key}")
            elif str(value) != raw:
                raise AssertionError(f"Mismatch in {name} #{iteration}, {key}")
    if set(groups) != EXPECTED or any(len(group) != 5 for group in groups.values()):
        raise AssertionError("Expected 5 runs per test configuration")

    print(f"Verified 35 matching JSON/CSV runs: {json_path.name}")
    for name in sorted(groups):
        key = "graph_recall_at_k" if name.startswith("graph_") else "wall_throughput_ops_per_sec"
        observation = statistics.median([item[key] for item in groups[name]])
        print(f"{name}: median {key} = {observation:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    check(args.json, args.csv)


if __name__ == "__main__":
    main()
