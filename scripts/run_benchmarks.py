#!/usr/bin/env python3
"""Reproducible benchmark collector; never fabricates results."""
import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def run(command, cwd):
    process = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=True)
    metrics = {}
    for line in process.stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            try:
                metrics[key] = float(value) if "." in value else int(value)
            except ValueError:
                metrics[key] = value
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="build/cortexlite")
    parser.add_argument("--vector-binary", default="build/cortexlite_vector_benchmark")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--vectors", type=int, default=1000)
    parser.add_argument("--output", default="reports/benchmarks-local.json")
    args = parser.parse_args()
    if not (1 <= args.repetitions <= 20 and args.iterations >= 100 and 32 <= args.vectors <= 20000):
        parser.error("repetitions 1..20, iterations >=100, vectors 32..20000")
    root = Path(__file__).resolve().parents[1]
    binary = str((root / args.binary).resolve())
    vector_binary = str((root / args.vector_binary).resolve())
    configurations = [
        ("fp32_scalar_t1", "fp32", "scalar", 1),
        ("fp32_simd_t1", "fp32", "simd", 1),
        ("fp32_scalar_t4", "fp32", "scalar", 4),
        ("int8_scalar_t1", "int8", "scalar", 1),
    ]
    machine = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unspecified",
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "compiler": subprocess.run(["c++", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    results = []
    for name, precision, kernel, threads in configurations:
        for iteration in range(args.repetitions):
            command = [binary, "benchmark", "--precision", precision,
                       "--kernel", kernel, "--threads", str(threads),
                       "--iterations", str(args.iterations)]
            entry = {"test": name, "run": iteration + 1,
                     "metrics": run(command, root), "command": command}
            results.append(entry)
    for ef in (16, 64, 128):
        for iteration in range(args.repetitions):
            command = [vector_binary, "--vectors", str(args.vectors), "--dimensions", "32",
                       "--queries", "100", "--k", "10", "--ef", str(ef), "--seed", "42"]
            results.append({"test": f"graph_ef{ef}", "run": iteration + 1,
                            "metrics": run(command, root), "command": command})
    report = {"schema_version": 1, "description": "Local CPU microbenchmarks; not production throughput",
              "host": machine, "configuration": vars(args), "runs": results}
    destination = root / args.output
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = destination.with_suffix(".csv")
    keys = sorted({k for x in results for k in x["metrics"]})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test", "run", *keys])
        writer.writeheader()
        for entry in results:
            writer.writerow({"test": entry["test"], "run": entry["run"], **entry["metrics"]})
    print(f"Saved {len(results)} measured runs: {destination} and {csv_path}")


if __name__ == "__main__":
    main()
