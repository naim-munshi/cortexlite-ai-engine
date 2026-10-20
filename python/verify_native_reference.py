#!/usr/bin/env python3
"""Independent stdlib-only FP32 reference check for the native C inference CLI.

Unlike verify_parity.py, this does not need PyTorch. It verifies classes and
the four-decimal confidence printed by the native executable using direct
scalar evaluation of the CLMODEL1 weights on every held-out input.
"""
import argparse
import csv
import math
from pathlib import Path
import re
import subprocess


def reference_parameters(model_path):
    words = Path(model_path).read_text(encoding="utf-8").split()
    if len(words) < 4 or words[0] != "CLMODEL1":
        raise ValueError("Invalid CLMODEL1 header")
    i, h, o = map(int, words[1:4])
    if min(i, h, o) <= 0:
        raise ValueError("Invalid dimensions")
    values = [float(w) for w in words[4:]]
    expected = h * i + h + o * h + o
    if len(values) != expected or not all(math.isfinite(v) for v in values):
        raise ValueError("Unexpected or nonfinite model weights")
    cursor = 0

    def take(n):
        nonlocal cursor
        block = values[cursor:cursor + n]
        cursor += n
        return block

    return (i, h, o, take(h * i), take(h), take(o * h), take(o))


def predict(params, xy):
    i, h, o, w1, b1, w2, b2 = params
    if len(xy) != i:
        raise ValueError("Input dimensions do not match the model")
    hidden = [
        max(0.0, sum(w1[r * i + c] * xy[c] for c in range(i)) + b1[r])
        for r in range(h)
    ]
    logits = [
        sum(w2[r * h + c] * hidden[c] for c in range(h)) + b2[r]
        for r in range(o)
    ]
    high = max(logits)
    exp_values = [math.exp(v - high) for v in logits]
    total = sum(exp_values)
    probs = [v / total for v in exp_values]
    label = max(range(o), key=lambda r: probs[r])
    return label, probs[label]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/cortexlite")
    parser.add_argument("--model", default="data/demo_model.clm")
    parser.add_argument("--eval", default="data/eval.csv")
    args = parser.parse_args()
    params = reference_parameters(args.model)
    verified = 0
    with open(args.eval, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            point = [float(row["x"]), float(row["y"])]
            expected_class, expected_confidence = predict(params, point)
            output = subprocess.check_output(
                [args.binary, "demo", "--model", args.model,
                 "--x", row["x"], "--y", row["y"]],
                text=True,
            )
            label_match = re.search(r"^predicted_class: (\d+)$", output, re.MULTILINE)
            conf_match = re.search(r"^confidence: ([0-9.]+)$", output, re.MULTILINE)
            if not label_match or not conf_match:
                raise AssertionError("Missing native prediction fields")
            actual_class = int(label_match.group(1))
            actual_confidence = float(conf_match.group(1))
            if actual_class != expected_class:
                raise AssertionError(f"{point}: class {actual_class} != {expected_class}")
            # Native demo displays only four digits after the decimal point.
            if abs(actual_confidence - expected_confidence) > 1e-4:
                raise AssertionError(
                    f"{point}: confidence {actual_confidence} != {expected_confidence:.7f}"
                )
            verified += 1
    if not verified:
        raise AssertionError("No evaluation rows")
    print(f"Independent Python/native reference parity: {verified} predictions passed")


if __name__ == "__main__":
    main()
