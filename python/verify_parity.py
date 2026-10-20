"""Validate C native model predictions against independently reconstructed PyTorch model."""
import argparse
import csv
from pathlib import Path
import re
import subprocess
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/cortexlite")
    parser.add_argument("--model", type=Path, default=Path("data/demo_model.clm"))
    parser.add_argument("--eval", type=Path, default=Path("data/eval.csv"))
    args = parser.parse_args()
    content = args.model.read_text().split()
    assert content[0] == "CLMODEL1"
    i, h, o = map(int, content[1:4])
    values = [float(x) for x in content[4:]]
    assert len(values) == h*i + h + o*h + o
    pos = 0
    def take(n):
        nonlocal pos
        arr = values[pos:pos+n]
        pos += n
        return torch.tensor(arr, dtype=torch.float32)
    m = torch.nn.Sequential(torch.nn.Linear(i, h), torch.nn.ReLU(), torch.nn.Linear(h, o))
    with torch.no_grad():
        m[0].weight.copy_(take(h*i).reshape(h, i))
        m[0].bias.copy_(take(h))
        m[2].weight.copy_(take(o*h).reshape(o, h))
        m[2].bias.copy_(take(o))
    count = 0
    with args.eval.open(newline="") as f:
        for r in csv.DictReader(f):
            point = [float(r["x"]), float(r["y"])]
            with torch.no_grad():
                probs = m(torch.tensor(point)).softmax(dim=0)
            expected = int(probs.argmax())
            conf = float(probs[expected])
            output = subprocess.check_output([
                args.binary, "demo", "--model", str(args.model),
                "--x", r["x"], "--y", r["y"]
            ], text=True)
            got_label = int(re.search(r"predicted_class: (\d+)", output).group(1))
            got_conf = float(re.search(r"confidence: ([0-9.]+)", output).group(1))
            assert got_label == expected, (point, got_label, expected)
            assert abs(got_conf - conf) < 1e-3, (point, got_conf, conf)
            count += 1
    print(f"PyTorch/native parity passed: {count} samples (class match, confidence tol 1e-3)")


if __name__ == "__main__":
    main()
