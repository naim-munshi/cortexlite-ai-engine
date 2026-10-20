"""Check that demo evaluation cannot silently regress in CI (stdlib only)."""
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./build/cortexlite")
    args = parser.parse_args()
    for mode in ("exact", "graph"):
        cmd = [args.binary, "evaluate", "--search", mode, "--ef", "32"]
        output = subprocess.check_output(cmd, text=True)
        metrics = {}
        for line in output.splitlines():
            if ": " in line:
                key, value = line.split(": ", 1)
                metrics[key] = value
        assert int(metrics["evaluation_samples"]) == 12
        for key in ("classification_accuracy", "retrieval_top1_label_accuracy"):
            value = float(metrics[key])
            assert value >= 0.90, f"{mode} {key} unexpectedly low: {value}"
        print(f"{mode}: {metrics}")


if __name__ == "__main__":
    main()
