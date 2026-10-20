"""Train a small, reproducible PyTorch MLP and export to portable C-readable CLMODEL1.

Install: python3 -m pip install torch
Run:     python3 python/train_model.py --output data/demo_model.clm
"""
import argparse
from pathlib import Path
import torch

CENTERS = [(-2.0, -2.0), (2.0, -2.0), (0.0, 2.0)]


def write_model(path: Path, network: torch.nn.Sequential) -> None:
    first, _, last = network
    assert isinstance(first, torch.nn.Linear) and isinstance(last, torch.nn.Linear)
    arrays = [first.weight, first.bias, last.weight, last.bias]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"CLMODEL1\n{first.in_features} {first.out_features} {last.out_features}\n")
        for param in arrays:
            fh.write(" ".join(f"{v:.9g}" for v in param.detach().cpu().reshape(-1).tolist()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/demo_model.clm"))
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    features, targets = [], []
    for label, center in enumerate(CENTERS):
        values = torch.randn(180, 2) * 0.50 + torch.tensor(center)
        features.append(values)
        targets.append(torch.full((180,), label, dtype=torch.long))
    x = torch.cat(features)
    y = torch.cat(targets)
    model = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    write_model(args.output, model)
    print(f"exported={args.output} training_accuracy={accuracy:.4f} classes=3 embeddings=8")


if __name__ == "__main__":
    main()
