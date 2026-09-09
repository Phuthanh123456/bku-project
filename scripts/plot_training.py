"""Vẽ loss/throughput từ metrics.csv do train.py sinh ra."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/logs/iwslt_base_v1_seed42/metrics.csv")
    parser.add_argument("--output", default="results/training_curves.png")
    args = parser.parse_args()

    data = pd.read_csv(args.input).dropna(subset=["buoc", "loss_train"])
    data["loss_smooth"] = data["loss_train"].rolling(10, min_periods=1).mean()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(data["buoc"], data["loss_train"], alpha=0.25, label="loss mỗi 50 bước")
    axes[0].plot(data["buoc"], data["loss_smooth"], linewidth=2, label="trung bình trượt 10 điểm")
    axes[0].set(xlabel="Bước optimizer", ylabel="Cross-entropy", title="Loss huấn luyện")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    axes[1].plot(data["buoc"], data["token_moi_giay"], color="#e76f51")
    axes[1].set(xlabel="Bước optimizer", ylabel="Token/giây", title="Throughput huấn luyện")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Đã lưu {output}")


if __name__ == "__main__":
    main()
