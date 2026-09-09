"""Vẽ benchmark KV-cache từ results/benchmark_latency.csv."""

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
    parser.add_argument("--input", default="results/benchmark_latency.csv")
    parser.add_argument("--output", default="results/latency_comparison.png")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    labels = data["KV cache"].map({False: "Không cache", True: "KV cache"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    data.set_index(labels)[["p50 (ms/câu)", "p95 (ms/câu)"]].plot.bar(
        ax=axes[0], rot=0, color=["#457b9d", "#e76f51"]
    )
    axes[0].set(title="Độ trễ 200 câu CPU", xlabel="", ylabel="ms/câu")
    bars = axes[1].bar(labels, data["câu/giây"], color=["#adb5bd", "#2a9d8f"])
    axes[1].bar_label(bars, fmt="%.2f")
    axes[1].set(title="Throughput", ylabel="câu/giây")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Đã lưu {output}")


if __name__ == "__main__":
    main()
