"""Vẽ so sánh Greedy/Beam từ các dòng điểm canonical tok:none."""

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
    parser.add_argument("--input", default="results/diem_chinh.csv")
    parser.add_argument("--output", default="results/search_comparison.png")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = data[
        data["Tập test"].eq("test")
        & data["BLEU tokenizer"].eq("none")
        & data["Search"].isin(["greedy", "beam"])
    ].drop_duplicates("Search", keep="last").set_index("Search").loc[["greedy", "beam"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    quality = data[["BLEU", "chrF++"]]
    quality.plot.bar(ax=axes[0], rot=0, color=["#2f6fed", "#2a9d8f"])
    axes[0].set(title="Chất lượng trên tst2013", xlabel="", ylabel="Điểm")
    axes[0].legend(loc="lower right")
    latency = data["Thời gian (giây)"] / data["Số câu"] * 1000
    bars = axes[1].bar(data.index, latency, color=["#457b9d", "#e76f51"])
    axes[1].bar_label(bars, fmt="%.0f ms")
    axes[1].set(title="Độ trễ trung bình theo batch", ylabel="ms/câu")
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Đã lưu {output}")


if __name__ == "__main__":
    main()
