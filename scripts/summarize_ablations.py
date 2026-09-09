"""Tổng hợp điểm thật của các lượt ablation; ô thiếu được giữ trống rõ ràng."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _norm(path: str) -> str:
    return path.replace("\\", "/").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="results/diem_chinh.csv")
    parser.add_argument("--plan", default="results/ke_hoach_ablation.csv")
    parser.add_argument("--output", default="results/tong_hop_diem_ablation.csv")
    args = parser.parse_args()

    plan = pd.read_csv(args.plan)
    scores_path = Path(args.scores)
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    score_by_checkpoint = {}
    if not scores.empty:
        for _, row in scores.iterrows():
            if str(row.get("Tập test", "")) == "test" and str(row.get("Search", "greedy")) == "greedy":
                score_by_checkpoint[_norm(str(row["Checkpoint"]))] = row

    detail = []
    for _, run in plan.iterrows():
        key = _norm(str(run["checkpoint"]))
        score = score_by_checkpoint.get(key)
        detail.append({
            "mã": run["mã"],
            "seed": int(run["seed"]),
            "số bước": int(score["Bước"]) if score is not None else int(run["số bước"]),
            "BLEU": float(score["BLEU"]) if score is not None else float("nan"),
            "chrF++": float(score["chrF++"]) if score is not None else float("nan"),
            "chữ ký BLEU": score["Chữ ký BLEU"] if score is not None else "CHƯA CÓ KẾT QUẢ",
            "trạng thái": "đã đánh giá" if score is not None else run["trạng thái"],
        })
    detail_df = pd.DataFrame(detail)
    output = GOC / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(output, index=False, encoding="utf-8-sig")

    summary = detail_df.groupby("mã", sort=False).agg(
        so_seed=("BLEU", "count"),
        bleu_mean=("BLEU", "mean"),
        bleu_std=("BLEU", "std"),
        chrf_mean=("chrF++", "mean"),
    ).reset_index()
    summary.to_csv(
        GOC / "results/tong_hop_ablation_mean_std.csv", index=False, encoding="utf-8-sig"
    )
    print(detail_df.to_string(index=False))
    print("\nTrung bình/độ lệch chuẩn (chỉ từ số đã đo):")
    print(summary.to_string(index=False))
    print(f"\nĐã lưu {output.relative_to(GOC)}")


if __name__ == "__main__":
    main()
