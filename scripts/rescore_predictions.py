"""Chấm lại CSV dự đoán đã lưu mà không chạy inference lần nữa."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--search", choices=["greedy", "beam"], required=True)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--step", type=int, default=6000)
    parser.add_argument("--scores", default="results/diem_chinh.csv")
    parser.add_argument("--expected-sentences", type=int, default=1268)
    args = parser.parse_args()

    import pandas as pd
    from nmt.eval.metrics import cham_bleu, cham_chrf

    predictions = pd.read_csv(args.input)
    if len(predictions) != args.expected_sentences:
        raise SystemExit(
            f"File có {len(predictions)} câu, cần {args.expected_sentences}; "
            "không ghi điểm mẫu vào bảng chính."
        )
    hypotheses = predictions["Dự đoán (VI)"].fillna("").tolist()
    references = predictions["Tham chiếu (VI)"].fillna("").tolist()
    bleu, bleu_sig = cham_bleu(hypotheses, references, tokenize="none")
    chrf, chrf_sig = cham_chrf(hypotheses, references)

    scores_path = Path(args.scores)
    old = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    if not old.empty:
        if "BLEU tokenizer" not in old:
            old["BLEU tokenizer"] = old["Chữ ký BLEU"].str.extract(r"tok:([^|]+)")[0]
        else:
            inferred = old["Chữ ký BLEU"].str.extract(r"tok:([^|]+)")[0]
            old["BLEU tokenizer"] = old["BLEU tokenizer"].fillna(inferred)
        if "Ghi chú" not in old:
            old["Ghi chú"] = ""
        old.loc[old["BLEU tokenizer"].eq("13a"), "Ghi chú"] = (
            "đối chiếu cũ; không dùng làm điểm chính vì input đã tokenized"
        )

    runtime = ""
    if not old.empty and "Search" in old:
        same = old[
            old["Search"].astype(str).eq(args.search)
            & old["Checkpoint"].astype(str).map(lambda x: x.replace("\\", "/")).eq(
                args.checkpoint.replace("\\", "/")
            )
        ]
        if not same.empty:
            runtime = same.iloc[-1].get("Thời gian (giây)", "")

    row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Tập test": args.split,
        "Hướng dịch": "En-Vi",
        "Số câu": len(predictions),
        "BLEU": bleu,
        "chrF++": chrf,
        "Chữ ký BLEU": bleu_sig,
        "Chữ ký chrF++": chrf_sig,
        "Checkpoint": args.checkpoint,
        "Bước": args.step,
        "Chế độ": "that",
        "Search": args.search,
        "Beam size": args.beam_size if args.search == "beam" else 1,
        "Length penalty": args.length_penalty if args.search == "beam" else "",
        "KV cache": True,
        "Thời gian (giây)": runtime,
        "BLEU tokenizer": "none",
        "Ghi chú": f"protocol chính; chấm lại từ {args.input}",
    }])
    combined = pd.concat([old, row], ignore_index=True)
    # Nếu chạy script lại, chỉ giữ dòng canonical mới nhất của đúng checkpoint/search.
    canonical = combined["BLEU tokenizer"].astype(str).eq("none")
    duplicate_keys = ["Checkpoint", "Tập test", "Search", "BLEU tokenizer"]
    keep_old = combined[~canonical]
    keep_canonical = combined[canonical].drop_duplicates(duplicate_keys, keep="last")
    combined = pd.concat([keep_old, keep_canonical], ignore_index=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(scores_path, index=False, encoding="utf-8-sig")
    print(f"BLEU {bleu:.2f} · chrF++ {chrf:.2f} · {bleu_sig}")
    print(f"Đã cập nhật {scores_path}")


if __name__ == "__main__":
    main()
