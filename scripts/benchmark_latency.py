"""Đo p50/p95 và throughput giải mã PyTorch — TASK 20."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, nap_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/iwslt_base_v1_seed42.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sentences", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--include-beam", action="store_true")
    parser.add_argument("--output", default="results/benchmark_latency.csv")
    args = parser.parse_args()
    if args.sentences <= 0 or args.warmup < 0:
        raise SystemExit("--sentences phải dương và --warmup không được âm")

    import numpy as np
    import pandas as pd
    import torch
    from nmt.data import BOS_ID, EOS_ID, PAD_ID, nap_tokenizer
    from nmt.eval.metrics import cham_bleu
    from nmt.inference.search import beam_search, greedy_search
    from nmt.model.masking import tao_padding_mask
    from nmt.model.transformer import TransformerNMT
    from nmt.training.checkpoint import CHE_DO_THAT, nap_checkpoint

    cfg = nap_config(args.config)
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Không thấy checkpoint: {checkpoint}")
    tokenizer = nap_tokenizer(cfg.du_lieu.tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNMT(cfg).to(device)
    info = nap_checkpoint(
        checkpoint, model, map_location=device,
        che_do_mong_doi=CHE_DO_THAT, khoi_phuc_rng=False,
    )
    model.eval()

    source_path = Path(str(cfg.du_lieu.test) + ".en")
    target_path = Path(str(cfg.du_lieu.test) + ".vi")
    sources = source_path.read_text(encoding="utf-8").splitlines()
    raw_refs = target_path.read_text(encoding="utf-8").splitlines()
    count = min(args.sentences, len(sources))
    sources, raw_refs = sources[:count], raw_refs[:count]
    references = [tokenizer.decode(tokenizer.encode(x).ids) for x in raw_refs]

    def translate(text: str, search: str, use_cache: bool):
        ids = tokenizer.encode(text).ids[:cfg.du_lieu.do_dai_toi_da]
        src_ids = torch.tensor([ids], dtype=torch.long, device=device)
        src_mask = tao_padding_mask(src_ids, PAD_ID)
        if search == "beam":
            output = beam_search(
                model, src_ids, src_mask, BOS_ID, EOS_ID,
                beam_size=cfg.sinh_cau.beam_size,
                he_so_phat_do_dai=cfg.sinh_cau.he_so_phat_do_dai,
                do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich,
                dung_kv_cache=use_cache,
            )
        else:
            output = greedy_search(
                model, src_ids, src_mask, BOS_ID, EOS_ID,
                do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich,
                dung_kv_cache=use_cache,
            )
        output_ids = output[0].tolist()
        if EOS_ID in output_ids:
            output_ids = output_ids[:output_ids.index(EOS_ID)]
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    variants = [
        ("PyTorch", "greedy", False),
        ("PyTorch", "greedy", True),
    ]
    if args.include_beam:
        variants.append(("PyTorch", "beam", True))

    rows = []
    for runtime, search, use_cache in variants:
        print(f"Đo {runtime} {search}, KV cache={use_cache}...")
        for text in sources[:args.warmup]:
            translate(text, search, use_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        hypotheses = []
        total_start = perf_counter()
        for index, text in enumerate(sources, start=1):
            start = perf_counter()
            hypotheses.append(translate(text, search, use_cache))
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((perf_counter() - start) * 1000)
            if index % 25 == 0:
                print(f"  {index}/{count}")
        total = perf_counter() - total_start
        bleu, signature = cham_bleu(hypotheses, references, tokenize="none")
        rows.append({
            "runtime": runtime,
            "thiết bị": device.type,
            "search": search,
            "beam size": cfg.sinh_cau.beam_size if search == "beam" else 1,
            "KV cache": use_cache,
            "số câu": count,
            "p50 (ms/câu)": float(np.percentile(latencies, 50)),
            "p95 (ms/câu)": float(np.percentile(latencies, 95)),
            "câu/giây": count / total,
            "dung lượng model (MB)": checkpoint.stat().st_size / 1024**2,
            "BLEU trên mẫu": bleu,
            "chữ ký BLEU": signature,
            "checkpoint bước": info["buoc"],
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Đã lưu {output}")
    print("ONNX và ONNX INT8 chưa được đo; không ghi số giả vào bảng PyTorch.")


if __name__ == "__main__":
    main()
