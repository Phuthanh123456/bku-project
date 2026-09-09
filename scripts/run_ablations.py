"""Lập kế hoạch hoặc chạy toàn bộ baseline/ablation với cùng ngân sách.

Mặc định chỉ ghi ``results/ke_hoach_ablation.csv`` và in lệnh. Thêm
``--execute`` trên máy có CUDA để huấn luyện tuần tự, rồi tự chấm tst2013.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))

from nmt.utils import nap_config


THI_NGHIEM = {
    "improved": "configs/baseline_cai_tien_6000.yaml",
    "vanilla": "configs/baseline_vanilla_architecture_6000.yaml",
    "a1": "configs/ablation_a1_layernorm.yaml",
    "a2": "configs/ablation_a2_warmup.yaml",
    "a3": "configs/ablation_a3_label_smoothing.yaml",
    "a4": "configs/ablation_a4_sincos.yaml",
    "a5": "configs/ablation_a5_relu.yaml",
    "a6": "configs/ablation_a6_post_norm.yaml",
}


def _checkpoint(cfg, seed: int, tu_dau: bool = False) -> Path:
    ten_chay = f"{cfg.thi_nghiem.ten}_seed{seed}"
    if tu_dau:
        ten_chay += "_tu_dau"
    return GOC / cfg.thi_nghiem.thu_muc_checkpoint / ten_chay / "tot_nhat.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Thực sự train và evaluate")
    parser.add_argument("--only", nargs="+", choices=THI_NGHIEM, default=list(THI_NGHIEM))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337])
    parser.add_argument("--resume", action="store_true", help="Truyền --tiep-tuc cho train.py")
    parser.add_argument("--repo-hub", default=None)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Cho phép train CPU (rất chậm; mặc định bị chặn)")
    parser.add_argument("--rerun", action="store_true", help="Chạy lại dù checkpoint đã có")
    args = parser.parse_args()

    if args.execute and not args.allow_cpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit(
                "Không có CUDA. Dừng trước khi vô tình chạy 16 lượt 48M tham số trên CPU.\n"
                "Hãy chạy script này trên Kaggle T4; --allow-cpu chỉ dành cho kiểm thử có chủ ý."
            )

    rows = []
    for ma in args.only:
        config_path = GOC / THI_NGHIEM[ma]
        cfg = nap_config(config_path)
        if cfg.huan_luyen.so_buoc_toi_da != 6000:
            raise SystemExit(f"{config_path} không khóa đúng 6000 bước")

        for seed in args.seeds:
            checkpoint = _checkpoint(cfg, seed)
            # Checkpoint seed 42 đã được Bảo public dưới tên lượt chạy cũ.
            if ma == "improved" and seed == 42:
                checkpoint_cu = GOC / "checkpoints/iwslt_base_v1_seed42/tot_nhat.pt"
                if checkpoint_cu.exists():
                    checkpoint = checkpoint_cu

            train_cmd = [
                "python", "scripts/train.py", "--config", THI_NGHIEM[ma],
                "--seed", str(seed), "--so-buoc", "6000",
            ]
            if args.resume:
                train_cmd.append("--tiep-tuc")
            elif args.rerun:
                train_cmd.append("--tu-dau")
            if args.repo_hub:
                train_cmd.extend(["--repo-hub", args.repo_hub])

            checkpoint_relative = checkpoint.relative_to(GOC)
            eval_cmd = [
                "python", "scripts/evaluate.py", "--config", THI_NGHIEM[ma],
                "--seed", str(seed), "--checkpoint", str(checkpoint_relative),
                "--split", "test", "--search", "greedy", "--kv-cache",
            ]
            status = "đã có checkpoint" if checkpoint.exists() else "chưa train"
            rows.append({
                "mã": ma,
                "cấu hình": THI_NGHIEM[ma],
                "seed": seed,
                "số bước": 6000,
                "checkpoint": str(checkpoint_relative),
                "trạng thái": status,
                "lệnh train": subprocess.list2cmdline(train_cmd),
                "lệnh evaluate": subprocess.list2cmdline(eval_cmd),
            })

            print(f"[{ma}/seed{seed}] {status}")
            if not args.execute:
                print("  " + subprocess.list2cmdline(train_cmd))
                continue
            if not checkpoint.exists() or args.rerun:
                subprocess.run(train_cmd, cwd=GOC, check=True)
                checkpoint = _checkpoint(cfg, seed, tu_dau=args.rerun)
                if not checkpoint.exists():
                    raise SystemExit(f"Train xong nhưng không thấy {checkpoint}")
                eval_cmd[eval_cmd.index("--checkpoint") + 1] = str(checkpoint.relative_to(GOC))
            subprocess.run(eval_cmd, cwd=GOC, check=True)

    output = GOC / "results/ke_hoach_ablation.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Đã ghi {len(rows)} lượt vào {output.relative_to(GOC)}")


if __name__ == "__main__":
    main()
