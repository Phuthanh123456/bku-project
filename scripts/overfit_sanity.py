"""TASK 10 — Bài test học thuộc 50 câu — CỔNG CHẶN cuối Tuần 2.  Người làm: My.

Lấy 50 cặp câu từ tập train, TẮT dropout, TẮT label smoothing, learning rate 1e-4,
huấn luyện tới khi loss xuống thật thấp, rồi dịch lại đúng 50 câu đó và chấm BLEU.

Yêu cầu: loss dưới 0,05 trong tối đa 500 bước, và BLEU trên 90.

Nếu đường loss chững lại ở mức 2 hay 3 thì CHẮC CHẮN có lỗi, thường là:
    - mask sai chiều
    - quên chia cho căn bậc hai của d_k
    - nhầm trục khi tính softmax
    - lệch một vị trí khi ghép cặp đầu vào và đầu ra của decoder

Nếu loss về 0 mà BLEU vẫn thấp thì lỗi nằm ở khâu giải mã hoặc ghép lại chữ,
không phải ở mô hình.

Sinh ra: results/overfit_loss.png, results/overfit_vi_du.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, luu_config, nap_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè thi_nghiem.seed")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    raise NotImplementedError("TASK 10 — My")


if __name__ == "__main__":
    main()
