"""TASK 12 — Cơ chế lưu checkpoint.  Người làm: Bảo.  [Training Infra • Bắt buộc]

Xong khi: thời gian đồng bộ checkpoint chiếm dưới 5 phần trăm tổng thời gian huấn luyện.

Lưu ĐẦY ĐỦ, thiếu một món là resume sai mà không báo lỗi:
    - trọng số mô hình
    - trạng thái optimizer   (thiếu cái này thì momentum của AdamW về 0, loss giật lên)
    - trạng thái scheduler
    - trạng thái GradScaler  (fp16)
    - số bước đã chạy, số epoch
    - bản sao cấu hình đã gộp
    - seed và trạng thái RNG (để resume ra đúng cùng một đường loss)

GHI FILE THEO KIỂU AN TOÀN: ghi ra file tạm rồi mới đổi tên (atomic rename).
Phiên Kaggle bị ngắt đúng lúc đang ghi mà ghi đè trực tiếp thì mất luôn cả
checkpoint cũ lẫn mới.
"""

from __future__ import annotations

from pathlib import Path


def luu_checkpoint(duong_dan: str | Path, model, optimizer, scheduler, scaler, buoc: int, epoch: int, cfg) -> None:
    raise NotImplementedError("TASK 12 — Bảo")


def nap_checkpoint(duong_dan: str | Path, model, optimizer=None, scheduler=None, scaler=None) -> dict:
    """Returns: dict có ít nhất các khóa buoc và epoch để chạy tiếp đúng chỗ cũ."""
    raise NotImplementedError("TASK 12 — Bảo")
