"""TASK 13 — Learning rate scheduler.  Người làm: Quân.  [Training • Bắt buộc]

Theo nhận xét 1 của mentor: Warmup KHÔNG còn là thành phần bắt buộc.
Mặc định trong base.yaml là `scheduler: co_dinh`. Ablation A2 (TASK 18) quyết
định có giữ warmup hay không.

Lưu ý cho A2: bài On Layer Normalization in the Transformer Architecture
(arxiv 2002.04745) cho thấy Pre-Norm vốn đã làm giảm nhu cầu warmup. Nên chênh
lệch giữa bật và tắt có thể rất nhỏ. Đó là điều bài báo dự đoán, không phải bug.
Ngược lại, Post-Norm (A6) gần như bắt buộc phải có warmup.
"""

from __future__ import annotations


class WarmupScheduler:
    """Learning rate tăng dần trong so_buoc_warmup bước đầu rồi giảm dần.

    Mục đích của phần tăng dần là tránh làm hỏng ma trận trọng số lúc mới khởi tạo.
    """

    def __init__(self, optimizer, d_model: int, so_buoc_warmup: int = 4000) -> None:
        raise NotImplementedError("TASK 13 — Quân")

    def step(self) -> None:
        raise NotImplementedError("TASK 13 — Quân")

    def state_dict(self) -> dict:
        raise NotImplementedError("TASK 13 — Quân")

    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError("TASK 13 — Quân")


class SchedulerCoDinh:
    """Learning rate giữ nguyên — mặc định, và là nhánh đối chứng của A2."""

    def __init__(self, optimizer, learning_rate: float) -> None:
        raise NotImplementedError("TASK 13 — Quân")

    def step(self) -> None:
        pass

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


def tao_scheduler(cfg, optimizer):
    """Factory đọc `toi_uu.scheduler` từ YAML."""
    kieu = cfg.toi_uu.scheduler
    if kieu == "warmup":
        return WarmupScheduler(optimizer, cfg.mo_hinh.d_model, cfg.toi_uu.so_buoc_warmup)
    if kieu == "co_dinh":
        return SchedulerCoDinh(optimizer, cfg.toi_uu.learning_rate)
    raise ValueError(f"toi_uu.scheduler khong hop le: {kieu} (phai la warmup hoac co_dinh)")
