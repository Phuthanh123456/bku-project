"""TASK 08 — Feed-Forward: SwiGLU + ReLU.  Người làm: Quân.
TASK 09 — Lớp Encoder / Decoder.         Người làm: Bảo.
[Model Core • Bắt buộc]

TASK 08 xong khi: số tham số hai phương án chênh nhau dưới 1% (bài test 9).
    SwiGLU: 3 x 512 x 688  = 1.056.768
    ReLU:   2 x 512 x 1024 = 1.048.576      -> chênh 0,8 phần trăm

Đặt d_ff sai (giữ 1024 cho SwiGLU) thì mô hình phình thêm khoảng 12 triệu
tham số mà không ai nhận ra.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    Ba ma trận. Nhánh cổng đi qua SiLU rồi nhân TỪNG PHẦN TỬ với nhánh thường,
    sau đó mới qua ma trận thứ ba thu hẹp về d_model.
    SiLU(z) = z * sigmoid(z).  Nguồn: GLU Variants Improve Transformer.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("TASK 08 — Quân")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TASK 08 — Quân")


class FeedForwardReLU(nn.Module):
    """Khối hai ma trận + ReLU của bản 2017 — đối chứng cho ablation A5."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("TASK 08 — Quân")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TASK 08 — Quân")


def tao_ffn(kieu: str, d_model: int, d_ff: int, dropout: float = 0.0) -> nn.Module:
    """Factory đọc `mo_hinh.kieu_ffn` từ YAML."""
    if kieu == "swiglu":
        return SwiGLU(d_model, d_ff, dropout)
    if kieu == "relu":
        return FeedForwardReLU(d_model, d_ff, dropout)
    raise ValueError(f"kieu_ffn không hợp lệ: {kieu} (phải là swiglu hoặc relu)")


class LopEncoder(nn.Module):
    """Một lớp encoder: self-attention (có RoPE) rồi tới feed forward.

    Pre-Norm : x = x + SubLayer(Norm(x))      <- mặc định, bản hiện đại
    Post-Norm: x = Norm(x + SubLayer(x))      <- đối chứng A6, bản 2017

    Hai kiểu chỉ khác nhau ở CHỖ ĐẶT phép chuẩn hóa, không khác gì nữa. Viết
    một lớp xử lý được cả hai, chọn bằng `mo_hinh.vi_tri_chuan_hoa`.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        raise NotImplementedError("TASK 09 — Bảo")

    def forward(self, x, mask=None, rope=None):
        raise NotImplementedError("TASK 09 — Bảo")


class LopDecoder(nn.Module):
    """Một lớp decoder, ba khối con theo thứ tự:

    1. masked self-attention  — CÓ áp RoPE
    2. cross-attention        — KHÔNG áp RoPE
    3. feed forward
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        raise NotImplementedError("TASK 09 — Bảo")

    def forward(self, x, bo_nho_encoder, mask_tu, mask_cheo=None, rope=None):
        raise NotImplementedError("TASK 09 — Bảo")
