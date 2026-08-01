"""TASK 05 — Attention & Multi-Head Attention.  Người làm: Bảo.  [Model Core • Bắt buộc]

Xong khi: đối chiếu với F.scaled_dot_product_attention sai lệch dưới 1e-5 (bài test 6).

Ghi rõ kích thước tensor vào chú thích ở TỪNG phép nhân ma trận. Lỗi hoán vị
chiều trong attention không làm chương trình báo lỗi — mô hình vẫn chạy, loss
vẫn giảm, vẫn dịch ra chữ, nhưng dịch sai.

CÁI BẪY fp16 SỐ 1:
    Khi che vị trí bị mask, dùng torch.finfo(scores.dtype).min
    TUYỆT ĐỐI KHÔNG dùng -1e9. Ở fp16 số lớn nhất biểu diễn được chỉ là 65504
    nên -1e9 biến thành âm vô cùng; gặp một hàng bị mask toàn bộ là softmax ra NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def attention_tich_vo_huong(
    q: torch.Tensor,            # (batch, so_head, len_q, d_head)
    k: torch.Tensor,            # (batch, so_head, len_k, d_head)
    v: torch.Tensor,            # (batch, so_head, len_k, d_head)
    mask: torch.Tensor | None = None,   # (batch, 1, len_q, len_k) — True nghĩa là ĐƯỢC nhìn
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(Q @ K^T / sqrt(d_k)) @ V

    Returns:
        (dau_ra, trong_so_attention)
        dau_ra:             (batch, so_head, len_q, d_head)
        trong_so_attention: (batch, so_head, len_q, len_k) — cần cho bản đồ nhiệt
    """
    raise NotImplementedError("TASK 05 — Bảo")


class MultiHeadAttention(nn.Module):
    """Dùng chung được cho cả self-attention lẫn cross-attention.

    Mask luôn truyền từ bên ngoài vào, module này không tự sinh mask.
    RoPE cũng truyền từ ngoài vào, vì cross-attention KHÔNG áp RoPE.
    """

    def __init__(self, d_model: int, so_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % so_head != 0:
            raise ValueError(f"d_model ({d_model}) phải chia hết cho so_head ({so_head})")
        self.d_model = d_model
        self.so_head = so_head
        self.d_head = d_model // so_head
        raise NotImplementedError("TASK 05 — Bảo")

    def forward(
        self,
        query: torch.Tensor,        # (batch, len_q, d_model)
        key: torch.Tensor,          # (batch, len_k, d_model)
        value: torch.Tensor,        # (batch, len_k, d_model)
        mask: torch.Tensor | None = None,
        rope=None,                  # đối tượng RoPE, hoặc None cho cross-attention
        vi_tri_bat_dau: int = 0,    # khác 0 khi sinh câu có KV cache, TASK 19
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("TASK 05 — Bảo")
