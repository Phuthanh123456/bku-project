"""TASK 05 — Attention & Multi-Head Attention.  Người làm: Quân.  [Model Core • Bắt buộc]

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
    d_head = q.shape[-1]
    # (batch, so_head, len_q, d_head) @ (batch, so_head, d_head, len_k)
    #   -> (batch, so_head, len_q, len_k)
    diem = (q @ k.transpose(-2, -1)) / (d_head ** 0.5)

    if mask is not None:
        # CÁI BẪY fp16 SỐ 1: -1e9 tràn thành -inf ở fp16 (max 65504) -> NaN cả hàng.
        gia_tri_am = torch.finfo(diem.dtype).min
        diem = diem.masked_fill(~mask, gia_tri_am)

    trong_so_attention = torch.softmax(diem, dim=-1)
    if mask is not None:
        # Hàng bị mask toàn bộ (toàn PAD) vẫn có thể ra giá trị nhỏ khác 0 sau
        # softmax do sai số số học; ép về đúng 0 cho khớp bài test 3.
        trong_so_attention = trong_so_attention.masked_fill(~mask, 0.0)

    if dropout is not None:
        trong_so_attention = dropout(trong_so_attention)

    # (batch, so_head, len_q, len_k) @ (batch, so_head, len_k, d_head)
    #   -> (batch, so_head, len_q, d_head)
    dau_ra = trong_so_attention @ v
    return dau_ra, trong_so_attention


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

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _tach_head(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, len, d_model) -> (batch, so_head, len, d_head)
        batch, do_dai, _ = x.shape
        return x.view(batch, do_dai, self.so_head, self.d_head).transpose(1, 2)

    def _gop_head(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, so_head, len, d_head) -> (batch, len, d_model)
        batch, so_head, do_dai, d_head = x.shape
        return x.transpose(1, 2).contiguous().view(batch, do_dai, so_head * d_head)

    def forward(
        self,
        query: torch.Tensor,        # (batch, len_q, d_model)
        key: torch.Tensor,          # (batch, len_k, d_model)
        value: torch.Tensor,        # (batch, len_k, d_model)
        mask: torch.Tensor | None = None,
        rope=None,                  # đối tượng RoPE, hoặc None cho cross-attention
        vi_tri_bat_dau: int = 0,    # khác 0 khi sinh câu có KV cache, TASK 19
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._tach_head(self.w_q(query))   # (batch, so_head, len_q, d_head)
        k = self._tach_head(self.w_k(key))     # (batch, so_head, len_k, d_head)
        v = self._tach_head(self.w_v(value))   # (batch, so_head, len_k, d_head)

        # RoPE chỉ áp cho self-attention (query, key), KHÔNG BAO GIỜ cho value.
        if rope is not None:
            q = rope(q, vi_tri_bat_dau)
            k = rope(k, vi_tri_bat_dau)

        dau_ra, trong_so_attention = attention_tich_vo_huong(q, k, v, mask, self.dropout)
        dau_ra = self.w_o(self._gop_head(dau_ra))
        return dau_ra, trong_so_attention
