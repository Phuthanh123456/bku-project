"""TASK 06 — Mã hóa vị trí: RoPE + sin-cos.  Người làm: Quân.  [Model Core • Bắt buộc]

Xong khi: đổi cấu hình `mo_hinh.ma_hoa_vi_tri` là chuyển được giữa hai phương án
mà KHÔNG sửa dòng code nào. Phương án sin-cos là đối chứng cho ablation A4.

Đây là thành phần khó nhất trong bốn thành phần mới. Chỗ dễ sai nhất là phần
GHÉP CẶP CHIỀU khi xoay.

CÁI BẪY fp16 SỐ 2:
    Bảng góc quay (inv_freq, cos, sin) BẮT BUỘC tính ở float32 rồi mới ép về
    dtype của tensor đầu vào. Nếu tính ở fp16 thì từ khoảng vị trí 100 trở đi
    góc quay mất độ phân giải và RoPE hỏng ÂM THẦM.
    Đáng chú ý: bài test 4 chạy ở float32 nên vẫn báo ĐẠT — lỗi chỉ lộ ra lúc
    huấn luyện thật. Bài test 12 (chạy trong autocast fp16) mới bắt được.

RoPE CHỈ ÁP CHO SELF-ATTENTION, và chỉ áp cho query với key.
Không áp cho cross-attention (bài test 11). Không áp cho value (bài test 4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary Position Embedding — https://arxiv.org/abs/2104.09864

    Xoay vector query và key đi một góc tỉ lệ với vị trí. Sau khi xoay, tích vô
    hướng giữa query ở vị trí m và key ở vị trí n chỉ còn phụ thuộc vào hiệu
    (m - n), tức khoảng cách giữa hai từ, chứ không phụ thuộc vị trí tuyệt đối.
    """

    def __init__(self, d_head: int, theta: float = 10000.0, do_dai_toi_da: int = 512) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head ({d_head}) phải là số chẵn để ghép cặp chiều khi xoay")
        self.d_head = d_head
        self.do_dai_toi_da = do_dai_toi_da

        # CÁI BẪY fp16 SỐ 2: bảng góc quay PHẢI tính ở float32.
        # inv_freq[i] = theta^(-2i/d_head), i = 0..d_head/2-1
        nua = d_head // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, nua, dtype=torch.float32) * 2.0 / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Bảng cos/sin cache sẵn tới do_dai_toi_da, tính ở float32.
        vi_tri = torch.arange(do_dai_toi_da, dtype=torch.float32)
        goc = torch.outer(vi_tri, inv_freq)   # (do_dai_toi_da, d_head/2)
        self.register_buffer("cos_cache", goc.cos(), persistent=False)
        self.register_buffer("sin_cache", goc.sin(), persistent=False)

    def _mo_rong_cache_neu_can(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self.cos_cache.shape[0]:
            return
        vi_tri = torch.arange(seq_len, dtype=torch.float32, device=device)
        goc = torch.outer(vi_tri, self.inv_freq.to(device=device, dtype=torch.float32))
        self.cos_cache = goc.cos()
        self.sin_cache = goc.sin()
        self.do_dai_toi_da = seq_len

    def forward(
        self,
        x: torch.Tensor,            # (batch, so_head, seq_len, d_head)
        vi_tri_bat_dau: int = 0,    # khác 0 khi sinh câu từng bước có KV cache
    ) -> torch.Tensor:
        """Áp phép quay. Chỉ gọi cho query và key — KHÔNG BAO GIỜ cho value."""
        seq_len = x.shape[-2]
        self._mo_rong_cache_neu_can(vi_tri_bat_dau + seq_len, x.device)

        # (seq_len, d_head/2), tính ở float32 rồi mới ép về dtype đầu vào.
        cos = self.cos_cache[vi_tri_bat_dau : vi_tri_bat_dau + seq_len].to(dtype=x.dtype)
        sin = self.sin_cache[vi_tri_bat_dau : vi_tri_bat_dau + seq_len].to(dtype=x.dtype)

        x1, x2 = x[..., : self.d_head // 2], x[..., self.d_head // 2 :]  # mỗi nửa (batch, so_head, seq_len, d_head/2)

        # Phép quay 2D áp cho từng cặp (x1_i, x2_i):
        #   ra1 = x1*cos - x2*sin
        #   ra2 = x1*sin + x2*cos
        ra1 = x1 * cos - x2 * sin
        ra2 = x1 * sin + x2 * cos
        return torch.cat([ra1, ra2], dim=-1)


class MaHoaViTriSinCos(nn.Module):
    """Positional Encoding sin-cos của bản 2017 — phương án đối chứng cho A4.

    Khác RoPE ở chỗ: cộng thẳng vector vị trí vào embedding, không đụng vào
    query/key bên trong attention.
    """

    def __init__(self, d_model: int, do_dai_toi_da: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # PE(pos, 2i)   = sin(pos / theta^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / theta^(2i/d_model))
        vi_tri = torch.arange(do_dai_toi_da, dtype=torch.float32).unsqueeze(1)   # (L, 1)
        chia = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )  # (d_model/2,)
        goc = vi_tri * chia   # (L, d_model/2)

        pe = torch.zeros(do_dai_toi_da, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(goc)
        pe[:, 1::2] = torch.cos(goc)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)   # (1, L, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> cùng kích thước."""
        seq_len = x.shape[1]
        if seq_len > self.pe.shape[1]:
            raise ValueError(
                f"seq_len ({seq_len}) vượt do_dai_toi_da đã cache ({self.pe.shape[1]}) của positional encoding"
            )
        x = x + self.pe[:, :seq_len, :].to(dtype=x.dtype)
        return self.dropout(x)
