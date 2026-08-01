"""TASK 06 — Mã hóa vị trí: RoPE + sin-cos.  Người làm: Bảo.  [Model Core • Bắt buộc]

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
        raise NotImplementedError("TASK 06 — Bảo")

    def forward(
        self,
        x: torch.Tensor,            # (batch, so_head, seq_len, d_head)
        vi_tri_bat_dau: int = 0,    # khác 0 khi sinh câu từng bước có KV cache
    ) -> torch.Tensor:
        """Áp phép quay. Chỉ gọi cho query và key — KHÔNG BAO GIỜ cho value."""
        raise NotImplementedError("TASK 06 — Bảo")


class MaHoaViTriSinCos(nn.Module):
    """Positional Encoding sin-cos của bản 2017 — phương án đối chứng cho A4.

    Khác RoPE ở chỗ: cộng thẳng vector vị trí vào embedding, không đụng vào
    query/key bên trong attention.
    """

    def __init__(self, d_model: int, do_dai_toi_da: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("TASK 06 — Bảo")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) -> cùng kích thước."""
        raise NotImplementedError("TASK 06 — Bảo")
