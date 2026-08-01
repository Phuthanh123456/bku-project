"""TASK 09 — Ghép Encoder-Decoder thành mô hình.  Người làm: Bảo.  [Model Core • Bắt buộc]

Xong khi: gradient chảy tới mọi tham số, không có giá trị NaN (bài test 5).

BA ĐIỂM BẮT BUỘC NHỚ:

1. Với Pre-Norm thì PHẢI có một phép chuẩn hóa cuối cùng NGAY TRƯỚC lớp Linear
   xuất ra từ vựng. Luồng residual không bao giờ được chuẩn hóa nên giá trị
   phình dần qua từng lớp. Quên bước này KHÔNG làm chương trình báo lỗi, chỉ
   khiến mô hình mất ổn định mà không rõ nguyên nhân. Bài test 10 bắt lỗi này.

2. RoPE chỉ áp cho self-attention của encoder và của decoder.
   KHÔNG áp cho cross-attention. Bài test 11 bắt lỗi này.

3. Encoder, decoder và lớp xuất dùng CHUNG một ma trận embedding.
   Khởi tạo trọng số theo Xavier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerNMT(nn.Module):
    """Transformer Encoder-Decoder tự viết, khoảng 48 triệu tham số.

    Toàn bộ lựa chọn kiến trúc đọc từ cfg.mo_hinh, nên đổi kiến trúc là đổi YAML
    chứ không sửa code. Đây là điều kiện để chạy được 4 thí nghiệm của TASK 17.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        raise NotImplementedError("TASK 09 — Bảo")

    def encode(self, src_ids, src_mask):
        raise NotImplementedError("TASK 09 — Bảo")

    def decode(self, tgt_ids, bo_nho_encoder, tgt_mask, src_mask):
        raise NotImplementedError("TASK 09 — Bảo")

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        """Returns: logits có kích thước (batch, len_tgt, vocab_size)."""
        raise NotImplementedError("TASK 09 — Bảo")

    def dem_tham_so(self) -> dict[str, int]:
        """Đếm tham số theo từng thành phần, cho bảng kết quả đầu ra của Phase 2.

        Returns:
            {"embedding": ..., "encoder": ..., "decoder": ..., "lop_xuat": ..., "tong": ...}

        Đối chiếu với con số tính tay trên giấy, sai lệch phải dưới 1 phần trăm.
        Bảng này chứng minh nhóm hiểu rõ mô hình mình viết ra chứ không chỉ ghép cho chạy.
        """
        raise NotImplementedError("TASK 09 — Bảo")
