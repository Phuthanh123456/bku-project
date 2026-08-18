"""TASK 09 — Ghép Encoder-Decoder thành mô hình.  Người làm: Quân.  [Model Core • Bắt buộc]

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

API SẴN CÓ — gọi đúng tên này, đừng viết lại:

    từ nmt.model.attention      MultiHeadAttention          (TASK 05, Quân)
    từ nmt.model.positional     RoPE, MaHoaViTriSinCos      (TASK 06, Quân)
    từ nmt.model.normalization  build_norm(kieu, d_model, eps)
                                ResidualConnection(d_model, kieu_chuan_hoa,
                                                   vi_tri_chuan_hoa, dropout, eps)
                                build_final_norm(kieu, d_model, vi_tri_chuan_hoa,
                                                 co_chuan_hoa_cuoi, eps)   (TASK 07, Bảo)
    từ nmt.model.layers         build_ffn(kieu, d_model, d_ff, dropout)
                                EncoderLayer, DecoderLayer                 (TASK 08, Bảo)

ResidualConnection đã xử lý sẵn CẢ Pre-Norm lẫn Post-Norm, nên trong lớp encoder
và decoder KHÔNG cần viết nhánh if nào cho `vi_tri_chuan_hoa`. Mỗi khối con cần
MỘT ResidualConnection RIÊNG — dùng chung một cái là hai khối xài chung trọng số
chuẩn hóa, chương trình vẫn chạy nhưng kiến trúc đã sai mà không báo lỗi.

CÒN THIẾU, cần làm trong TASK 09: một hàm chọn mã hóa vị trí theo cấu hình, kiểu

    def build_positional(cfg): ...   # đọc cfg.mo_hinh.ma_hoa_vi_tri

Thiếu hàm này thì tiêu chí "Xong khi" của TASK 06 (đổi cấu hình là chuyển được
giữa RoPE và sin-cos mà không sửa code) chưa chứng minh được, và ablation A4 của
TASK 17 không chạy được.
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
        raise NotImplementedError("TASK 09 — Quân")

    def encode(self, src_ids, src_mask):
        raise NotImplementedError("TASK 09 — Quân")

    def decode(self, tgt_ids, bo_nho_encoder, tgt_mask, src_mask):
        raise NotImplementedError("TASK 09 — Quân")

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        """Returns: logits có kích thước (batch, len_tgt, vocab_size)."""
        raise NotImplementedError("TASK 09 — Quân")

    def dem_tham_so(self) -> dict[str, int]:
        """Đếm tham số theo từng thành phần, cho bảng kết quả đầu ra của Phase 2.

        Returns:
            {"embedding": ..., "encoder": ..., "decoder": ..., "lop_xuat": ..., "tong": ...}

        Đối chiếu với con số tính tay trên giấy, sai lệch phải dưới 1 phần trăm.
        Bảng này chứng minh nhóm hiểu rõ mô hình mình viết ra chứ không chỉ ghép cho chạy.
        """
        raise NotImplementedError("TASK 09 — Quân")
