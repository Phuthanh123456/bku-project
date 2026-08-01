"""TASK 07 — Chuẩn hóa: RMSNorm + LayerNorm.  Người làm: Quân.  [Model Core • Bắt buộc]

Xong khi: đối chiếu RMSNorm tự viết với torch.nn.RMSNorm sai lệch dưới 1e-5 (bài test 6).

Trả lời trực tiếp nhận xét 3 của mentor: KHÔNG mặc định RMSNorm tốt hơn
LayerNorm, phải cài cả hai rồi để ablation A1 quyết định bằng số liệu.
A1 là thí nghiệm ưu tiên cao nhất vì mentor nêu trực tiếp.

CÁI BẪY fp16 SỐ 3:
    Phép tính trung bình bình phương BẮT BUỘC thực hiện ở float32 rồi mới ép
    kết quả về dtype ban đầu, giống cách các mô hình mã nguồn mở đang làm.
    Ở fp16 giá trị bình phương dễ vượt 65504 và tràn thành vô cùng, kéo theo
    NaN cho cả mô hình.

RMSNorm khác LayerNorm ở hai điểm: KHÔNG trừ trung bình, và KHÔNG có độ lệch
cộng thêm. Bài test 8 bắt đúng lỗi cài nhầm thành LayerNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * weight — https://arxiv.org/abs/1910.07467

    Khoảng 10 dòng, đây là thành phần dễ nhất trong bốn thành phần mới.
    Nhớ hằng số epsilon chống chia cho 0.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError("TASK 07 — Quân")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TASK 07 — Quân")


class LayerNorm(nn.Module):
    """LayerNorm tự viết — phương án đối chứng cho ablation A1.

    Trừ trung bình, chia độ lệch chuẩn, có cả hệ số nhân và độ lệch cộng thêm.
    Tự viết chứ không gọi nn.LayerNorm, vì quy tắc của đồ án là không dùng lớp
    dựng sẵn trong src/.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        raise NotImplementedError("TASK 07 — Quân")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("TASK 07 — Quân")


def tao_chuan_hoa(kieu: str, d_model: int, eps: float = 1e-6) -> nn.Module:
    """Factory đọc `mo_hinh.kieu_chuan_hoa` từ YAML.

    Cả mô hình chỉ được tạo lớp chuẩn hóa qua hàm này, nhờ vậy đổi kiến trúc
    chỉ cần sửa một dòng cấu hình.
    """
    if kieu == "rmsnorm":
        return RMSNorm(d_model, eps)
    if kieu == "layernorm":
        return LayerNorm(d_model, eps)
    raise ValueError(f"kieu_chuan_hoa không hợp lệ: {kieu} (phải là rmsnorm hoặc layernorm)")
