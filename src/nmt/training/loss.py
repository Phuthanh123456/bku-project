"""TASK 13 — Hàm loss.  Người làm: Quân.  [Training • Bắt buộc]

Label Smoothing MẶC ĐỊNH TẮT (label_smoothing = 0.0) theo nhận xét 1 của mentor.
Đặt 0.1 thì bật, khi đó không bắt mô hình tin tuyệt đối vào một đáp án duy nhất.
Ablation A3 (TASK 18) quyết định giữ hay bỏ.

Nhớ bỏ qua vị trí <pad> khi tính loss, nếu không thì mô hình được thưởng vì
đoán đúng token đệm và loss trông đẹp hơn thực tế.
"""

from __future__ import annotations

import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    """Cross entropy có label smoothing, bỏ qua vị trí padding.

    smoothing = 0.0 thì phải cho ra kết quả GIỐNG HỆT cross entropy thường —
    đây là phép kiểm rẻ tiền nên viết luôn thành unit test.
    """

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.0) -> None:
        super().__init__()
        raise NotImplementedError("TASK 13 — Quân")

    def forward(self, logits, nhan):
        raise NotImplementedError("TASK 13 — Quân")
