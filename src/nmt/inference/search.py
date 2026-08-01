"""TASK 16 — Greedy Search.       Người làm: My.   [Evaluation • Bắt buộc]
TASK 19 — Beam Search & KV Cache. Người làm: Phú.  [Inference • ƯU TIÊN THẤP]

Theo nhận xét 6 của mentor: Beam Search và KV Cache chưa cần vội, ưu tiên chạy
được model trước. Chấm điểm baseline ở TASK 16 dùng Greedy.
TASK 19 là task ĐẦU TIÊN BỊ CẮT nếu trễ tiến độ.

Bài test rẻ nhất mà bắt được hầu hết lỗi beam search:
    đặt beam = 1 thì Beam Search phải cho kết quả TRÙNG KHỚP TỪNG CHỮ với Greedy.
    Nằm ở tests/test_beam_search.py.

Yêu cầu của TASK 19: Beam phải cho BLEU cao hơn Greedy ít nhất 0,5 điểm.
Không cao hơn thì gần như chắc chắn Beam Search đang có lỗi, phải kiểm tra lại
chứ đừng ghi vào báo cáo là "beam không hiệu quả".
"""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_search(model, src_ids, src_mask, bos_id: int, eos_id: int, do_dai_toi_da: int = 128):
    """Mỗi bước chọn luôn từ có xác suất cao nhất."""
    raise NotImplementedError("TASK 16 — My")


@torch.no_grad()
def beam_search(
    model, src_ids, src_mask, bos_id: int, eos_id: int,
    beam_size: int = 4, he_so_phat_do_dai: float = 1.0, do_dai_toi_da: int = 128,
):
    """Giữ lại beam_size phương án tốt nhất ở mỗi bước.

    Kèm hệ số phạt độ dài để mô hình không thiên vị câu quá ngắn.
    """
    raise NotImplementedError("TASK 19 — Phú")
