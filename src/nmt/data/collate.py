"""TASK 04 — DataLoader gom batch theo số token.  Người làm: My.  [Data Pipeline • Bắt buộc]

Xong khi: batch mẫu đúng kích thước, tỉ lệ ô đệm thừa dưới 25 phần trăm.

Gom batch theo TỔNG SỐ TOKEN chứ không theo số câu. Gom theo số câu thì batch
toàn câu ngắn sẽ lãng phí bộ nhớ, còn batch toàn câu dài thì tràn bộ nhớ.
Xếp các câu có độ dài gần nhau vào cùng một batch để đỡ phí padding.

Nhớ truyền seed_cho_worker và sinh_generator từ nmt.utils.seed vào DataLoader,
nếu không thì thứ tự trộn dữ liệu đổi theo số worker và mất tính tái lập.
"""

from __future__ import annotations


def gom_batch_theo_token(dataset, so_token_moi_batch: int = 4096) -> list[list[int]]:
    """Trả về danh sách các batch, mỗi batch là danh sách chỉ số câu."""
    raise NotImplementedError("TASK 04 — My")


def ghep_batch(cac_mau, pad_id: int):
    """collate_fn: đệm cho bằng nhau, sinh padding mask và causal mask.

    Returns: dict gồm src_ids, tgt_ids, src_mask, tgt_mask, nhan (labels).
    """
    raise NotImplementedError("TASK 04 — My")


def ti_le_o_dem(batch, pad_id: int) -> float:
    """Tỉ lệ phần trăm ô đệm thừa trong batch. Yêu cầu dưới 25%."""
    raise NotImplementedError("TASK 04 — My")
