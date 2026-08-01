"""TASK 16 — Chấm điểm BLEU và chrF++.  Người làm: My.  [Evaluation • Bắt buộc]

Xong khi: BLEU trên tst2013 tối thiểu 19, mức tốt là từ 22 trở lên.

BẮT BUỘC ghi lại nguyên văn CHUỖI CHỮ KÝ (signature) mà sacrebleu xuất ra.
Thiếu chuỗi này thì điểm BLEU không so sánh được với bất kỳ bài báo nào, vì
BLEU chỉ so sánh được khi hai bên dùng cùng một cách tokenize.

chrF++ đo mức trùng khớp ở cấp nhóm ký tự. Thước đo này quan trọng với tiếng
Việt vì tiếng Việt viết rời từng âm tiết.
"""

from __future__ import annotations


def cham_bleu(du_doan: list[str], tham_chieu: list[str]) -> tuple[float, str]:
    """Returns: (điểm BLEU, chuỗi chữ ký của sacrebleu)."""
    raise NotImplementedError("TASK 16 — My")


def cham_chrf(du_doan: list[str], tham_chieu: list[str]) -> tuple[float, str]:
    """chrF++ (word_order=2). Returns: (điểm, chuỗi chữ ký)."""
    raise NotImplementedError("TASK 16 — My")
