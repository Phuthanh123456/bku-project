"""TASK 04 — Dataset.  Người làm: My.  [Data Pipeline • Bắt buộc]

Đọc hai file văn bản song song, mã hóa thành số, trả về từng cặp câu.
Kiểm tra ngay lúc khởi tạo: số dòng file tiếng Anh phải bằng đúng số dòng
file tiếng Việt, lệch một dòng là toàn bộ dữ liệu bị lệch cặp.
"""

from __future__ import annotations

from torch.utils.data import Dataset


class DuLieuSongNgu(Dataset):
    def __init__(self, duong_dan_en, duong_dan_vi, tokenizer, do_dai_toi_da: int = 100) -> None:
        raise NotImplementedError("TASK 04 — My")

    def __len__(self) -> int:
        raise NotImplementedError("TASK 04 — My")

    def __getitem__(self, i: int):
        raise NotImplementedError("TASK 04 — My")
