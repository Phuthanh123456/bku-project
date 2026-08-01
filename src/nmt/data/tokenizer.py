"""TASK 03 — BPE Tokenizer 32k.  Người làm: My.  [NLP • Bắt buộc]

Xong khi: token lạ trên dev dưới 0,5 phần trăm; mã hóa rồi giải mã ngược
khớp hoàn toàn trên hơn 99 phần trăm số câu.

Một bộ từ điển BPE 32.000 DÙNG CHUNG cho cả tiếng Anh và tiếng Việt.
Bốn token đặc biệt: <pad> <s> </s> <unk>.
Đăng ký sẵn thêm hai token <2en> và <2vi> để dành cho hướng phát triển gộp
hai chiều dịch vào một mô hình. Đăng ký ngay từ đầu vì thêm token sau khi đã
train xong tokenizer thì phải train lại toàn bộ.
"""

from __future__ import annotations

from pathlib import Path

PAD, BOS, EOS, UNK = "<pad>", "<s>", "</s>", "<unk>"
TOKEN_DE_DANH = ["<2en>", "<2vi>"]


def train_tokenizer(
    duong_dan_van_ban: list[str | Path],
    vocab_size: int = 32000,
    duong_dan_luu: str | Path = "artifacts/tokenizer/tokenizer.json",
):
    raise NotImplementedError("TASK 03 — My")


def nap_tokenizer(duong_dan: str | Path):
    raise NotImplementedError("TASK 03 — My")


def kiem_tra_ma_hoa_giai_ma(tokenizer, cau: list[str]) -> float:
    """Mã hóa rồi giải mã ngược, trả về tỉ lệ câu khớp hoàn toàn. Yêu cầu > 99%."""
    raise NotImplementedError("TASK 03 — My")


def ti_le_token_la(tokenizer, cau: list[str]) -> float:
    """Tỉ lệ phần trăm token <unk> trên tập dev. Yêu cầu dưới 0,5%."""
    raise NotImplementedError("TASK 03 — My")
