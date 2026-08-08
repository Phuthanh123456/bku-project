"""TASK 03/04 — Wrapper tokenizer.  Người làm: My.  [NLP • Bắt buộc]

Bọc tokenizer HuggingFace thành giao diện gọn gàng cho phần còn lại của codebase.
Cũng cung cấp các hàm kiểm tra chất lượng tokenizer (fertility, UNK rate).

Bốn token đặc biệt (ID cố định, KHÔNG đổi):
    <pad> = 0,  <unk> = 1,  <bos> = 2,  <eos> = 3
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tokenizers import Tokenizer as HFTokenizer

# Token strings và ID tương ứng — nhất quán với train_tokenizer.py
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# Token dự phòng để dành (đăng ký ngay từ đầu — không phải đặc thù dự án này
# nhưng hữu ích nếu sau muốn gộp 2 chiều dịch vào 1 mô hình)
TOKEN_DE_DANH = ["<2en>", "<2vi>"]


def nap_tokenizer(duong_dan: str | Path) -> "HFTokenizer":
    """Load tokenizer từ file .json đã train (TASK 03).

    Args:
        duong_dan: đường dẫn tới artifacts/tokenizer/tokenizer.json

    Returns:
        Tokenizer instance của HuggingFace

    Example::
        tok = nap_tokenizer("artifacts/tokenizer/tokenizer.json")
        ids = tok.encode("Hello world").ids   # → [...]
        text = tok.decode(ids)               # → "Hello world"
    """
    from tokenizers import Tokenizer  # pyrefly: ignore [missing-import]

    duong_dan = Path(duong_dan)
    if not duong_dan.exists():
        raise FileNotFoundError(
            f"Khong tim thay tokenizer tai: {duong_dan}\n"
            "Hay chay scripts/train_tokenizer.py truoc."
        )
    return Tokenizer.from_file(str(duong_dan))


def train_tokenizer(
    duong_dan_van_ban: list[str | Path],
    vocab_size: int = 32000,
    duong_dan_luu: str | Path = "artifacts/tokenizer/tokenizer.json",
) -> "HFTokenizer":
    """Train BPE tokenizer trên danh sách file văn bản.

    Hàm tiện ích — logic chi tiết nằm ở scripts/train_tokenizer.py.
    Hàm này để các module khác (ví dụ ablation script) gọi lại nếu cần
    train lại tokenizer với tham số khác mà không cần import script.

    Args:
        duong_dan_van_ban: danh sách đường dẫn tới file .en, .vi
        vocab_size: kích thước vocab (mặc định 32000)
        duong_dan_luu: nơi lưu tokenizer.json
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders  # pyrefly: ignore

    special_tokens = [PAD, UNK, BOS, EOS]

    tokenizer = Tokenizer(models.BPE(unk_token=UNK))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
    )

    # Đọc tất cả file thành dòng
    tat_ca_dong: list[str] = []
    for duong_dan in duong_dan_van_ban:
        tat_ca_dong.extend(Path(duong_dan).read_text(encoding="utf-8").splitlines())

    tokenizer.train_from_iterator(tat_ca_dong, trainer=trainer)

    duong_dan_luu = Path(duong_dan_luu)
    duong_dan_luu.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(duong_dan_luu))

    return tokenizer


def kiem_tra_ma_hoa_giai_ma(tokenizer: "HFTokenizer", cau: list[str]) -> float:
    """Mã hóa rồi giải mã ngược, trả về tỉ lệ câu khớp hoàn toàn.

    Yêu cầu: > 99%.

    Args:
        tokenizer: tokenizer đã load
        cau: danh sách câu gốc (chưa encode)

    Returns:
        Tỉ lệ [0.0, 1.0] số câu decode ra khớp với input gốc
    """
    if not cau:
        return 1.0
    dem_khop = sum(
        tokenizer.decode(tokenizer.encode(c).ids) == c
        for c in cau
    )
    return dem_khop / len(cau)


def ti_le_token_la(tokenizer: "HFTokenizer", cau: list[str]) -> float:
    """Tỉ lệ phần trăm token <unk> trên tập dev.

    Yêu cầu: dưới 0.5%.

    Args:
        tokenizer: tokenizer đã load
        cau: danh sách câu cần kiểm tra

    Returns:
        Tỉ lệ [0.0, 1.0] token UNK trong toàn bộ output
    """
    if not cau:
        return 0.0
    tong_unk = 0
    tong_token = 0
    for c in cau:
        ids = tokenizer.encode(c).ids
        tong_unk += ids.count(UNK_ID)
        tong_token += len(ids)
    return tong_unk / tong_token if tong_token > 0 else 0.0
