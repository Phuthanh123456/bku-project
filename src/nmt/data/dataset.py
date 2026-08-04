"""TASK 04 — Dataset.  Người làm: My.  [Data Pipeline • Bắt buộc]

Đọc hai file văn bản song song, mã hóa thành số, trả về từng cặp câu.
Kiểm tra ngay lúc khởi tạo: số dòng file tiếng Anh phải bằng đúng số dòng
file tiếng Việt, lệch một dòng là toàn bộ dữ liệu bị lệch cặp.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from tokenizers import Tokenizer

# ID của các special token — phải khớp với thứ tự khai báo trong TASK 03
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class DuLieuSongNgu(Dataset):
    """Dataset song ngữ En-Vi cho bài toán dịch máy.

    Mỗi mẫu trả về một dict:
        {
            "src_ids": Tensor[int64],  # token IDs tiếng Anh, KHÔNG có BOS/EOS
            "tgt_ids": Tensor[int64],  # token IDs tiếng Việt, CÓ BOS ở đầu và EOS ở cuối
        }

    Về thiết kế BOS/EOS:
        - src (encoder input): không cần BOS/EOS, encoder đọc toàn bộ câu nguồn
          như một chuỗi; BOS/EOS không mang thêm thông tin gì cho encoder.
        - tgt (decoder): BOS ở đầu là tín hiệu bắt đầu sinh câu; EOS ở cuối
          là tín hiệu dừng. collate_fn sẽ tách tgt thành:
            decoder_input  = tgt[:-1]  (có BOS, không có EOS)
            labels         = tgt[1:]   (không có BOS, có EOS)
          để teacher forcing hoạt động đúng.

    Câu quá dài (> do_dai_toi_da) sẽ bị CẮT BỚT (truncate), không bỏ qua,
    vì bước cắt lọc đã được thực hiện ở TASK 02 với ngưỡng tính bằng từ;
    ngưỡng ở đây tính bằng subword (có thể hơi nhiều hơn một chút).
    """

    def __init__(
        self,
        duong_dan_en: str | Path,
        duong_dan_vi: str | Path,
        tokenizer: "Tokenizer",
        do_dai_toi_da: int = 100,
    ) -> None:
        duong_dan_en = Path(duong_dan_en)
        duong_dan_vi = Path(duong_dan_vi)

        if not duong_dan_en.exists():
            raise FileNotFoundError(f"Khong tim thay: {duong_dan_en}")
        if not duong_dan_vi.exists():
            raise FileNotFoundError(f"Khong tim thay: {duong_dan_vi}")

        dong_en = duong_dan_en.read_text(encoding="utf-8").splitlines()
        dong_vi = duong_dan_vi.read_text(encoding="utf-8").splitlines()

        # Bất biến quan trọng nhất: EN phải bằng VI về số dòng
        if len(dong_en) != len(dong_vi):
            raise RuntimeError(
                f"So dong EN ({len(dong_en)}) != VI ({len(dong_vi)}) — "
                "du lieu bi lech cap! Chay lai prepare_data.py."
            )

        self._tokenizer = tokenizer
        self._do_dai_toi_da = do_dai_toi_da

        # Mã hóa toàn bộ corpus ngay khi khởi tạo — đơn giản, tránh overhead
        # trong __getitem__ khi DataLoader gọi nhiều lần song song.
        # Với 131k câu × 2 ngôn ngữ, thời gian encode ~5-10 giây — chấp nhận được.
        self._src: list[list[int]] = []
        self._tgt: list[list[int]] = []

        for en, vi in zip(dong_en, dong_vi):
            src_ids = tokenizer.encode(en).ids[:do_dai_toi_da]
            # tgt có thêm BOS ở đầu và EOS ở cuối
            tgt_ids = (
                [BOS_ID]
                + tokenizer.encode(vi).ids[:do_dai_toi_da]
                + [EOS_ID]
            )
            self._src.append(src_ids)
            self._tgt.append(tgt_ids)

    def __len__(self) -> int:
        return len(self._src)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "src_ids": torch.tensor(self._src[i], dtype=torch.long),
            "tgt_ids": torch.tensor(self._tgt[i], dtype=torch.long),
        }

    def do_dai_src(self, i: int) -> int:
        """Độ dài câu nguồn (dùng để sắp xếp batch theo độ dài)."""
        return len(self._src[i])
