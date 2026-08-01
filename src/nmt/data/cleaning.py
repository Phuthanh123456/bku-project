"""TASK 02 — Làm sạch dữ liệu IWSLT.  Người làm: My.  [Data • Bắt buộc]

Xong khi: bảng thống kê lọc đầy đủ, số câu test lọt vào train bằng 0.

Thứ tự các bước lọc (giữ đúng thứ tự này để bảng thống kê đọc được):
    1. Chuẩn hóa Unicode về dạng NFC.
       BẮT BUỘC với tiếng Việt. Không làm thì cùng một chữ có dấu bị máy coi là
       hai chữ khác nhau, tokenizer học ra hai token riêng, BLEU tụt mà không rõ lý do.
    2. Bỏ dòng rỗng.
    3. Bỏ cặp câu trùng lặp.
    4. Bỏ câu dài quá 100 từ.
    5. Bỏ cặp có độ dài hai bên lệch nhau quá 3 lần — thường là dữ liệu bị lệch dòng.
    6. Kiểm tra câu của tập test có lọt vào tập train không.

Mỗi bước phải ghi lại: tên bước, số cặp trước khi lọc, số cặp bị loại,
số cặp còn lại, tỉ lệ phần trăm bị loại. Bảng này cho thấy có bước nào cắt quá tay không.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThongKeLoc:
    """Một hàng của bảng thống kê lọc dữ liệu."""

    ten_buoc: str
    truoc: int
    bi_loai: int
    con_lai: int

    @property
    def ti_le_loai(self) -> float:
        return 100.0 * self.bi_loai / self.truoc if self.truoc else 0.0


def chuan_hoa_nfc(van_ban: str) -> str:
    raise NotImplementedError("TASK 02 — My")


def lam_sach_cap_cau(
    cau_en: list[str],
    cau_vi: list[str],
    do_dai_toi_da: int = 100,
    ti_le_lech_toi_da: float = 3.0,
) -> tuple[list[str], list[str], list[ThongKeLoc]]:
    """Returns: (en đã sạch, vi đã sạch, bảng thống kê từng bước)."""
    raise NotImplementedError("TASK 02 — My")


def kiem_tra_ro_ri(train_en: list[str], test_en: list[str]) -> list[int]:
    """Trả về chỉ số các câu train bị trùng với test.

    Con số mong muốn là 0. Khác 0 thì phải loại các câu đó khỏi tập train
    RỒI GHI LẠI CON SỐ VÀO BÁO CÁO — đừng lặng lẽ xóa.
    """
    raise NotImplementedError("TASK 02 — My")
