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

import unicodedata
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
    """Chuẩn hóa Unicode về dạng NFC.

    BẮT BUỘC với tiếng Việt: ký tự có dấu dạng NFD (tổ hợp nhiều codepoint)
    và NFC (một codepoint duy nhất) trông giống hệt nhau nhưng máy coi là khác
    nhau. Không chuẩn hóa thì tokenizer học ra hai token riêng cho cùng một chữ,
    BLEU tụt mà không rõ lý do.
    """
    return unicodedata.normalize("NFC", van_ban)


def lam_sach_cap_cau(
    cau_en: list[str],
    cau_vi: list[str],
    do_dai_toi_da: int = 100,
    ti_le_lech_toi_da: float = 3.0,
) -> tuple[list[str], list[str], list[ThongKeLoc]]:
    """Làm sạch và lọc cặp câu EN-VI cho tập TRAIN.

    Áp dụng toàn bộ pipeline lọc — KHÔNG dùng cho dev/test
    (dev/test chỉ chuẩn hóa NFC, giữ nguyên số dòng).

    Returns: (en đã sạch, vi đã sạch, bảng thống kê từng bước).
    """
    bang_thong_ke: list[ThongKeLoc] = []

    # --- Bước 1: Chuẩn hóa NFC ---
    cau_en = [chuan_hoa_nfc(c.strip()) for c in cau_en]
    cau_vi = [chuan_hoa_nfc(c.strip()) for c in cau_vi]

    # --- Bước 2: Bỏ dòng rỗng ---
    truoc = len(cau_en)
    cap_loc: list[tuple[str, str]] = [
        (en, vi) for en, vi in zip(cau_en, cau_vi) if en and vi
    ]
    bi_loai = truoc - len(cap_loc)
    bang_thong_ke.append(ThongKeLoc("bo_rong", truoc, bi_loai, len(cap_loc)))

    # --- Bước 3: Bỏ trùng lặp ---
    truoc = len(cap_loc)
    da_gap: set[tuple[str, str]] = set()
    cap_khong_trung: list[tuple[str, str]] = []
    for cap in cap_loc:
        if cap not in da_gap:
            da_gap.add(cap)
            cap_khong_trung.append(cap)
    bi_loai = truoc - len(cap_khong_trung)
    bang_thong_ke.append(ThongKeLoc("bo_trung", truoc, bi_loai, len(cap_khong_trung)))
    cap_loc = cap_khong_trung

    # --- Bước 4: Bỏ câu quá dài ---
    truoc = len(cap_loc)
    cap_loc = [
        (en, vi) for en, vi in cap_loc
        if len(en.split()) <= do_dai_toi_da and len(vi.split()) <= do_dai_toi_da
    ]
    bi_loai = truoc - len(cap_loc)
    bang_thong_ke.append(ThongKeLoc("bo_qua_dai", truoc, bi_loai, len(cap_loc)))

    # --- Bước 5: Bỏ cặp lệch độ dài quá ti_le_lech_toi_da lần ---
    truoc = len(cap_loc)
    cap_loc_moi: list[tuple[str, str]] = []
    for en, vi in cap_loc:
        n_en = len(en.split())
        n_vi = len(vi.split())
        # Tránh chia cho 0: nếu một bên rỗng đã bị lọc ở bước 2
        min_len = min(n_en, n_vi)
        max_len = max(n_en, n_vi)
        if min_len > 0 and max_len / min_len <= ti_le_lech_toi_da:
            cap_loc_moi.append((en, vi))
    bi_loai = truoc - len(cap_loc_moi)
    bang_thong_ke.append(ThongKeLoc("bo_lech_do_dai", truoc, bi_loai, len(cap_loc_moi)))
    cap_loc = cap_loc_moi

    # Tách lại thành 2 list
    ket_qua_en = [en for en, _ in cap_loc]
    ket_qua_vi = [vi for _, vi in cap_loc]

    return ket_qua_en, ket_qua_vi, bang_thong_ke


def kiem_tra_ro_ri(
    train_en: list[str],
    train_vi: list[str],
    test_en: list[str],
    test_vi: list[str],
) -> list[int]:
    """Trả về chỉ số các câu train bị trùng với test (so khớp theo CẶP EN+VI).

    So khớp theo CẶP (en, vi) để tránh false positive:
    câu EN ngắn như "Thank you." có thể xuất hiện nhiều lần với nhiều bản dịch
    VI khác nhau — chỉ loại khi cả hai vế đều trùng.

    Con số mong muốn là 0. Khác 0 thì phải loại các câu đó khỏi tập train
    RỒI GHI LẠI CON SỐ VÀO BÁO CÁO — đừng lặng lẽ xóa.
    """
    # Tạo tập hợp các cặp test để tra cứu O(1)
    tap_test: set[tuple[str, str]] = set(zip(test_en, test_vi))

    # Tìm chỉ số train bị trùng
    chi_so_bi_trung: list[int] = [
        i for i, (en, vi) in enumerate(zip(train_en, train_vi))
        if (en, vi) in tap_test
    ]

    return chi_so_bi_trung
