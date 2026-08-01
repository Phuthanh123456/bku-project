"""TASK 19 — Kiểm tra Beam Search.  Người làm: Phú.  [Inference • Ưu tiên thấp]

Hạ ưu tiên theo nhận xét 6 của mentor. Chỉ làm sau khi TASK 16 đã có điểm
baseline bằng Greedy. Đây là task ĐẦU TIÊN BỊ CẮT nếu trễ tiến độ.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="TASK 19 — Phú. Bỏ skip khi beam_search đã xong.")


def test_beam_bang_1_trung_khop_greedy():
    """Đặt beam = 1 thì Beam Search phải cho ra kết quả TRÙNG KHỚP TỪNG CHỮ với
    Greedy Search.

    Đây là bài test rẻ nhất nhưng bắt được hầu hết lỗi beam search: sai chỗ cộng
    log xác suất, quên xử lý câu đã kết thúc, sai thứ tự khi chọn top-k.
    """
    raise NotImplementedError


def test_beam_dung_lai_khi_gap_eos():
    """Câu đã sinh ra <eos> thì không được sinh thêm gì nữa, và không được
    chiếm chỗ của các phương án còn sống."""
    raise NotImplementedError


def test_he_so_phat_do_dai_co_tac_dung():
    """Tăng hệ số phạt độ dài thì câu dịch ra phải dài hơn (trung bình).
    Không có phần này thì beam thiên vị câu ngắn vì log xác suất cộng dồn càng
    dài càng âm."""
    raise NotImplementedError
