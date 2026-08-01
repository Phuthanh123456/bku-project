"""TASK 12 + TASK 14 — Kiểm tra checkpoint và phục hồi.  Người làm: Quân.

TASK 14 xong khi: hai đường loss chênh nhau dưới 1 phần trăm tại cùng một bước.

Hình hai đường loss chồng lên nhau của thí nghiệm giết phiên là HÌNH CÓ GIÁ TRỊ
NHẤT của cả đồ án — dùng cho báo cáo, slide và cả CV — vì nó chứng minh cơ chế
phục hồi hoạt động thật chứ không chỉ nói suông.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="TASK 12 — Quân. Bỏ skip khi checkpoint.py đã xong.")


def test_luu_roi_nap_lai_ra_dung_trong_so():
    """Lưu rồi nạp lại, mọi tensor trọng số phải giống hệt từng phần tử."""
    raise NotImplementedError


def test_nap_lai_du_trang_thai_optimizer_va_scheduler():
    """Thiếu trạng thái optimizer thì momentum của AdamW về 0 và loss giật lên
    ngay sau khi resume — nhìn đường loss là thấy một vết gãy.
    Thiếu trạng thái scheduler thì learning rate nhảy về đầu."""
    raise NotImplementedError


def test_ghi_an_toan_khong_lam_hong_checkpoint_cu():
    """Mô phỏng bị ngắt giữa lúc đang ghi. Checkpoint CŨ phải còn nguyên vẹn và
    nạp lại được. Đây là lý do phải ghi ra file tạm rồi mới đổi tên."""
    raise NotImplementedError


@pytest.mark.slow
def test_giet_phien_roi_resume_ra_cung_duong_loss():
    """TASK 14. Chạy N bước liên tục, ghi lại loss từng bước.
    Chạy lại: dừng ở bước N/3 và 2N/3, mỗi lần nạp checkpoint rồi chạy tiếp.
    Loss tại cùng một bước phải chênh nhau dưới 1 phần trăm.

    Muốn hai đường trùng khít thì checkpoint phải lưu cả trạng thái RNG và vị
    trí trong DataLoader, nếu không thì sau khi resume mô hình gặp thứ tự dữ
    liệu khác và hai đường tách nhau dần.
    """
    raise NotImplementedError
