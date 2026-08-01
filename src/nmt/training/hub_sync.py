"""TASK 01 + TASK 12 — Đồng bộ Hugging Face Hub.  Người làm: Quân.  [Training Infra • Bắt buộc]

Nhóm dùng bản free trên Kaggle, đôi khi rớt mạng rồi F5 lại là mất sạch
checkpoint, nên phải đẩy checkpoint qua Hugging Face.

BẢO MẬT — đọc token theo đúng thứ tự này, KHÔNG viết thẳng token vào code:
    1. Kaggle Secrets (Add-ons -> Secrets), tên biến HF_TOKEN
    2. Biến môi trường HF_TOKEN (khi chạy ở máy cá nhân)
    3. Không có thì báo lỗi rõ ràng, đừng chạy tiếp rồi hỏng ở bước cuối

Giữ hai bản trên Hub: bản mới nhất để chạy tiếp, bản tốt nhất theo loss dev.
Xóa bớt bản cũ để repo không phình to.
Đẩy CẢ thư mục log lên cùng, nếu không thì mất log mỗi lần kernel chết và
không vẽ được đường loss liền mạch cho thí nghiệm giết phiên (TASK 14).
"""

from __future__ import annotations

import os


def doc_token() -> str:
    """Kaggle Secrets -> biến môi trường -> báo lỗi."""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    raise RuntimeError(
        "Không tìm thấy Hugging Face token.\n"
        "  - Trên Kaggle: Add-ons -> Secrets -> thêm HF_TOKEN (token loại Write)\n"
        "  - Trên máy cá nhân: đặt biến môi trường HF_TOKEN\n"
        "TUYỆT ĐỐI không viết token thẳng vào code hay commit lên git."
    )


def day_len_hub(duong_dan_file, repo_id: str, ten_tren_hub: str, so_lan_thu: int = 3) -> None:
    """Đẩy file lên Hub, có cơ chế thử lại khi mạng lỗi."""
    raise NotImplementedError("TASK 12 — Quân")


def tai_checkpoint_moi_nhat(repo_id: str, thu_muc_luu):
    """Trả về đường dẫn checkpoint mới nhất, hoặc None nếu Hub chưa có gì."""
    raise NotImplementedError("TASK 12 — Quân")
