"""Bài test canh giữ quy tắc cốt lõi của đồ án — TASK 01 (Phú).

Cả đề tài dựa trên một lời hứa: "toàn bộ kiến trúc mạng do nhóm tự viết".
Nếu lúc bí ai đó lỡ gọi F.scaled_dot_product_attention cho nhanh rồi quên gỡ,
thì tới lúc bảo vệ mới phát hiện là quá muộn.

Bài test này quét toàn bộ src/ và bắt lỗi ngay khi commit.

Các lớp tham chiếu của PyTorch VẪN ĐƯỢC PHÉP dùng trong tests/ để đối chiếu số
(bài kiểm tra số 6) — thư mục tests/ không nằm trong phạm vi quét.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

CAM_DUNG = {
    "nn.Transformer": "phải tự viết Encoder-Decoder",
    "nn.TransformerEncoder": "phải tự viết Encoder",
    "nn.TransformerDecoder": "phải tự viết Decoder",
    "nn.TransformerEncoderLayer": "phải tự viết lớp Encoder",
    "nn.TransformerDecoderLayer": "phải tự viết lớp Decoder",
    "nn.MultiheadAttention": "phải tự viết Multi-Head Attention (TASK 05)",
    "scaled_dot_product_attention": "phải tự viết lõi attention (TASK 05)",
    "nn.LayerNorm": "phải tự viết LayerNorm (TASK 07)",
    "nn.RMSNorm": "phải tự viết RMSNorm (TASK 07)",
    "from transformers": "không dùng mô hình đã huấn luyện sẵn",
}


def _cac_file_python() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _bo_chu_thich_va_docstring(noi_dung: str) -> str:
    """Chỉ quét CODE THẬT. Docstring của cả nhóm nhắc tên các lớp bị cấm rất
    nhiều lần (để cảnh báo nhau), nên không loại ra thì test đỏ oan."""
    noi_dung = re.sub(r'"""(?:.|\n)*?"""', "", noi_dung)
    noi_dung = re.sub(r"'''(?:.|\n)*?'''", "", noi_dung)
    noi_dung = re.sub(r"#.*", "", noi_dung)
    return noi_dung


@pytest.mark.parametrize("duong_dan", _cac_file_python(), ids=lambda p: p.name)
def test_src_khong_dung_lop_dung_san(duong_dan: Path):
    ma_nguon = _bo_chu_thich_va_docstring(duong_dan.read_text(encoding="utf-8"))

    vi_pham = [
        f"{cam} ({ly_do})" for cam, ly_do in CAM_DUNG.items() if cam in ma_nguon
    ]

    assert not vi_pham, (
        f"{duong_dan.relative_to(SRC)} dùng lớp bị cấm trong src/: {vi_pham}\n"
        "Các lớp này chỉ được phép xuất hiện trong tests/ để đối chiếu số."
    )


def test_khong_co_token_lo_trong_ma_nguon():
    """Chặn việc lỡ dán Hugging Face token vào code rồi commit lên GitHub."""
    mau_token = re.compile(r"hf_[A-Za-z0-9]{30,}")

    for duong_dan in _cac_file_python():
        noi_dung = duong_dan.read_text(encoding="utf-8")
        assert not mau_token.search(noi_dung), (
            f"Phát hiện Hugging Face token trong {duong_dan}! "
            "Gỡ ngay, thu hồi token trên Hugging Face, rồi tạo token mới."
        )
