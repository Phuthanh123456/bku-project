"""Cấu hình chung cho pytest — TASK 01 (Phú)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Cho phép `import nmt` ngay cả khi chưa chạy `pip install -e .`
# Tiện khi chạy test trên Kaggle, ở đó cài package hơi phiền.
GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))


@pytest.fixture(scope="session")
def duong_dan_goc() -> Path:
    return GOC


@pytest.fixture
def cfg_goc():
    """Cấu hình base.yaml đã nạp sẵn."""
    from nmt.utils import nap_config

    return nap_config(GOC / "configs" / "base.yaml")


@pytest.fixture(autouse=True)
def _co_dinh_seed():
    """Mọi bài test đều chạy với seed cố định.

    Không có fixture này thì một bài test so sánh số có thể lúc xanh lúc đỏ,
    và cả nhóm sẽ mất buổi tối đi tìm lỗi không tồn tại.
    """
    try:
        from nmt.utils import dat_seed

        dat_seed(42)
    except ImportError:
        pass


def pytest_collection_modifyitems(config, items):
    """Tự skip các bài test cần GPU khi máy không có CUDA."""
    try:
        import torch

        co_gpu = torch.cuda.is_available()
    except ImportError:
        co_gpu = False

    if co_gpu:
        return

    bo_qua = pytest.mark.skip(reason="cần GPU, máy này không có CUDA")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(bo_qua)
