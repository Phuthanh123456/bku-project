"""TASK 01 — Bằng chứng cho tiêu chí "chạy lại cùng seed cho ra kết quả giống hệt".

Bài test này ĐÃ CHẠY ĐƯỢC ngay từ đầu dự án. Nếu nó đỏ thì đừng tin bất kỳ
bảng ablation nào của TASK 17 và 18, vì lúc đó không phân biệt được chênh lệch
do đổi kiến trúc với chênh lệch do ngẫu nhiên.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nmt.utils import dat_seed, nap_config

GOC = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- seed
def test_cung_seed_cho_cung_so_ngau_nhien():
    import numpy as np
    import torch

    dat_seed(42)
    a_torch = torch.randn(10)
    a_numpy = np.random.rand(10)

    dat_seed(42)
    b_torch = torch.randn(10)
    b_numpy = np.random.rand(10)

    assert torch.equal(a_torch, b_torch), "torch không tái lập được với cùng seed"
    assert (a_numpy == b_numpy).all(), "numpy không tái lập được với cùng seed"


def test_khac_seed_cho_khac_so_ngau_nhien():
    """Kiểm ngược lại: dat_seed thực sự có tác dụng chứ không phải luôn trả về
    cùng một thứ vì lý do nào khác."""
    import torch

    dat_seed(42)
    a = torch.randn(10)
    dat_seed(1337)
    b = torch.randn(10)

    assert not torch.equal(a, b)


def test_khoi_tao_lop_linear_tai_lap_duoc():
    """Quan trọng hơn test trên: trọng số khởi tạo của mạng phải giống hệt,
    vì đó mới là thứ quyết định đường loss."""
    import torch
    import torch.nn as nn

    dat_seed(42)
    w1 = nn.Linear(512, 688).weight.detach().clone()
    dat_seed(42)
    w2 = nn.Linear(512, 688).weight.detach().clone()

    assert torch.equal(w1, w2)


# --------------------------------------------------------------- cấu hình
def test_nap_config_goc():
    cfg = nap_config(GOC / "configs" / "base.yaml")

    assert cfg.mo_hinh.d_model == 512
    assert cfg.mo_hinh.d_model % cfg.mo_hinh.so_head == 0
    assert cfg.thi_nghiem.seed == 42


def test_mac_dinh_theo_nhan_xet_mentor():
    """Nhận xét 1 của mentor: Warmup và Label Smoothing KHÔNG phải bắt buộc,
    mặc định phải TẮT. Giữ hay bỏ do TASK 18 quyết định.

    Bài test này chặn việc ai đó lỡ bật lại trong base.yaml rồi cả nhóm chạy
    ablation trên một baseline sai.
    """
    cfg = nap_config(GOC / "configs" / "base.yaml")

    assert cfg.toi_uu.scheduler == "co_dinh", "Warmup phải TẮT trong base.yaml"
    assert cfg.toi_uu.label_smoothing == 0.0, "Label Smoothing phải TẮT trong base.yaml"


def test_bat_buoc_fp16_khong_bf16():
    """T4 của Kaggle là Turing (compute capability 7.5), không có phần cứng bf16."""
    cfg = nap_config(GOC / "configs" / "base.yaml")
    assert cfg.toi_uu.kieu_do_chinh_xac == "fp16"


def test_rope_khong_ap_cho_cross_attention():
    cfg = nap_config(GOC / "configs" / "base.yaml")
    assert cfg.mo_hinh.rope_ap_cho_cross_attention is False


def test_pre_norm_phai_kem_chuan_hoa_cuoi():
    cfg = nap_config(GOC / "configs" / "base.yaml")
    if cfg.mo_hinh.vi_tri_chuan_hoa == "pre":
        assert cfg.mo_hinh.co_chuan_hoa_cuoi is True


# --------------------------------------------------------------- kế thừa ablation
@pytest.mark.parametrize(
    "ten_file,khoa,gia_tri",
    [
        ("ablation_a1_layernorm.yaml", "mo_hinh.kieu_chuan_hoa", "layernorm"),
        ("ablation_a2_warmup.yaml", "toi_uu.scheduler", "warmup"),
        ("ablation_a3_label_smoothing.yaml", "toi_uu.label_smoothing", 0.1),
        ("ablation_a4_sincos.yaml", "mo_hinh.ma_hoa_vi_tri", "sinusoidal"),
        ("ablation_a5_relu.yaml", "mo_hinh.kieu_ffn", "relu"),
        ("ablation_a6_post_norm.yaml", "mo_hinh.vi_tri_chuan_hoa", "post"),
    ],
)
def test_file_ablation_ghi_de_dung_yeu_to(ten_file, khoa, gia_tri):
    cfg = nap_config(GOC / "configs" / ten_file)
    assert cfg.get_sau(khoa) == gia_tri


def test_ablation_ke_thua_phan_con_lai_tu_base():
    """Điểm mấu chốt: file A1 chỉ ghi kieu_chuan_hoa, mọi thứ khác phải giống
    hệt base. Nếu không thì đang so hai mô hình khác nhau ở nhiều yếu tố và
    ablation mất ý nghĩa."""
    base = nap_config(GOC / "configs" / "base.yaml")
    a1 = nap_config(GOC / "configs" / "ablation_a1_layernorm.yaml")

    assert a1.mo_hinh.d_model == base.mo_hinh.d_model
    assert a1.mo_hinh.so_head == base.mo_hinh.so_head
    assert a1.mo_hinh.dropout == base.mo_hinh.dropout
    assert a1.toi_uu.learning_rate == base.toi_uu.learning_rate
    assert a1.mo_hinh.kieu_chuan_hoa != base.mo_hinh.kieu_chuan_hoa


def test_a5_doi_ca_d_ff_de_giu_so_tham_so_tuong_duong():
    """SwiGLU 3 ma trận vs ReLU 2 ma trận. Không chỉnh d_ff theo thì đang so hai
    mô hình khác cỡ."""
    base = nap_config(GOC / "configs" / "base.yaml")
    a5 = nap_config(GOC / "configs" / "ablation_a5_relu.yaml")

    tham_so_swiglu = 3 * base.mo_hinh.d_model * base.mo_hinh.d_ff
    tham_so_relu = 2 * a5.mo_hinh.d_model * a5.mo_hinh.d_ff

    assert tham_so_swiglu == 1_056_768
    assert tham_so_relu == 1_048_576
    chenh_lech = abs(tham_so_swiglu - tham_so_relu) / tham_so_relu
    assert chenh_lech < 0.01, f"chênh {chenh_lech:.1%}, yêu cầu dưới 1%"


# --------------------------------------------------------------- chặn cấu hình sai
def test_chan_bf16(tmp_path):
    (tmp_path / "xau.yaml").write_text(
        "toi_uu:\n  kieu_do_chinh_xac: bf16\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="bf16"):
        nap_config(tmp_path / "xau.yaml")


def test_chan_pre_norm_thieu_chuan_hoa_cuoi(tmp_path):
    (tmp_path / "xau.yaml").write_text(
        "mo_hinh:\n  vi_tri_chuan_hoa: pre\n  co_chuan_hoa_cuoi: false\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="co_chuan_hoa_cuoi"):
        nap_config(tmp_path / "xau.yaml")


def test_chan_d_model_khong_chia_het_cho_so_head(tmp_path):
    (tmp_path / "xau.yaml").write_text(
        "mo_hinh:\n  d_model: 512\n  so_head: 7\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="chia hết"):
        nap_config(tmp_path / "xau.yaml")
