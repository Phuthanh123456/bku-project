"""Khóa protocol baseline mới để lần chạy sau không lệch ngân sách âm thầm."""

from pathlib import Path

from nmt.utils import nap_config


GOC = Path(__file__).resolve().parents[1]


def _flatten(nut, tien_to=""):
    ket_qua = {}
    for khoa, gia_tri in nut.items():
        duong_dan = f"{tien_to}.{khoa}" if tien_to else khoa
        if isinstance(gia_tri, dict):
            ket_qua.update(_flatten(gia_tri, duong_dan))
        else:
            ket_qua[duong_dan] = gia_tri
    return ket_qua


def test_hai_baseline_dung_cung_ngan_sach_co_dinh():
    cai_tien = nap_config(GOC / "configs" / "baseline_cai_tien_6000.yaml")
    vanilla = nap_config(GOC / "configs" / "baseline_vanilla_architecture_6000.yaml")

    for cfg in (cai_tien, vanilla):
        assert cfg.huan_luyen.so_buoc_toi_da == 6000
        assert cfg.huan_luyen.dung_som_sau > (
            cfg.huan_luyen.so_buoc_toi_da // cfg.huan_luyen.danh_gia_moi
        )

    assert vanilla.du_lieu == cai_tien.du_lieu
    assert vanilla.toi_uu == cai_tien.toi_uu


def test_baseline_vanilla_chi_doi_goi_kien_truc_da_cong_bo():
    cai_tien = nap_config(GOC / "configs" / "baseline_cai_tien_6000.yaml")
    vanilla = nap_config(GOC / "configs" / "baseline_vanilla_architecture_6000.yaml")
    phang_cai_tien = _flatten(cai_tien)
    phang_vanilla = _flatten(vanilla)

    khac_nhau = {
        khoa
        for khoa, gia_tri in phang_cai_tien.items()
        if phang_vanilla.get(khoa) != gia_tri
    }
    assert khac_nhau == {
        "thi_nghiem.ten",
        "mo_hinh.ma_hoa_vi_tri",
        "mo_hinh.vi_tri_chuan_hoa",
        "mo_hinh.co_chuan_hoa_cuoi",
        "mo_hinh.kieu_chuan_hoa",
        "mo_hinh.kieu_ffn",
        "mo_hinh.d_ff",
    }

    assert vanilla.mo_hinh.ma_hoa_vi_tri == "sinusoidal"
    assert vanilla.mo_hinh.vi_tri_chuan_hoa == "post"
    assert vanilla.mo_hinh.co_chuan_hoa_cuoi is False
    assert vanilla.mo_hinh.kieu_chuan_hoa == "layernorm"
    assert vanilla.mo_hinh.kieu_ffn == "relu"


def test_ffn_vanilla_va_swiglu_khop_quy_mo_duoi_mot_phan_tram():
    cai_tien = nap_config(GOC / "configs" / "baseline_cai_tien_6000.yaml")
    vanilla = nap_config(GOC / "configs" / "baseline_vanilla_architecture_6000.yaml")

    tham_so_swiglu = 3 * cai_tien.mo_hinh.d_model * cai_tien.mo_hinh.d_ff
    tham_so_relu = 2 * vanilla.mo_hinh.d_model * vanilla.mo_hinh.d_ff
    chenh_lech = abs(tham_so_swiglu - tham_so_relu) / tham_so_relu

    assert chenh_lech < 0.01


def test_moi_ablation_dung_cung_ngan_sach_6000_buoc():
    for ten in (
        "ablation_a1_layernorm.yaml",
        "ablation_a2_warmup.yaml",
        "ablation_a3_label_smoothing.yaml",
        "ablation_a4_sincos.yaml",
        "ablation_a5_relu.yaml",
        "ablation_a6_post_norm.yaml",
    ):
        cfg = nap_config(GOC / "configs" / ten)
        assert cfg.huan_luyen.so_buoc_toi_da == 6000, ten
        assert cfg.huan_luyen.dung_som_sau > (
            cfg.huan_luyen.so_buoc_toi_da // cfg.huan_luyen.danh_gia_moi
        ), ten
