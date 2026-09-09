"""TASK 19 — Kiểm tra Beam Search và KV Cache.  Người làm: Phú.

Ba bài test đầu do nhóm trưởng đặt khung sẵn. Ba bài sau là của KV cache, thêm
vào vì cache mở ra một lớp lỗi mới: cache chỉ đổi TỐC ĐỘ, nên mọi sai sót trong
đó đều biểu hiện thành "chạy được nhưng dịch sai" chứ không thành exception.
"""

from __future__ import annotations

import pytest
import torch

BOS, EOS, PAD = 2, 3, 0


def _cfg_nho(cfg_goc, **ghi_de):
    """Model tí hon để test chạy trong vài giây thay vì vài phút.

    Dropout ép về 0: có dropout thì hai lượt chạy cùng đầu vào vẫn ra khác nhau,
    và bài test "hai chế độ phải trùng khớp" sẽ đỏ vì lý do không liên quan.
    """
    cfg = cfg_goc
    cfg["du_lieu"]["vocab_size"] = 64
    cfg["mo_hinh"]["d_model"] = 32
    cfg["mo_hinh"]["so_head"] = 2
    cfg["mo_hinh"]["so_lop_encoder"] = 2
    cfg["mo_hinh"]["so_lop_decoder"] = 2
    cfg["mo_hinh"]["d_ff"] = 64
    cfg["mo_hinh"]["dropout"] = 0.0
    for khoa, gia_tri in ghi_de.items():
        cfg["mo_hinh"][khoa] = gia_tri
    return cfg


def _dung_model(cfg):
    from nmt.model.transformer import TransformerNMT

    torch.manual_seed(42)
    model = TransformerNMT(cfg)
    model.eval()          # tắt dropout, nếu quên thì mọi so sánh dưới đây vô nghĩa
    return model


def _nguon(batch: int = 3, do_dai: int = 7, vocab: int = 64):
    torch.manual_seed(7)
    src_ids = torch.randint(4, vocab, (batch, do_dai))
    src_mask = torch.ones(batch, 1, 1, do_dai, dtype=torch.bool)
    return src_ids, src_mask


def _cat_tai_eos(chuoi: torch.Tensor) -> list[int]:
    """Bỏ bos ở đầu và cắt tại eos đầu tiên.

    Greedy đệm eos cho tới hết do_dai_toi_da còn beam dừng ngay khi xong, nên hai
    hàm trả về tensor dài khác nhau dù NỘI DUNG giống hệt. So thẳng tensor sẽ đỏ
    vì phần đệm chứ không phải vì bản dịch khác nhau.
    """
    ids = chuoi.tolist()
    if ids and ids[0] == BOS:
        ids = ids[1:]
    return ids[: ids.index(EOS)] if EOS in ids else ids


# ===================================================================== khung sẵn


def test_beam_bang_1_trung_khop_greedy(cfg_goc):
    """Đặt beam = 1 thì Beam Search phải cho ra kết quả TRÙNG KHỚP TỪNG CHỮ với
    Greedy Search.

    Đây là bài test rẻ nhất nhưng bắt được hầu hết lỗi beam search: sai chỗ cộng
    log xác suất, quên xử lý câu đã kết thúc, sai thứ tự khi chọn top-k.
    """
    from nmt.inference.search import beam_search, greedy_search

    model = _dung_model(_cfg_nho(cfg_goc))
    src_ids, src_mask = _nguon()

    kq_greedy = greedy_search(model, src_ids, src_mask, BOS, EOS, do_dai_toi_da=20)
    kq_beam = beam_search(model, src_ids, src_mask, BOS, EOS,
                          beam_size=1, he_so_phat_do_dai=1.0, do_dai_toi_da=20)

    for i in range(src_ids.size(0)):
        assert _cat_tai_eos(kq_greedy[i]) == _cat_tai_eos(kq_beam[i]), (
            f"Câu {i}: beam=1 phải trùng greedy.\n"
            f"  greedy: {_cat_tai_eos(kq_greedy[i])}\n"
            f"  beam  : {_cat_tai_eos(kq_beam[i])}"
        )


def test_beam_dung_lai_khi_gap_eos(cfg_goc):
    """Câu đã sinh ra <eos> thì không được sinh thêm gì nữa, và không được
    chiếm chỗ của các phương án còn sống."""
    from nmt.inference.search import beam_search

    model = _dung_model(_cfg_nho(cfg_goc))
    src_ids, src_mask = _nguon()

    kq = beam_search(model, src_ids, src_mask, BOS, EOS,
                     beam_size=4, do_dai_toi_da=20)

    for i in range(src_ids.size(0)):
        ids = kq[i].tolist()
        if EOS not in ids:
            continue
        # Sau eos đầu tiên chỉ được còn eos (phần đệm), không được có token thật.
        sau_eos = ids[ids.index(EOS) + 1:]
        assert all(t == EOS for t in sau_eos), (
            f"Câu {i} còn sinh thêm token sau <eos>: {sau_eos}"
        )


def test_he_so_phat_do_dai_co_tac_dung():
    """Tăng hệ số phạt độ dài thì câu dịch ra phải dài hơn (trung bình).
    Không có phần này thì beam thiên vị câu ngắn vì log xác suất cộng dồn càng
    dài càng âm.

    Kiểm bằng chính phép toán thay vì chạy model: mô hình chưa huấn luyện sinh
    câu gần như ngẫu nhiên, nên "trung bình dài hơn" có thể đúng lần này sai lần
    sau. Kiểm tính chất toán học thì kết luận chắc chắn và không bao giờ chập chờn.
    """
    from nmt.inference.search import _phat_do_dai

    # he_so = 0 nghĩa là tắt phạt: mọi độ dài đều chia cho 1.
    assert _phat_do_dai(5, 0.0) == pytest.approx(1.0)
    assert _phat_do_dai(50, 0.0) == pytest.approx(1.0)

    # he_so > 0: hệ số phải TĂNG theo độ dài.
    assert _phat_do_dai(20, 1.0) > _phat_do_dai(5, 1.0)
    assert _phat_do_dai(50, 2.0) > _phat_do_dai(50, 1.0)

    # Hệ quả thực chất: với hai phương án cùng log-xác suất thô, phương án DÀI
    # HƠN phải được điểm cao hơn sau khi chia cho hệ số phạt. Log-xác suất luôn
    # âm nên chia cho số lớn hơn cho ra số LỚN hơn (gần 0 hơn).
    log_xac_suat = -12.0
    diem_ngan = log_xac_suat / _phat_do_dai(4, 1.0)
    diem_dai = log_xac_suat / _phat_do_dai(18, 1.0)
    assert diem_dai > diem_ngan


# ======================================================================= KV cache


@pytest.mark.parametrize("ma_hoa_vi_tri", ["rope", "sinusoidal"])
def test_kv_cache_khong_lam_doi_ket_qua_greedy(cfg_goc, ma_hoa_vi_tri):
    """Bật và tắt KV cache phải cho ra CÙNG một chuỗi token.

    Chạy cho cả hai kiểu mã hóa vị trí vì sin-cos có một cái bẫy riêng: mỗi bước
    có cache chỉ đưa vào MỘT token, nên nếu không truyền vi_tri_bat_dau xuống thì
    mọi token đều bị cộng vector vị trí 0. Mô hình mất sạch thứ tự và dịch ra chữ
    lộn xộn — mà không có lỗi nào được ném. RoPE không dính vì nó vốn nhận
    vi_tri_bat_dau từ đầu.
    """
    from nmt.inference.search import greedy_search

    model = _dung_model(_cfg_nho(cfg_goc, ma_hoa_vi_tri=ma_hoa_vi_tri))
    src_ids, src_mask = _nguon()

    khong_cache = greedy_search(model, src_ids, src_mask, BOS, EOS,
                                do_dai_toi_da=20, dung_kv_cache=False)
    co_cache = greedy_search(model, src_ids, src_mask, BOS, EOS,
                             do_dai_toi_da=20, dung_kv_cache=True)

    assert torch.equal(khong_cache, co_cache), (
        f"KV cache làm đổi kết quả với ma_hoa_vi_tri={ma_hoa_vi_tri}. "
        "Cache chỉ được đổi tốc độ, không được đổi bản dịch."
    )


@pytest.mark.parametrize("ma_hoa_vi_tri", ["rope", "sinusoidal"])
def test_kv_cache_khong_lam_doi_ket_qua_beam(cfg_goc, ma_hoa_vi_tri):
    """Như trên nhưng cho beam search.

    Beam có thêm một đường hỏng riêng: sau mỗi bước thứ tự các beam bị xáo, nên
    cache phải được xáo theo cùng chỉ số. Quên xáo thì lịch sử của beam này bị
    gán cho beam kia và câu dịch là bản ghép của hai lịch sử khác nhau.
    """
    from nmt.inference.search import beam_search

    model = _dung_model(_cfg_nho(cfg_goc, ma_hoa_vi_tri=ma_hoa_vi_tri))
    src_ids, src_mask = _nguon()

    khong_cache = beam_search(model, src_ids, src_mask, BOS, EOS, beam_size=3,
                              do_dai_toi_da=20, dung_kv_cache=False)
    co_cache = beam_search(model, src_ids, src_mask, BOS, EOS, beam_size=3,
                           do_dai_toi_da=20, dung_kv_cache=True)

    for i in range(src_ids.size(0)):
        assert _cat_tai_eos(khong_cache[i]) == _cat_tai_eos(co_cache[i]), (
            f"Câu {i}: cache làm đổi kết quả beam (ma_hoa_vi_tri={ma_hoa_vi_tri})"
        )


def test_beam_tra_ve_du_so_cau(cfg_goc):
    """Số câu trả về phải bằng số câu đưa vào, kể cả khi có câu chạy hết độ dài
    tối đa mà chưa gặp eos.

    Thiếu một câu thì sacrebleu vẫn chấm được và vẫn ra một con số trông hợp lý,
    nhưng là điểm của hai tập lệch nhau. Đúng loại lỗi mà _kiem_dau_vao chặn.
    """
    from nmt.inference.search import beam_search

    model = _dung_model(_cfg_nho(cfg_goc))
    src_ids, src_mask = _nguon(batch=5)

    # do_dai_toi_da rất ngắn để ép một số câu chưa kịp sinh eos
    kq = beam_search(model, src_ids, src_mask, BOS, EOS, beam_size=4, do_dai_toi_da=3)

    assert kq.size(0) == 5
    assert kq.dtype == torch.long
    assert (kq[:, 0] == BOS).all(), "Mọi câu trả về đều phải bắt đầu bằng <bos>"


@pytest.mark.parametrize("ma_hoa_vi_tri", ["rope", "sinusoidal"])
def test_bang_vi_tri_du_dai_cho_chuoi_dai_nhat(cfg_goc, ma_hoa_vi_tri):
    """Bảng vị trí phải đủ chỗ cho chuỗi DÀI NHẤT thật sự chạy qua mô hình.

    Hai chỗ vượt quá du_lieu.do_dai_toi_da:
      huấn luyện : chuỗi đích là BOS + do_dai_toi_da token = do_dai_toi_da + 1
      sinh câu   : chạy tới sinh_cau.do_dai_toi_da_khi_dich, mặc định 128

    Bản cũ dựng bảng đúng bằng do_dai_toi_da nên lần đánh giá đầu tiên đã chết:
        ValueError: vị trí 101 vượt do_dai_toi_da đã cache (100)

    Lỗi nằm im suốt từ đầu dự án vì lượt chạy chính dùng RoPE, mà nhánh RoPE
    không cộng bảng vị trí ở ngoài nên không chạm giới hạn. Nó chỉ nổ khi A0 và
    A4 chạy — hai cấu hình sin-cos đầu tiên — và nổ sau 40 phút smoke test trên
    Kaggle. Bài test này chạy trong vài giây và bắt đúng nó.
    """
    from nmt.model.masking import tao_causal_mask

    cfg = _cfg_nho(cfg_goc, ma_hoa_vi_tri=ma_hoa_vi_tri)
    model = _dung_model(cfg)

    dai_nhat = max(cfg.du_lieu.do_dai_toi_da + 1, cfg.sinh_cau.do_dai_toi_da_khi_dich)
    src = torch.randint(4, 64, (2, cfg.du_lieu.do_dai_toi_da))
    tgt = torch.randint(4, 64, (2, dai_nhat))
    src_mask = torch.ones(2, 1, 1, src.size(1), dtype=torch.bool)

    with torch.no_grad():
        logits = model(src, tgt, src_mask, tao_causal_mask(tgt.size(1)))

    assert logits.shape[:2] == (2, dai_nhat)


def test_beam_size_khong_hop_le_thi_bao_loi(cfg_goc):
    """beam_size = 0 phải ném lỗi ngay chứ không lặng lẽ trả về tensor rỗng."""
    from nmt.inference.search import beam_search

    model = _dung_model(_cfg_nho(cfg_goc))
    src_ids, src_mask = _nguon()

    with pytest.raises(ValueError, match="beam_size"):
        beam_search(model, src_ids, src_mask, BOS, EOS, beam_size=0)
