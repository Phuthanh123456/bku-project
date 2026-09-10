"""TASK 16 — Greedy Search.       Người làm: My.   [Evaluation • Bắt buộc]
TASK 19 — Beam Search & KV Cache. Người làm: Phú.  [Inference • ƯU TIÊN THẤP]

Theo nhận xét 6 của mentor: Beam Search và KV Cache chưa cần vội, ưu tiên chạy
được model trước. Chấm điểm baseline ở TASK 16 dùng Greedy.
TASK 19 là task ĐẦU TIÊN BỊ CẮT nếu trễ tiến độ.

Bài test rẻ nhất mà bắt được hầu hết lỗi beam search:
    đặt beam = 1 thì Beam Search phải cho kết quả TRÙNG KHỚP TỪNG CHỮ với Greedy.
    Nằm ở tests/test_beam_search.py.

Yêu cầu của TASK 19: Beam phải cho BLEU cao hơn Greedy ít nhất 0,5 điểm.
Không cao hơn thì gần như chắc chắn Beam Search đang có lỗi, phải kiểm tra lại
chứ đừng ghi vào báo cáo là "beam không hiệu quả".

KV CACHE — vì sao nhanh hơn. Không cache thì ở bước thứ t, decoder nhận vào cả
tiền tố dài t và tính lại key/value cho toàn bộ t vị trí, dù t-1 vị trí đầu
không hề đổi. Sinh câu dài n token tốn O(n^2) lượt chiếu. Có cache thì mỗi bước
chỉ tính cho đúng token mới: O(n). Chi tiết cách cài nằm ở MultiHeadAttention.

BẪY LỚN NHẤT CỦA KV CACHE. Mỗi bước chỉ đưa vào MỘT token, nên nếu mã hóa vị trí
tuyệt đối (sin-cos) không được cho biết đang ở vị trí nào thì nó luôn cộng vector
vị trí 0. Mô hình mất sạch thông tin thứ tự và dịch ra chữ lộn xộn, mà không có
lỗi nào được ném ra. Vì vậy vi_tri_bat_dau phải đi suốt từ đây xuống tận
MaHoaViTriSinCos. Bài test kiểm đúng chuyện này: bật cache và tắt cache phải cho
ra CÙNG một chuỗi token.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _sap_xep_lai_cache(cache: dict, chi_so: torch.Tensor) -> None:
    """Hoán vị chiều batch của mọi tensor trong cache theo chi_so, sửa tại chỗ.

    Beam search xáo trộn thứ tự các phương án sau mỗi bước: phương án tốt nhất ở
    bước này có thể sinh ra từ beam số 3 của bước trước. Cache phải được xáo theo
    cùng thứ tự đó, nếu không thì lịch sử của beam A bị gán cho beam B — mô hình
    vẫn chạy và vẫn ra câu, nhưng là câu ghép từ hai lịch sử khác nhau.
    """
    for kho_lop in cache.values():
        for ngan in kho_lop.values():          # "self" và "cross"
            for ten in ("k", "v"):
                if ten in ngan:
                    ngan[ten] = ngan[ten].index_select(0, chi_so)


def _phat_do_dai(do_dai: int, he_so: float) -> float:
    """Hệ số phạt độ dài kiểu GNMT: ((5 + |Y|) / 6) ^ he_so.

    Không có nó thì beam search thiên vị câu ngắn một cách có hệ thống, vì mỗi
    token thêm vào đều cộng thêm một log-xác suất âm. Chia điểm cho hệ số này để
    câu dài không bị phạt oan. he_so = 0 nghĩa là tắt phạt.
    """
    return ((5.0 + do_dai) / 6.0) ** he_so


@torch.no_grad()
def greedy_search(model, src_ids, src_mask, bos_id: int, eos_id: int,
                  do_dai_toi_da: int = 128, dung_kv_cache: bool = False):
    """Mỗi bước chọn luôn từ có xác suất cao nhất.

    dung_kv_cache chỉ đổi TỐC ĐỘ, không được đổi kết quả. Bài test so hai chế độ
    phải ra cùng một chuỗi token.
    """
    from nmt.model.masking import tao_causal_mask

    batch_size = src_ids.size(0)
    device = src_ids.device

    # Tiền tính toán bo_nho_encoder (chỉ làm 1 lần)
    bo_nho_encoder = model.encode(src_ids, src_mask)

    # Khởi tạo chuỗi đích với bos_id
    tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    # Đánh dấu các sequence đã sinh ra eos_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    cache: dict | None = {} if dung_kv_cache else None

    for _ in range(do_dai_toi_da):
        if cache is None:
            # Không cache: đưa cả tiền tố vào, cần causal mask để chặn nhìn trước.
            dau_vao = tgt_ids
            vi_tri_bat_dau = 0
            tgt_mask = tao_causal_mask(tgt_ids.size(1), device=device)
        else:
            # Có cache: chỉ đưa token mới nhất. Không cần causal mask vì trong
            # cache chỉ có quá khứ — tính nhân quả được bảo đảm bởi cấu trúc.
            dau_vao = tgt_ids[:, -1:]
            vi_tri_bat_dau = tgt_ids.size(1) - 1
            tgt_mask = None

        decoder_output = model.decode(dau_vao, bo_nho_encoder, tgt_mask, src_mask,
                                      vi_tri_bat_dau=vi_tri_bat_dau, cache=cache)
        logits = model.output_projection(decoder_output)

        # Lấy logits của token cuối cùng vừa được sinh ra
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Câu nào đã xong thì ghi eos_id để cache và độ dài vẫn thẳng hàng
        next_tokens = next_tokens.masked_fill(finished, eos_id)

        tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(-1)], dim=-1)
        finished |= (next_tokens == eos_id)

        if finished.all():
            break

    return tgt_ids


@torch.no_grad()
def beam_search(
    model, src_ids, src_mask, bos_id: int, eos_id: int,
    beam_size: int = 4, he_so_phat_do_dai: float = 1.0, do_dai_toi_da: int = 128,
    dung_kv_cache: bool = True,
):
    """Giữ lại beam_size phương án tốt nhất ở mỗi bước.

    Kèm hệ số phạt độ dài để mô hình không thiên vị câu quá ngắn.

    Trả về (batch, len) giống hệt greedy_search để hai hàm thay thế được cho
    nhau trong evaluate.py. Câu ngắn hơn được đệm bằng eos_id.
    """
    from nmt.model.masking import tao_causal_mask

    if beam_size < 1:
        raise ValueError(f"beam_size phải >= 1, nhận được {beam_size}")

    batch_size = src_ids.size(0)
    device = src_ids.device
    B = batch_size * beam_size

    bo_nho_encoder = model.encode(src_ids, src_mask)

    # Nhân bản nguồn cho từng beam. repeat_interleave chứ KHÔNG phải repeat:
    # ta cần [câu0, câu0, câu0, câu0, câu1, câu1, ...] để beam của cùng một câu
    # nằm liền nhau, nhờ vậy view(batch, beam, ...) ở dưới mới đúng nhóm.
    bo_nho_encoder = bo_nho_encoder.repeat_interleave(beam_size, dim=0)
    src_mask_beam = (src_mask.repeat_interleave(beam_size, dim=0)
                     if src_mask is not None else None)

    tgt_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    # Điểm tích lũy (log-xác suất). Bước đầu chỉ cho beam 0 của mỗi câu được mở
    # rộng; nếu để cả beam_size beam cùng điểm 0 thì chúng sinh ra beam_size
    # phương án GIỐNG HỆT nhau và beam search suy biến thành greedy.
    diem = torch.full((batch_size, beam_size), float("-inf"), device=device)
    diem[:, 0] = 0.0
    diem = diem.view(-1)                                    # (B,)

    cache: dict | None = {} if dung_kv_cache else None

    # Kho câu đã hoàn tất, mỗi câu nguồn một danh sách (điểm đã phạt, chuỗi token)
    da_xong: list[list[tuple[float, torch.Tensor]]] = [[] for _ in range(batch_size)]

    for _ in range(do_dai_toi_da):
        if cache is None:
            dau_vao, vi_tri_bat_dau = tgt_ids, 0
            tgt_mask = tao_causal_mask(tgt_ids.size(1), device=device)
        else:
            dau_vao = tgt_ids[:, -1:]
            vi_tri_bat_dau = tgt_ids.size(1) - 1
            tgt_mask = None

        decoder_output = model.decode(dau_vao, bo_nho_encoder, tgt_mask, src_mask_beam,
                                      vi_tri_bat_dau=vi_tri_bat_dau, cache=cache)
        logits = model.output_projection(decoder_output)[:, -1, :]      # (B, vocab)
        log_xac_suat = F.log_softmax(logits.float(), dim=-1)             # (B, vocab)
        vocab_size = log_xac_suat.size(-1)

        # Cộng dồn log-xác suất: điểm của câu là TỔNG chứ không phải tích, vì đã
        # ở thang log. Beam có điểm -inf (chưa kích hoạt / đã xong) vẫn giữ -inf.
        diem_moi = diem.unsqueeze(-1) + log_xac_suat                     # (B, vocab)
        diem_moi = diem_moi.view(batch_size, beam_size * vocab_size)

        # Chọn beam_size phương án tốt nhất trong TOÀN BỘ tổ hợp (beam x vocab)
        diem_top, chi_so_phang = diem_moi.topk(beam_size, dim=-1)        # (batch, beam)
        beam_nguon = torch.div(chi_so_phang, vocab_size, rounding_mode="floor")
        token_moi = chi_so_phang % vocab_size

        # Chỉ số tuyệt đối trong batch phẳng, để xáo lại chuỗi và cache
        offset = torch.arange(batch_size, device=device).unsqueeze(-1) * beam_size
        chi_so_tuyet_doi = (beam_nguon + offset).view(-1)                # (B,)

        tgt_ids = torch.cat(
            [tgt_ids.index_select(0, chi_so_tuyet_doi), token_moi.view(-1, 1)], dim=-1
        )
        if cache is not None:
            _sap_xep_lai_cache(cache, chi_so_tuyet_doi)
        diem = diem_top.reshape(-1)

        # Tách ra những beam vừa sinh eos: chúng KHÔNG được mở rộng tiếp nữa.
        vua_xong = (token_moi.reshape(-1) == eos_id) & torch.isfinite(diem)
        if vua_xong.any():
            do_dai = tgt_ids.size(1) - 1        # trừ bos
            phat = _phat_do_dai(do_dai, he_so_phat_do_dai)
            for vi_tri in vua_xong.nonzero(as_tuple=False).flatten().tolist():
                cau = vi_tri // beam_size
                da_xong[cau].append((diem[vi_tri].item() / phat, tgt_ids[vi_tri].clone()))
            # Ép điểm về -inf để beam đã xong không chiếm chỗ ở bước sau
            diem = diem.masked_fill(vua_xong, float("-inf"))

        # Dừng khi mọi câu đều đã đủ beam_size phương án hoàn tất
        if all(len(ds) >= beam_size for ds in da_xong):
            break
        if not torch.isfinite(diem).any():
            break

    # ----------------------------------------------------------------- chọn ra
    # Câu nào chạy hết do_dai_toi_da mà chưa gặp eos thì vẫn phải có kết quả,
    # nếu không thì batch bị thiếu câu và sacrebleu chấm hai tập lệch nhau —
    # lỗi đó vẫn ra một con số BLEU trông bình thường. Xem _kiem_dau_vao.
    ket_qua: list[torch.Tensor] = []
    for cau in range(batch_size):
        if da_xong[cau]:
            ket_qua.append(max(da_xong[cau], key=lambda x: x[0])[1])
        else:
            khoi = diem.view(batch_size, beam_size)[cau]
            tot_nhat = int(torch.argmax(torch.nan_to_num(khoi, neginf=-1e30)).item())
            ket_qua.append(tgt_ids[cau * beam_size + tot_nhat].clone())

    dai_nhat = max(c.size(0) for c in ket_qua)
    dau_ra = torch.full((batch_size, dai_nhat), eos_id, dtype=torch.long, device=device)
    for i, c in enumerate(ket_qua):
        dau_ra[i, : c.size(0)] = c
    return dau_ra
