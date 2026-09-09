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
"""

from __future__ import annotations

import torch


@torch.no_grad()
def greedy_search(
    model,
    src_ids,
    src_mask,
    bos_id: int,
    eos_id: int,
    do_dai_toi_da: int = 128,
    dung_kv_cache: bool = False,
):
    """Mỗi bước chọn luôn từ có xác suất cao nhất."""
    from nmt.model.masking import tao_causal_mask

    batch_size = src_ids.size(0)
    device = src_ids.device

    # Tiền tính toán bo_nho_encoder (chỉ làm 1 lần)
    bo_nho_encoder = model.encode(src_ids, src_mask)

    # Khởi tạo chuỗi đích với bos_id
    tgt_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)

    # Đánh dấu các sequence đã sinh ra eos_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    caches = None
    pad_id = getattr(model, "pad_id", eos_id)

    for _ in range(do_dai_toi_da):
        seq_len = tgt_ids.size(1)
        if dung_kv_cache:
            decoder_output, caches = model.decode_step(
                tgt_ids[:, -1:],
                bo_nho_encoder,
                src_mask,
                caches=caches,
                vi_tri_bat_dau=seq_len - 1,
            )
        else:
            # Tạo causal mask cho tgt_ids hiện tại
            tgt_mask = tao_causal_mask(seq_len, device=device)
            decoder_output = model.decode(tgt_ids, bo_nho_encoder, tgt_mask, src_mask)
        logits = model.output_projection(decoder_output)

        # Lấy logits của token cuối cùng vừa được sinh ra
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # Câu đã xong chỉ nối PAD; token EOS đầu tiên vẫn được giữ nguyên.
        next_tokens = next_tokens.masked_fill(finished, pad_id)

        # Gắn token mới vào tgt_ids
        tgt_ids = torch.cat([tgt_ids, next_tokens.unsqueeze(-1)], dim=-1)

        # Cập nhật trạng thái finished
        finished |= (next_tokens == eos_id)

        # Dừng sớm nếu tất cả sequence đều đã sinh ra eos_id
        if finished.all():
            break

    return tgt_ids


@torch.no_grad()
def beam_search(
    model, src_ids, src_mask, bos_id: int, eos_id: int,
    beam_size: int = 4, he_so_phat_do_dai: float = 1.0, do_dai_toi_da: int = 128,
    dung_kv_cache: bool = False,
):
    """Giữ lại beam_size phương án tốt nhất ở mỗi bước.

    Kèm hệ số phạt độ dài để mô hình không thiên vị câu quá ngắn.
    """
    if beam_size < 1:
        raise ValueError("beam_size phải lớn hơn hoặc bằng 1")
    if he_so_phat_do_dai < 0:
        raise ValueError("he_so_phat_do_dai không được âm")

    # Đây vừa là tối ưu vừa là giao kèo kiểm thử quan trọng nhất của TASK 19.
    if beam_size == 1:
        return greedy_search(
            model,
            src_ids,
            src_mask,
            bos_id,
            eos_id,
            do_dai_toi_da=do_dai_toi_da,
            dung_kv_cache=dung_kv_cache,
        )

    from nmt.model.masking import tao_causal_mask

    batch_size = src_ids.size(0)
    device = src_ids.device
    pad_id = getattr(model, "pad_id", eos_id)

    # Encoder chỉ chạy một lần cho mỗi câu rồi nhân theo số beam.
    memory = model.encode(src_ids, src_mask)
    memory = memory.repeat_interleave(beam_size, dim=0)
    if src_mask is not None:
        src_mask = src_mask.repeat_interleave(beam_size, dim=0)

    sequences = torch.full(
        (batch_size, beam_size, 1), bos_id, dtype=torch.long, device=device
    )
    scores = torch.full((batch_size, beam_size), -torch.inf, device=device)
    scores[:, 0] = 0.0
    lengths = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)
    finished = torch.zeros((batch_size, beam_size), dtype=torch.bool, device=device)
    caches = None

    for _ in range(do_dai_toi_da):
        flat_sequences = sequences.reshape(batch_size * beam_size, -1)
        if dung_kv_cache:
            decoder_output, caches = model.decode_step(
                flat_sequences[:, -1:],
                memory,
                src_mask,
                caches=caches,
                vi_tri_bat_dau=flat_sequences.size(1) - 1,
            )
        else:
            tgt_mask = tao_causal_mask(flat_sequences.size(1), device=device)
            decoder_output = model.decode(flat_sequences, memory, tgt_mask, src_mask)

        logits = model.output_projection(decoder_output)[:, -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        vocab_size = log_probs.size(-1)
        log_probs = log_probs.view(batch_size, beam_size, vocab_size)

        # Beam đã kết thúc chỉ được nối PAD với xác suất log bằng 0. Nhờ vậy
        # không có token nội dung nào xuất hiện sau EOS và điểm không bị đổi.
        log_probs = log_probs.masked_fill(finished.unsqueeze(-1), -torch.inf)
        log_probs[:, :, pad_id] = torch.where(
            finished,
            torch.zeros_like(scores),
            log_probs[:, :, pad_id],
        )

        candidate_scores = scores.unsqueeze(-1) + log_probs
        candidate_scores = candidate_scores.view(batch_size, beam_size * vocab_size)
        scores_moi, flat_indices = candidate_scores.topk(beam_size, dim=-1)
        parent_beams = torch.div(flat_indices, vocab_size, rounding_mode="floor")
        next_tokens = flat_indices.remainder(vocab_size)

        gather_sequences = parent_beams.unsqueeze(-1).expand(
            -1, -1, sequences.size(-1)
        )
        sequences = torch.gather(sequences, 1, gather_sequences)
        sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

        parent_finished = torch.gather(finished, 1, parent_beams)
        parent_lengths = torch.gather(lengths, 1, parent_beams)
        lengths = parent_lengths + (~parent_finished).long()
        finished = parent_finished | next_tokens.eq(eos_id)
        scores = scores_moi

        if dung_kv_cache:
            # Cache vừa tạo thuộc về beam cha. Sắp lại đúng thứ tự các beam con
            # được top-k chọn, nếu không câu vẫn chạy nhưng token sẽ âm thầm sai.
            offsets = (
                torch.arange(batch_size, device=device).unsqueeze(1) * beam_size
            )
            parent_flat = (parent_beams + offsets).reshape(-1)
            caches = [
                {
                    ten: tuple(t.index_select(0, parent_flat) for t in kv)
                    for ten, kv in layer_cache.items()
                }
                for layer_cache in caches
            ]

        if finished.all():
            break

    length_penalty = ((5.0 + lengths.clamp_min(1).float()) / 6.0).pow(
        he_so_phat_do_dai
    )
    normalized_scores = scores / length_penalty
    best_beams = normalized_scores.argmax(dim=-1)
    batch_indices = torch.arange(batch_size, device=device)
    return sequences[batch_indices, best_beams]
