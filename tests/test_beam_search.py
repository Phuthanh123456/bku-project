"""Kiểm tra Beam Search và KV cache — TASK 19."""

from __future__ import annotations

import copy
import math

import pytest
import torch
import torch.nn as nn

from nmt.inference.search import beam_search, greedy_search
from nmt.model.masking import tao_padding_mask
from nmt.model.transformer import TransformerNMT


def _tiny_model(cfg_goc, positional: str = "rope") -> TransformerNMT:
    cfg = copy.deepcopy(cfg_goc)
    cfg.du_lieu.vocab_size = 32
    cfg.du_lieu.do_dai_toi_da = 32
    cfg.mo_hinh.d_model = 16
    cfg.mo_hinh.so_head = 4
    cfg.mo_hinh.so_lop_encoder = 2
    cfg.mo_hinh.so_lop_decoder = 2
    cfg.mo_hinh.d_ff = 24
    cfg.mo_hinh.dropout = 0.0
    cfg.mo_hinh.ma_hoa_vi_tri = positional
    return TransformerNMT(cfg).eval()


def _source():
    src = torch.tensor([[5, 6, 3, 0], [7, 8, 9, 3]])
    return src, tao_padding_mask(src, pad_id=0)


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _LengthPolicyModel(nn.Module):
    """Mô hình giả nhỏ: có một câu ngắn tốt và một câu dài gần tốt bằng."""

    pad_id = 0
    vocab_size = 6

    def __init__(self):
        super().__init__()
        self.output_projection = _Identity()

    def encode(self, src_ids, src_mask):
        return torch.zeros(src_ids.size(0), 1, self.vocab_size)

    def decode(self, tgt_ids, memory, tgt_mask, src_mask):
        batch, length = tgt_ids.shape
        logits = torch.full((batch, length, self.vocab_size), -50.0)
        for row, sequence in enumerate(tgt_ids.tolist()):
            count_long = sequence.count(4)
            if count_long == 0:
                # EOS có xác suất 0,55; nhánh dài có xác suất 0,45.
                logits[row, -1, 3] = math.log(0.55)
                logits[row, -1, 4] = math.log(0.45)
            elif count_long < 3:
                logits[row, -1, 4] = 0.0
            else:
                logits[row, -1, 3] = 0.0
        return logits


def test_beam_bang_1_trung_khop_greedy(cfg_goc):
    model = _tiny_model(cfg_goc)
    src, src_mask = _source()
    greedy = greedy_search(model, src, src_mask, 2, 3, do_dai_toi_da=8)
    beam = beam_search(model, src, src_mask, 2, 3, beam_size=1, do_dai_toi_da=8)
    assert torch.equal(beam, greedy)


def test_beam_dung_lai_khi_gap_eos():
    model = _LengthPolicyModel()
    src = torch.tensor([[5]])
    result = beam_search(
        model, src, None, 2, 3, beam_size=2, he_so_phat_do_dai=0.0, do_dai_toi_da=6
    )
    eos_position = result[0].tolist().index(3)
    assert result[0, eos_position + 1 :].eq(model.pad_id).all()


def test_he_so_phat_do_dai_co_tac_dung():
    model = _LengthPolicyModel()
    src = torch.tensor([[5]])
    ngan = beam_search(
        model, src, None, 2, 3, beam_size=2, he_so_phat_do_dai=0.0, do_dai_toi_da=6
    )
    dai = beam_search(
        model, src, None, 2, 3, beam_size=2, he_so_phat_do_dai=4.0, do_dai_toi_da=6
    )
    assert dai[0].ne(model.pad_id).sum() > ngan[0].ne(model.pad_id).sum()


@pytest.mark.parametrize("positional", ["rope", "sinusoidal"])
def test_greedy_kv_cache_trung_khop_decode_day_du(cfg_goc, positional):
    model = _tiny_model(cfg_goc, positional)
    src, src_mask = _source()
    day_du = greedy_search(model, src, src_mask, 2, 3, 8, dung_kv_cache=False)
    co_cache = greedy_search(model, src, src_mask, 2, 3, 8, dung_kv_cache=True)
    assert torch.equal(day_du, co_cache)


def test_beam_kv_cache_trung_khop_decode_day_du(cfg_goc):
    model = _tiny_model(cfg_goc)
    src, src_mask = _source()
    day_du = beam_search(model, src, src_mask, 2, 3, 3, 1.0, 8, False)
    co_cache = beam_search(model, src, src_mask, 2, 3, 3, 1.0, 8, True)
    assert torch.equal(day_du, co_cache)
