"""Package nmt.data — TASK 04 (My).

Cung cấp:
    DuLieuSongNgu      Dataset song ngữ En-Vi
    tao_dataloader     Tạo DataLoader với token-bucket batching
    ghep_batch         collate_fn: padding + mask
    ti_le_o_dem        Đo tỉ lệ padding trong batch
    gom_batch_theo_token  Tính danh sách batch (debug)
    nap_tokenizer      Load tokenizer từ file .json
    kiem_tra_ma_hoa_giai_ma  Kiểm tra round-trip encode/decode
    ti_le_token_la     Tỉ lệ UNK trên tập dev
"""

from nmt.data.collate import (
    GomBatchTheoTokenSampler,
    ghep_batch,
    gom_batch_theo_token,
    ti_le_o_dem,
    tao_dataloader,
)
from nmt.data.dataset import DuLieuSongNgu
from nmt.data.tokenizer import (
    nap_tokenizer,
    train_tokenizer,
    kiem_tra_ma_hoa_giai_ma,
    ti_le_token_la,
    PAD_ID,
    UNK_ID,
    BOS_ID,
    EOS_ID,
)

__all__ = [
    "DuLieuSongNgu",
    "GomBatchTheoTokenSampler",
    "ghep_batch",
    "gom_batch_theo_token",
    "ti_le_o_dem",
    "tao_dataloader",
    "nap_tokenizer",
    "train_tokenizer",
    "kiem_tra_ma_hoa_giai_ma",
    "ti_le_token_la",
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "EOS_ID",
]
