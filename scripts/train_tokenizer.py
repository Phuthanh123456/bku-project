"""TASK 03 — Train BPE tokenizer 32k dùng chung hai ngôn ngữ.  Người làm: My.

Dùng thư viện `tokenizers` của Hugging Face (KHÔNG dùng transformers hay sentencepiece).
Tokenizer học CHỈ trên tập train, KHÔNG nhìn vào dev/test — tránh data leakage.

Chạy lại từ đầu bằng đúng 1 lệnh:
    python scripts/train_tokenizer.py

Sinh ra:
    artifacts/tokenizer/tokenizer.json          (file chính, load bằng tokenizers.Tokenizer.from_file)
    artifacts/tokenizer/vocab.json              (vocabulary ánh xạ token → id)
    artifacts/tokenizer/merges.txt              (BPE merge rules)
    results/thong_ke_tokenizer.csv              (30 token hay gặp nhất, fertility, unk rate)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from pathlib import Path

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, luu_config, nap_config

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train_tokenizer")

# ---------------------------------------------------------------------------
# Token đặc biệt — thứ tự QUAN TRỌNG, phải nhất quán với TASK 04, 05, 09, 11
#   0: <pad>   — padding (attention_mask = 0 tại vị trí này)
#   1: <unk>   — unknown (không nên xuất hiện sau khi train BPE đủ lớn)
#   2: <bos>   — beginning of sequence (decoder input)
#   3: <eos>   — end of sequence (decoder output, dùng để dừng sinh câu)
#   4: <2en>   — language token English (dành sẵn cho multilingual, TASK 03 yêu cầu)
#   5: <2vi>   — language token Vietnamese
# Chốt dùng <bos>/<eos>, không dùng <s>/</s> — Quân biết khi làm TASK 09.
# ---------------------------------------------------------------------------
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<2en>", "<2vi>"]
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
L2EN_ID = 4   # language token English
L2VI_ID = 5   # language token Vietnamese


# ---------------------------------------------------------------------------
# Tiện ích đọc file văn bản
# ---------------------------------------------------------------------------

def _doc_cac_dong(duong_dan: Path) -> list[str]:
    """Đọc file văn bản, trả về danh sách dòng (đã strip, bỏ dòng rỗng)."""
    return [
        dong.strip()
        for dong in duong_dan.read_text(encoding="utf-8").splitlines()
        if dong.strip()
    ]


# ---------------------------------------------------------------------------
# Thống kê tokenizer
# ---------------------------------------------------------------------------

def _tinh_fertility(tokenizer, cac_cau: list[str]) -> float:
    """Fertility = số subword trung bình trên mỗi từ (tách theo khoảng trắng).

    Giá trị kỳ vọng BPE tốt: 1.0–1.5. Cao hơn 2.0 nghĩa là vocab quá nhỏ
    hoặc dữ liệu quá đa dạng so với kích thước vocab.
    """
    tong_subword = 0
    tong_tu = 0
    for cau in cac_cau:
        so_tu = len(cau.split())
        if so_tu == 0:
            continue
        ma_hoa = tokenizer.encode(cau)
        tong_subword += len(ma_hoa.ids)
        tong_tu += so_tu
    return tong_subword / tong_tu if tong_tu > 0 else 0.0


def _tinh_unk_rate(tokenizer, cac_cau: list[str]) -> float:
    """Tỉ lệ token UNK trong toàn bộ output của tokenizer.

    Sau khi train BPE 32k trên cùng tập dữ liệu, giá trị này nên gần 0.
    Nếu > 0.1% cần kiểm tra lại quá trình tiền xử lý Unicode.
    """
    tong_unk = 0
    tong_token = 0
    for cau in cac_cau:
        ids = tokenizer.encode(cau).ids
        tong_unk += ids.count(UNK_ID)
        tong_token += len(ids)
    return tong_unk / tong_token if tong_token > 0 else 0.0


def _lay_30_token_pho_bien(tokenizer, cac_cau: list[str]) -> list[tuple[str, int]]:
    """Trả về 30 token xuất hiện nhiều nhất (trừ special tokens)."""
    dem = Counter()
    for cau in cac_cau:
        ids = tokenizer.encode(cau).ids
        for id_ in ids:
            if id_ not in (PAD_ID, UNK_ID, BOS_ID, EOS_ID, L2EN_ID, L2VI_ID):
                dem[tokenizer.id_to_token(id_)] += 1
    return dem.most_common(30)


def _ghi_csv_thong_ke(
    duong_dan: Path,
    fertility_train_en: float,
    fertility_train_vi: float,
    unk_rate_train: float,
    unk_rate_dev: float,
    top30: list[tuple[str, int]],
) -> None:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["fertility_train_en", f"{fertility_train_en:.4f}"])
        w.writerow(["fertility_train_vi", f"{fertility_train_vi:.4f}"])
        w.writerow(["unk_rate_train_pct", f"{unk_rate_train * 100:.4f}"])
        w.writerow(["unk_rate_dev_pct",   f"{unk_rate_dev * 100:.4f}"])
        w.writerow(["---top30_tokens---", "count"])
        for token, cnt in top30:
            w.writerow([token, cnt])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Ghi de thi_nghiem.seed")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    vocab_size: int = cfg.du_lieu.vocab_size          # 32000
    goc = Path(__file__).resolve().parents[1]

    # --- Đường dẫn ---
    thu_muc_processed = goc / "data" / "processed"
    thu_muc_artifact  = goc / "artifacts" / "tokenizer"
    thu_muc_results   = goc / "results"

    file_train_en = thu_muc_processed / "train.en"
    file_train_vi = thu_muc_processed / "train.vi"
    file_dev_en   = thu_muc_processed / "tst2012.en"
    file_dev_vi   = thu_muc_processed / "tst2012.vi"

    for f in [file_train_en, file_train_vi]:
        if not f.exists():
            raise FileNotFoundError(
                f"Khong tim thay: {f}\n"
                "Hay chay scripts/prepare_data.py truoc."
            )

    log.info("=" * 60)
    log.info("TASK 03 - Train BPE Tokenizer 32k (dung chung En+Vi)")
    log.info(f"  Config    : {args.config}")
    log.info(f"  Seed      : {cfg.thi_nghiem.seed}")
    log.info(f"  Vocab size: {vocab_size:,}")
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # BƯỚC 1 — Đọc corpus train (chỉ train, không dev/test)
    # -----------------------------------------------------------------------
    log.info("\n[Buoc 1] Doc corpus train ...")
    dong_en = _doc_cac_dong(file_train_en)
    dong_vi = _doc_cac_dong(file_train_vi)
    log.info(f"  train.en : {len(dong_en):,} dong")
    log.info(f"  train.vi : {len(dong_vi):,} dong")

    corpus = dong_en + dong_vi   # gộp 2 ngôn ngữ, tokenizer học chung
    log.info(f"  Tong corpus: {len(corpus):,} dong ({len(dong_en):,} En + {len(dong_vi):,} Vi)")

    # -----------------------------------------------------------------------
    # BƯỚC 2 — Cấu hình và train BPE tokenizer
    # -----------------------------------------------------------------------
    log.info("\n[Buoc 2] Train BPE tokenizer ...")

    try:
        # pyrefly: ignore [missing-import]
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    except ImportError:
        raise ImportError(
            "Thieu thu vien 'tokenizers'. Cai bang:\n"
            "  conda run -n envi-nmt pip install tokenizers"
        )

    # Mô hình BPE với token đặc biệt
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    # Pre-tokenizer: ByteLevel — cùng cách với GPT-2, xử lý tốt tiếng Việt
    # vì tách theo byte, không mất thông tin với ký tự Unicode đặc biệt.
    # add_prefix_space=False để không thêm khoảng trắng đầu câu.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder tương ứng với ByteLevel pre-tokenizer
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,           # subword phải xuất hiện ≥ 2 lần mới vào vocab
        show_progress=True,
    )

    log.info(f"  Dang train tren {len(corpus):,} dong ... (co the mat 1-2 phut)")
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    log.info("  Train xong!")

    # Kiểm tra id của special tokens đúng thứ tự đã khai báo
    for ten, id_mong_muon in zip(SPECIAL_TOKENS, [PAD_ID, UNK_ID, BOS_ID, EOS_ID, L2EN_ID, L2VI_ID]):
        id_thuc_te = tokenizer.token_to_id(ten)
        if id_thuc_te != id_mong_muon:
            raise RuntimeError(
                f"Special token '{ten}' co id={id_thuc_te}, mong muon id={id_mong_muon}.\n"
                "Kiem tra lai thu tu khai bao SPECIAL_TOKENS."
            )

    vocab_thuc_te = tokenizer.get_vocab_size()
    log.info(f"  Vocab size thuc te: {vocab_thuc_te:,} (mong muon: {vocab_size:,})")

    # -----------------------------------------------------------------------
    # BƯỚC 3 — Lưu tokenizer
    # -----------------------------------------------------------------------
    log.info("\n[Buoc 3] Luu tokenizer ...")
    thu_muc_artifact.mkdir(parents=True, exist_ok=True)

    # File chính — dùng để load trong training và inference
    tokenizer_path = thu_muc_artifact / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    log.info(f"  tokenizer.json : {tokenizer_path.stat().st_size:,} bytes")

    # Xuất vocab và merges riêng để dễ đọc / debug
    model = tokenizer.model
    vocab_path  = thu_muc_artifact / "vocab.json"
    merges_path = thu_muc_artifact / "merges.txt"
    model.save(str(thu_muc_artifact))
    log.info(f"  vocab.json     : {vocab_path.stat().st_size:,} bytes")
    log.info(f"  merges.txt     : {merges_path.stat().st_size:,} bytes")

    # -----------------------------------------------------------------------
    # BƯỚC 4 — Tính thống kê
    # -----------------------------------------------------------------------
    log.info("\n[Buoc 4] Tinh thong ke ...")

    dong_dev_en = _doc_cac_dong(file_dev_en)
    dong_dev_vi = _doc_cac_dong(file_dev_vi)

    # Dùng mẫu 5000 câu đầu để tính nhanh (đủ đại diện)
    MẪU = 5000
    mau_en = dong_en[:MẪU]
    mau_vi = dong_vi[:MẪU]
    mau_dev = (dong_dev_en + dong_dev_vi)

    fertility_en  = _tinh_fertility(tokenizer, mau_en)
    fertility_vi  = _tinh_fertility(tokenizer, mau_vi)
    unk_train     = _tinh_unk_rate(tokenizer, mau_en + mau_vi)
    unk_dev       = _tinh_unk_rate(tokenizer, mau_dev)
    top30         = _lay_30_token_pho_bien(tokenizer, mau_en + mau_vi)

    log.info(f"  Fertility EN (train) : {fertility_en:.4f} subword/tu")
    log.info(f"  Fertility VI (train) : {fertility_vi:.4f} subword/tu")
    log.info(f"  UNK rate train       : {unk_train * 100:.4f}%")
    log.info(f"  UNK rate dev         : {unk_dev * 100:.4f}%")

    csv_path = thu_muc_results / "thong_ke_tokenizer.csv"
    _ghi_csv_thong_ke(csv_path, fertility_en, fertility_vi, unk_train, unk_dev, top30)
    log.info(f"  Da ghi: {csv_path}")

    # Lưu config đã dùng
    luu_config(cfg, thu_muc_results / "config_tokenizer.yaml")

    # -----------------------------------------------------------------------
    # Kiểm tra nhanh: encode/decode 1 câu mẫu
    # -----------------------------------------------------------------------
    log.info("\n[Kiem tra nhanh] encode/decode thu 1 cau:")
    cau_thu = "The quick brown fox jumps over the lazy dog."
    encoded = tokenizer.encode(cau_thu)
    decoded = tokenizer.decode(encoded.ids)
    log.info(f"  Input  : {cau_thu}")
    log.info(f"  Tokens : {encoded.tokens[:15]} ...")
    log.info(f"  IDs    : {encoded.ids[:15]} ...")
    log.info(f"  Decoded: {decoded}")

    cau_vi = "Xin chào, tôi đang học xây dựng mô hình dịch máy."
    encoded_vi = tokenizer.encode(cau_vi)
    decoded_vi = tokenizer.decode(encoded_vi.ids)
    log.info(f"  Input  : {cau_vi}")
    log.info(f"  Tokens : {encoded_vi.tokens[:15]} ...")
    log.info(f"  Decoded: {decoded_vi}")

    # -----------------------------------------------------------------------
    # Tổng kết
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("Tong ket TASK 03:")
    log.info(f"  Vocab size        : {vocab_thuc_te:,}")
    log.info(f"  Fertility EN      : {fertility_en:.4f}")
    log.info(f"  Fertility VI      : {fertility_vi:.4f}")
    log.info(f"  UNK rate (train)  : {unk_train * 100:.4f}%")
    log.info(f"  UNK rate (dev)    : {unk_dev * 100:.4f}%")
    log.info(f"  Luu tai           : {tokenizer_path}")
    log.info("=" * 60)
    log.info("TASK 03 hoan tat.")
    log.info("Buoc tiep theo: python scripts/train_tokenizer.py da xong,")
    log.info("  kiem tra bang: python -c \"from tokenizers import Tokenizer; "
             "t=Tokenizer.from_file('artifacts/tokenizer/tokenizer.json'); "
             "print(t.get_vocab_size())\"")


if __name__ == "__main__":
    main()
