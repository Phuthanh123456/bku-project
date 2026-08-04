"""TASK 02 — Tải và làm sạch dữ liệu IWSLT 2015 En-Vi.  Người làm: My.

Chạy lại từ đầu bằng đúng 1 lệnh:
    python scripts/prepare_data.py

Sinh ra:
    data/raw/           (giải nén từ 3 tgz)
    data/processed/{train,tst2012,tst2013}.{en,vi}
    results/thong_ke_loc.csv
    results/config_da_dung.yaml
    docs/bao_cao_du_lieu.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import logging
import re
import sys
import tarfile
import unicodedata
import urllib.request
from pathlib import Path
from typing import NamedTuple

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, luu_config, nap_config

# ---------------------------------------------------------------------------
# Logger chuẩn stdlib (BoGhiLog dành cho training metrics, không dùng ở đây)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("prepare_data")

# ---------------------------------------------------------------------------
# Nguồn dữ liệu — mirror GitHub của stefan-it/nmt-en-vi (ổn định)
# ---------------------------------------------------------------------------
_BASE_URL = "https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data"

NGUON_DU_LIEU = [
    {
        "ten_tgz": "train-en-vi.tgz",
        "url": f"{_BASE_URL}/train-en-vi.tgz",
        # sha256 của file gốc để phát hiện tải hỏng
        "sha256": None,   # để None → bỏ qua kiểm tra hash (đã đặt chú thích bên dưới)
        "files": ["train.en", "train.vi"],
        "split": "train",
    },
    {
        "ten_tgz": "dev-2012-en-vi.tgz",
        "url": f"{_BASE_URL}/dev-2012-en-vi.tgz",
        "sha256": None,
        "files": ["tst2012.en", "tst2012.vi"],
        "split": "tst2012",
    },
    {
        "ten_tgz": "test-2013-en-vi.tgz",
        "url": f"{_BASE_URL}/test-2013-en-vi.tgz",
        "sha256": None,
        "files": ["tst2013.en", "tst2013.vi"],
        "split": "tst2013",
    },
]

# ---------------------------------------------------------------------------
# Ngưỡng lọc — ĐỌC TỪ configs/base.yaml  (khóa du_lieu)
# Nếu muốn thay đổi, sửa base.yaml, KHÔNG sửa code ở đây.
# ---------------------------------------------------------------------------
_DO_DAI_TOI_THIEU = 1   # tokens (từ/BPE units)  — luôn lọc câu rỗng
# _DO_DAI_TOI_DA được lấy từ cfg.du_lieu.do_dai_toi_da


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------

def _sha256_file(duong_dan: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with duong_dan.open("rb") as f:
        while True:
            bloc = f.read(chunk)
            if not bloc:
                break
            h.update(bloc)
    return h.hexdigest()


def _tai_file(url: str, dich: Path, sha256: str | None = None) -> None:
    """Tải URL về *dich*; bỏ qua nếu file đã tồn tại & hash khớp."""
    if dich.exists():
        if sha256 is None or _sha256_file(dich) == sha256:
            log.info(f"  ✔ Đã có sẵn: {dich.name}, bỏ qua tải lại.")
            return
        log.warning(f"  ⚠ Hash không khớp, tải lại: {dich.name}")

    log.info(f"  ↓ Đang tải: {url}")
    dich.parent.mkdir(parents=True, exist_ok=True)

    def _tien_trinh(bo_block: int, kich_thuoc_block: int, tong: int) -> None:
        if tong > 0:
            pct = min(100, bo_block * kich_thuoc_block * 100 // tong)
            print(f"\r    {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dich, reporthook=_tien_trinh)
    print()  # xuống dòng sau progress bar
    log.info(f"  ✔ Đã lưu: {dich} ({dich.stat().st_size:,} bytes)")

    if sha256:
        thuc_te = _sha256_file(dich)
        if thuc_te != sha256:
            raise RuntimeError(
                f"SHA-256 không khớp cho {dich.name}.\n"
                f"  Kỳ vọng : {sha256}\n"
                f"  Thực tế : {thuc_te}"
            )


def _giai_nen(tgz: Path, thu_muc: Path) -> None:
    """Giải nén tgz vào thu_muc nếu chưa có đủ file."""
    log.info(f"  Giai nen: {tgz.name} -> {thu_muc}/")
    thu_muc.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(path=thu_muc)


# ---------------------------------------------------------------------------
# Làm sạch 1 câu
# ---------------------------------------------------------------------------

_RE_HTML_TAG   = re.compile(r"<[^>]+>")
_RE_NHIEU_TRONG = re.compile(r"\s+")


def _lam_sach_cau(cau: str) -> str:
    """Chuẩn hoá 1 chuỗi văn bản:

    1. Strip
    2. Giải mã HTML entity (&amp; → &, &lt; → <, …)
    3. Bỏ tag HTML còn sót (<br />, <i>, …)
    4. Chuẩn hoá Unicode NFC (quan trọng với tiếng Việt: ô + dấu → 1 codepoint)
    5. Thay nhiều khoảng trắng liên tiếp bằng 1 dấu cách
    6. Strip lại
    """
    cau = cau.strip()
    cau = html.unescape(cau)                    # bước 2
    cau = _RE_HTML_TAG.sub(" ", cau)            # bước 3
    cau = unicodedata.normalize("NFC", cau)     # bước 4
    cau = _RE_NHIEU_TRONG.sub(" ", cau)         # bước 5
    return cau.strip()


def _dem_token_don_gian(cau: str) -> int:
    """Đếm token thô (tách theo khoảng trắng) — chỉ dùng để lọc độ dài."""
    return len(cau.split())


# ---------------------------------------------------------------------------
# Thống kê lọc
# ---------------------------------------------------------------------------

class ThongKe(NamedTuple):
    split: str
    truoc_loc: int
    sau_loc: int
    bi_rong: int
    bi_trung: int
    bi_qua_ngan: int
    bi_qua_dai: int

    @property
    def bi_loai(self) -> int:
        return self.truoc_loc - self.sau_loc


# ---------------------------------------------------------------------------
# Hàm lõi: đọc → làm sạch → lọc → ghi
# ---------------------------------------------------------------------------

def _xu_ly_split(
    *,
    split: str,
    file_en_raw: Path,
    file_vi_raw: Path,
    thu_muc_processed: Path,
    do_dai_toi_da: int,
) -> ThongKe:
    """Đọc cặp file raw, làm sạch + lọc, ghi ra processed."""

    log.info(f"\n[{split}] Đang xử lý …")

    # --- Đọc ---
    dong_en = file_en_raw.read_text(encoding="utf-8").splitlines()
    dong_vi = file_vi_raw.read_text(encoding="utf-8").splitlines()

    if len(dong_en) != len(dong_vi):
        raise RuntimeError(
            f"[{split}] Số dòng EN ({len(dong_en)}) ≠ VI ({len(dong_vi)}) trong raw — "
            "dữ liệu gốc đã bị lệch cặp!"
        )

    truoc_loc = len(dong_en)

    # --- Làm sạch từng dòng ---
    dong_en = [_lam_sach_cau(c) for c in dong_en]
    dong_vi = [_lam_sach_cau(c) for c in dong_vi]

    # --- Lọc ---
    dem_rong     = 0
    dem_trung    = 0
    dem_qua_ngan = 0
    dem_qua_dai  = 0

    ket_qua_en: list[str] = []
    ket_qua_vi: list[str] = []
    da_gap: set[tuple[str, str]] = set()

    for en, vi in zip(dong_en, dong_vi):
        # 1. Bỏ câu rỗng (một trong hai bên)
        if not en or not vi:
            dem_rong += 1
            continue

        # 2. Bỏ trùng lặp (theo cặp)
        cap = (en, vi)
        if cap in da_gap:
            dem_trung += 1
            continue
        da_gap.add(cap)

        # 3. Lọc quá ngắn
        n_en = _dem_token_don_gian(en)
        n_vi = _dem_token_don_gian(vi)
        if n_en < _DO_DAI_TOI_THIEU or n_vi < _DO_DAI_TOI_THIEU:
            dem_qua_ngan += 1
            continue

        # 4. Lọc quá dài  (dùng do_dai_toi_da từ config)
        if n_en > do_dai_toi_da or n_vi > do_dai_toi_da:
            dem_qua_dai += 1
            continue

        ket_qua_en.append(en)
        ket_qua_vi.append(vi)

    sau_loc = len(ket_qua_en)

    # --- Ghi ra processed ---
    thu_muc_processed.mkdir(parents=True, exist_ok=True)
    (thu_muc_processed / f"{split}.en").write_text(
        "\n".join(ket_qua_en) + "\n", encoding="utf-8"
    )
    (thu_muc_processed / f"{split}.vi").write_text(
        "\n".join(ket_qua_vi) + "\n", encoding="utf-8"
    )

    # --- Kiểm tra bất biến: số dòng EN == VI ---
    en_lines = len((thu_muc_processed / f"{split}.en").read_text(encoding="utf-8").splitlines())
    vi_lines = len((thu_muc_processed / f"{split}.vi").read_text(encoding="utf-8").splitlines())
    if en_lines != vi_lines:
        raise RuntimeError(
            f"[{split}] BUG: sau xử lý EN ({en_lines}) ≠ VI ({vi_lines}). Báo ngay cho nhóm!"
        )

    log.info(
        f"[{split}] {truoc_loc:>7,} cap -> {sau_loc:>7,} cap  "
        f"(-rong:{dem_rong}  -trung:{dem_trung}  "
        f"-ngan:{dem_qua_ngan}  -dai:{dem_qua_dai})"
    )

    return ThongKe(
        split=split,
        truoc_loc=truoc_loc,
        sau_loc=sau_loc,
        bi_rong=dem_rong,
        bi_trung=dem_trung,
        bi_qua_ngan=dem_qua_ngan,
        bi_qua_dai=dem_qua_dai,
    )


# ---------------------------------------------------------------------------
# Ghi báo cáo
# ---------------------------------------------------------------------------

def _ghi_csv(danh_sach: list[ThongKe], duong_dan: Path) -> None:
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    with duong_dan.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split", "truoc_loc", "sau_loc", "bi_loai",
            "bi_rong", "bi_trung", "bi_qua_ngan", "bi_qua_dai",
        ])
        for tk in danh_sach:
            writer.writerow([
                tk.split, tk.truoc_loc, tk.sau_loc, tk.bi_loai,
                tk.bi_rong, tk.bi_trung, tk.bi_qua_ngan, tk.bi_qua_dai,
            ])


def _ghi_bao_cao(
    danh_sach: list[ThongKe],
    duong_dan: Path,
    do_dai_toi_da: int,
    seed: int,
) -> None:
    """Ghi docs/bao_cao_du_lieu.md để người khác tái lập được."""
    duong_dan.parent.mkdir(parents=True, exist_ok=True)
    dong: list[str] = [
        "# Báo cáo dữ liệu — TASK 02",
        "",
        "**Người làm:** My  |  **Seed:** " + str(seed),
        "",
        "## Nguồn",
        "",
        f"IWSLT 2015 English–Vietnamese, mirror: `{_BASE_URL}`",
        "",
        "| File tgz | URL |",
        "|---|---|",
    ]
    for info in NGUON_DU_LIEU:
        dong.append(f"| `{info['ten_tgz']}` | {info['url']} |")

    dong += [
        "",
        "## Cấu trúc thư mục sau khi chạy",
        "",
        "```",
        "data/",
        "├── raw/",
        "│   ├── train.en  train.vi",
        "│   ├── tst2012.en  tst2012.vi",
        "│   └── tst2013.en  tst2013.vi",
        "└── processed/",
        "    ├── train.en  train.vi",
        "    ├── tst2012.en  tst2012.vi",
        "    └── tst2013.en  tst2013.vi",
        "```",
        "",
        "## Thống kê lọc",
        "",
        f"> Ngưỡng lọc: `do_dai_toi_da = {do_dai_toi_da}` token (đọc từ `configs/base.yaml`)",
        "",
        "| Split | Trước lọc | Sau lọc | Bị loại | Rỗng | Trùng | Quá ngắn | Quá dài |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for tk in danh_sach:
        dong.append(
            f"| {tk.split} | {tk.truoc_loc:,} | {tk.sau_loc:,} | {tk.bi_loai:,} "
            f"| {tk.bi_rong} | {tk.bi_trung} | {tk.bi_qua_ngan} | {tk.bi_qua_dai} |"
        )

    dong += [
        "",
        "## Lưu ý tái lập",
        "",
        "Chạy lại từ đầu bằng:",
        "```bash",
        "python scripts/prepare_data.py --config configs/base.yaml",
        "```",
        "",
        "Sau khi chạy xong, kiểm tra bất biến quan trọng:",
        "```bash",
        "wc -l data/processed/train.en data/processed/train.vi",
        "wc -l data/processed/tst2012.en data/processed/tst2012.vi",
        "wc -l data/processed/tst2013.en data/processed/tst2013.vi",
        "```",
        "Số dòng EN và VI của mỗi split **phải bằng nhau** (script đã tự kiểm tra và báo lỗi nếu lệch).",
    ]

    duong_dan.write_text("\n".join(dong) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml", help="Duong dan toi file cau hinh YAML")
    parser.add_argument("--seed", type=int, default=None, help="Ghi de thi_nghiem.seed")
    args = parser.parse_args()

    # --- Nap cau hinh ---
    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    do_dai_toi_da: int = cfg.du_lieu.do_dai_toi_da  # nguong loc cau qua dai

    # --- Thiet lap duong dan ---
    goc = Path(__file__).resolve().parents[1]   # thu muc goc cua repo
    thu_muc_raw       = goc / "data" / "raw"
    thu_muc_processed = goc / "data" / "processed"
    thu_muc_results   = goc / "results"
    thu_muc_docs      = goc / "docs"

    log.info("=" * 60)
    log.info("TASK 02 — Tải & làm sạch dữ liệu IWSLT 2015 En-Vi")
    log.info(f"  Config       : {args.config}")
    log.info(f"  Seed         : {cfg.thi_nghiem.seed}")
    log.info(f"  do_dai_toi_da: {do_dai_toi_da} tokens")
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # BƯỚC 1 — Tải và giải nén
    # -----------------------------------------------------------------------
    log.info("\n[Bước 1] Tải dữ liệu …")

    for info in NGUON_DU_LIEU:
        tgz_path = thu_muc_raw / info["ten_tgz"]

        # Tai
        _tai_file(info["url"], tgz_path, sha256=info["sha256"])

        # Kiem tra xem file da giai nen chua (tranh giai nen lai mat thoi gian)
        files_can = [thu_muc_raw / f for f in info["files"]]
        if all(p.exists() for p in files_can):
            log.info(f"  Da giai nen san: {[p.name for p in files_can]}")
        else:
            _giai_nen(tgz_path, thu_muc_raw)

        # Xác nhận tồn tại
        for p in files_can:
            if not p.exists():
                raise FileNotFoundError(
                    f"Sau giải nén vẫn không thấy: {p}\n"
                    "Kiểm tra lại cấu trúc bên trong file tgz."
                )
            log.info(f"  ✔ {p.name}  ({p.stat().st_size:,} bytes)")

    # -----------------------------------------------------------------------
    # BƯỚC 2 — Làm sạch và lọc
    # -----------------------------------------------------------------------
    log.info("\n[Bước 2] Làm sạch & lọc …")

    tat_ca_thong_ke: list[ThongKe] = []

    for info in NGUON_DU_LIEU:
        split = info["split"]
        file_en = thu_muc_raw / info["files"][0]   # *.en
        file_vi = thu_muc_raw / info["files"][1]   # *.vi

        tk = _xu_ly_split(
            split=split,
            file_en_raw=file_en,
            file_vi_raw=file_vi,
            thu_muc_processed=thu_muc_processed,
            do_dai_toi_da=do_dai_toi_da,
        )
        tat_ca_thong_ke.append(tk)

    # -----------------------------------------------------------------------
    # BƯỚC 3 — Ghi báo cáo
    # -----------------------------------------------------------------------
    log.info("\n[Bước 3] Ghi báo cáo …")

    csv_path  = thu_muc_results / "thong_ke_loc.csv"
    bao_cao   = thu_muc_docs    / "bao_cao_du_lieu.md"

    _ghi_csv(tat_ca_thong_ke, csv_path)
    _ghi_bao_cao(tat_ca_thong_ke, bao_cao, do_dai_toi_da, cfg.thi_nghiem.seed)

    # Lưu lại config đã dùng (nguyên tắc tái lập)
    luu_config(cfg, thu_muc_results / "config_da_dung.yaml")

    log.info(f"  ✔ Thống kê : {csv_path}")
    log.info(f"  ✔ Báo cáo  : {bao_cao}")

    # -----------------------------------------------------------------------
    # Tổng kết cuối
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("Tổng kết:")
    log.info(f"  {'Split':<10} {'Trước':>10} {'Sau':>10} {'Bị loại':>10}")
    log.info(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for tk in tat_ca_thong_ke:
        log.info(f"  {tk.split:<10} {tk.truoc_loc:>10,} {tk.sau_loc:>10,} {tk.bi_loai:>10,}")
    log.info("=" * 60)
    log.info("✅ TASK 02 hoàn tất. Kiểm tra bất biến:")
    log.info("   wc -l data/processed/train.en data/processed/train.vi")
    log.info("   (số dòng EN phải bằng VI cho mỗi split)")


if __name__ == "__main__":
    main()
