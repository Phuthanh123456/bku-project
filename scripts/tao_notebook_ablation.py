"""Sinh notebooks/03_kaggle_ablation.ipynb — TOÀN BỘ ablation trong một notebook.

SINH BẰNG MÃ, KHÔNG SỬA TAY. Mục 2.7 Sưu tập lỗi.md: vá JSON của .ipynb bằng
script vá tại chỗ đã từng xoá mất nguyên một cell, và cell sau dùng biến chưa
định nghĩa. Muốn đổi notebook thì sửa file này rồi chạy lại.

    python scripts/tao_notebook_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
DUONG_DAN = GOC / "notebooks" / "03_kaggle_ablation.ipynb"

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")


def md(noi_dung: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": noi_dung.strip("\n").splitlines(keepends=True)}


def code(noi_dung: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": noi_dung.strip("\n").splitlines(keepends=True)}


CAC_CELL = [
    md(r"""
# ENVI-NMT — Toàn bộ ablation trên một notebook Kaggle T4

**TASK 17 + TASK 18 + A0 (thí nghiệm thầy đề nghị).**

## Chạy Run All HAI LẦN

| Lượt | Đặt gì ở Cell 2 | Mất bao lâu | Để làm gì |
|---|---|---|---|
| **1** | `SMOKE_TEST = True` | ~10 phút | Chứng minh cả 14 lượt chạy trót lọt. Ghi vào nhánh `smoke/` trên Hub, **không đụng** kết quả thật |
| **2** | `SMOKE_TEST = False` | ~9 giờ mỗi phiên, cần 2 phiên | Chạy thật |

Lượt 1 mà đỏ ở đâu thì **sửa xong hãy sang lượt 2**. Đó là toàn bộ lý do smoke
test tồn tại: bắt lỗi trước khi đốt 18 giờ GPU.

## Ngân sách GPU — vì sao cần 2 phiên

Đo thật ở lượt huấn luyện trước: **1,42 giây/bước** trên T4.

| Nhóm | Thí nghiệm | Số seed | Số lượt |
|---|---|---|---|
| Đối chứng | bản cải tiến | 2 | 2 |
| A0 | vanilla 2017 vs cải tiến | 2 | 2 |
| TASK 17 | A1 LayerNorm · A4 sin-cos · A5 ReLU · A6 Post-Norm | 2 | 8 |
| TASK 18 | A2 warmup · A3 label smoothing | 1 | 2 |
| | | | **14 lượt** |

14 lượt × 3.000 bước × 1,42 giây ≈ **17 giờ**, cộng thời gian chấm BLEU.
Phiên Kaggle bị cắt ở **12 giờ**, quota **30 giờ/tuần**.

Nên `GIO_TOI_DA = 9.0`: chạy được lượt nào hay lượt đó, xong lượt nào là đẩy kết
quả lên Hub ngay. **Phiên sau Run All lại là tự bỏ qua những lượt đã xong.**
Notebook crash giữa chừng cũng không mất gì.

## Trước khi bấm Run All, kiểm đủ 4 thứ

| # | Việc | Chỗ làm |
|---|---|---|
| 1 | **GPU T4** | panel phải → Accelerator → **GPU T4 x2** |
| 2 | **Internet BẬT** | panel phải → Settings → Internet → **On** (mặc định TẮT) |
| 3 | **Dataset mã nguồn** | Add Input → Datasets → bản zip repo |
| 4 | **HF_TOKEN** | Add-ons → Secrets → `HF_TOKEN` loại **Write** → **bật công tắc cho notebook này** |

> Gắn Secret **sau** khi phiên đã khởi động thì phải **Run → Restart session**.

## Upload gì vào Kaggle và đặt tên ra sao

**Dataset 1 — mã nguồn (BẮT BUỘC).**

```
Tên dataset : bku-project-code
Nội dung    : nén cả thư mục repo thành .zip rồi upload
              (bỏ .git, OUTPUT_KAGGEL, output_kaggle_huggingface, *.pt)
```

Nhận ra đúng chưa: trong dataset phải thấy `src/nmt/` và `configs/`.
Notebook quét **4 cấp** trong `/kaggle/input` nên đặt sâu nông đều tìm ra.

**Dataset 2 — dữ liệu và tokenizer (NÊN CÓ, đỡ 10 phút mỗi phiên).**

```
Tên dataset : bku-project-data
Nội dung    : artifacts/tokenizer/tokenizer.json
              data/processed/*.en, *.vi
```

Sinh bằng `python scripts/dong_goi_tiep_tuc.py` ở máy. Không gắn cũng chạy được,
notebook sẽ tự tải và train lại tokenizer.

> **Đừng upload file `.pt` vào dataset.** Checkpoint nằm trên Hugging Face; nhét
> vào dataset chỉ làm nặng và dễ lẫn bản smoke với bản thật.
"""),

    md("## Cell 1 — Cài đặt. TOÀN BỘ thư viện nằm ở đây."),
    code(r'''
# Mọi thư viện cài ở ĐÚNG MỘT CHỖ này. Rải rác giữa notebook thì tới cell thứ 6
# mới phát hiện thiếu gói, mà lúc đó đã tốn hàng giờ GPU.
#
# GHIM PHIÊN BẢN — ba quyết định, mỗi cái có lý do riêng:
#
#   tokenizers==0.21.0   GIỮ NGUYÊN. tokenizer.json trên Hub được train bằng
#                        đúng bản này. Đổi bản đọc là rủi ro không cần thiết.
#   sacrebleu==2.6.0     NÂNG từ 2.4.3. Đây là thư viện chấm điểm, nên dùng bản
#                        hiện hành để chữ ký so được với bài báo mới.
#   huggingface_hub      KHÔNG GHIM. Bản mới nhất là 1.x, tức đã ĐỔI MAJOR so với
#                        0.27 — đúng loại bẫy ở mục 2.4 Sưu tập lỗi. Mà ghim bản
#                        cũ lại ép Kaggle hạ cấp gói nó cài sẵn, dễ xung đột.
#                        Đường an toàn: dùng bản của Kaggle, rồi KIỂM ở dưới xem
#                        mấy hàm mình cần có còn tồn tại không.
!pip install -q "tokenizers==0.21.0" "sacrebleu==2.6.0" "PyYAML>=6.0.2"

import importlib

print("Phiên bản thật đang dùng:")
for ten in ["torch", "numpy", "tokenizers", "sacrebleu", "huggingface_hub",
            "yaml", "matplotlib", "pandas"]:
    try:
        m = importlib.import_module(ten)
        print(f"  {ten:18s} {getattr(m, '__version__', '(không có __version__)')}")
    except ImportError as loi:
        print(f"  {ten:18s} THIẾU — {loi}")
        raise

# Không tin lời hứa của số phiên bản — kiểm thẳng mấy hàm mình thật sự gọi.
# Thiếu một hàm thì hỏng ở cell thứ 6, sau khi đã tiêu vài giờ GPU.
import huggingface_hub as _hf

for _ten_ham in ("HfApi", "hf_hub_download"):
    if not hasattr(_hf, _ten_ham):
        raise ImportError(
            f"huggingface_hub {_hf.__version__} không có {_ten_ham}. "
            "API đã đổi, phải sửa src/nmt/training/hub_sync.py trước khi chạy tiếp."
        )
print(f"\nhuggingface_hub {_hf.__version__}: đủ hàm cần dùng.")
print("Cài đặt xong.")
'''),

    md(r"""
## Cell 2 — CẤU HÌNH. Mọi công tắc nằm ở đây.

Đây là cell duy nhất cần sửa tay.
"""),
    code(r'''
# ===================== CÔNG TẮC =====================

SMOKE_TEST = True      # True: 14 lượt x 60 bước, ~10 phút. False: chạy thật.

REPO_HUB = "mgbao/envi-nmt-scratch-transformer"    # ĐỔI THÀNH TÀI KHOẢN HF CỦA CẬU

# NGÂN SÁCH BƯỚC — dùng chung cho MỌI thí nghiệm.
# Đây là con số quan trọng nhất của cả notebook: ablation chỉ có nghĩa khi mọi
# cấu hình tiêu đúng một ngân sách như nhau. Đổi số này thì phải chạy lại TẤT CẢ,
# không được trộn kết quả của hai ngân sách khác nhau vào cùng một bảng.
#
#   1.500 bước  ~9,6 giờ tổng   BLEU còn quá thấp để so, không khuyên
#   3.000 bước  ~17 giờ tổng    cân bằng, mặc định
#   4.000 bước  ~23 giờ tổng    sát quota 30 giờ/tuần
NGAN_SACH_BUOC = 3000

# Ngân sách giờ CỦA PHIÊN NÀY. Phiên Kaggle bị cắt ở 12 giờ; để 9 giờ thì còn dư
# thời gian đẩy kết quả lên Hub và thoát sạch.
GIO_TOI_DA = 9.0

# Chạy lại chỉ vài thí nghiệm: ["doi_chung", "a0"]. None = chạy hết.
CHI_THI_NGHIEM = None

# ====================================================

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Mục 1.3 của Sưu tập lỗi.md: /kaggle/input CHỈ ĐỌC. Viết theo tư duy Colab, nơi
# một thư mục vừa là nguồn vừa là chỗ ghi, sẽ chết ngay ở lệnh tạo thư mục đầu.
IS_KAGGLE = os.path.isdir("/kaggle/working")
OUT_BASE = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()

# Mục 1.2: độ sâu dataset trong /kaggle/input KHÔNG cố định. Quét 4 cấp thay vì
# đoán, và in ra những gì đã quét để còn đối chiếu.
CAC_MAU = ["/kaggle/input/*", "/kaggle/input/*/*",
           "/kaggle/input/*/*/*", "/kaggle/input/*/*/*/*"]


def tim_goc_repo():
    if os.environ.get("NMT_ROOT"):
        return Path(os.environ["NMT_ROOT"])
    for mau in CAC_MAU:
        for duong_dan in sorted(glob.glob(mau)):
            if (Path(duong_dan) / "src" / "nmt").is_dir():
                return Path(duong_dan)
    return None


REPO = tim_goc_repo()
if REPO is None and (Path.cwd() / "src" / "nmt").is_dir():
    REPO = Path.cwd()

if REPO is None:
    print("KHÔNG TÌM THẤY repo. Các thư mục đã quét:")
    for mau in CAC_MAU:
        for d in sorted(glob.glob(mau))[:40]:
            print("   ", d)
    raise SystemExit("Add Input > Datasets > thêm dataset 'bku-project-code', "
                     "hoặc đặt biến môi trường NMT_ROOT.")

# Repo nằm trong /kaggle/input thì CHỈ ĐỌC, mà script cần ghi results/ và
# artifacts/. Chép sang chỗ ghi được rồi chạy ở đó.
if IS_KAGGLE and str(REPO).startswith("/kaggle/input"):
    DICH = OUT_BASE / "bku-project"
    if not DICH.exists():
        shutil.copytree(REPO, DICH)
    REPO = DICH

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))


def gop_dataset_du_lieu():
    """Gộp dataset dữ liệu/tokenizer (nếu có gắn) vào bản làm việc.

    Thiếu bước này thì notebook phải tải và train lại tokenizer mỗi phiên, mất
    khoảng 10 phút GPU cho một việc đã làm xong từ lâu.
    """
    da_gop = []
    for mau in CAC_MAU:
        for d in sorted(glob.glob(mau)):
            p = Path(d)
            co_du_lieu = ((p / "artifacts" / "tokenizer").is_dir()
                          or (p / "data" / "processed").is_dir())
            if not co_du_lieu or (p / "src" / "nmt").is_dir():
                continue
            for nhanh in ("artifacts", "data"):
                if (p / nhanh).is_dir():
                    shutil.copytree(p / nhanh, REPO / nhanh, dirs_exist_ok=True)
                    da_gop.append(f"{p.name}/{nhanh}")
    return da_gop


DA_GOP = gop_dataset_du_lieu()
print("Đã gộp dataset dữ liệu:", ", ".join(DA_GOP) if DA_GOP
      else "(không có — sẽ tự tải và train tokenizer)")


def chay(lenh, mo_ta=""):
    """Chạy một lệnh, in log trực tiếp, dừng notebook nếu lệnh thất bại."""
    print(f"\n{'=' * 70}\n$ {lenh}\n{'=' * 70}", flush=True)
    ket_qua = subprocess.run(lenh, shell=True)
    if ket_qua.returncode != 0:
        raise RuntimeError(f"Lệnh thất bại (mã {ket_qua.returncode}): {mo_ta or lenh}")
    return ket_qua


SO_BUOC = 60 if SMOKE_TEST else NGAN_SACH_BUOC
print(f"\nCHẾ ĐỘ        : {'SMOKE TEST' if SMOKE_TEST else 'CHẠY THẬT'}")
print(f"NGÂN SÁCH BƯỚC: {SO_BUOC:,} cho mỗi lượt".replace(",", "."))
print(f"GIỜ TỐI ĐA    : {GIO_TOI_DA}")
print(f"IS_KAGGLE     : {IS_KAGGLE}")
print(f"REPO          : {REPO}   (ghi được: {os.access(REPO, os.W_OK)})")
print(f"Nhánh Hub     : {'smoke/ablation/...' if SMOKE_TEST else 'ablation/...'}")
'''),

    md(r"""
## Cell 3 — Kiểm GPU, token, và THỬ GHI THẬT lên Hub

Cell này tốn 10 giây và nó thay thế cho bài học đắt nhất của nhóm: một lượt chạy
13 tiếng đã mất trắng vì repo trên Hub chưa được tạo, nên mọi lần đẩy checkpoint
đều thất bại âm thầm suốt cả lượt.
"""),
    code(r'''
import torch

print(f"CUDA có sẵn : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    kha_nang = torch.cuda.get_device_capability()
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Compute cap : {kha_nang}  "
          f"({'có bf16' if kha_nang[0] >= 8 else 'CHỈ fp16 — T4 là Turing'})")
    print(f"Số GPU      : {torch.cuda.device_count()}")
else:
    print("KHÔNG CÓ GPU. Panel phải > Accelerator > GPU T4 x2, rồi Restart session.")

# Mục 1.12: Kaggle cấp T4 x2 nên thư viện dễ tự bọc DataParallel. Đồ án này
# huấn luyện một GPU, ép rõ ràng để không phải đoán.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- token và ghi thử ---
from nmt.training.hub_sync import doc_token

TOKEN = doc_token()
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id=REPO_HUB, token=TOKEN, private=True, exist_ok=True)

# GHI THỬ MỘT FILE THẬT. Tạo repo thành công chưa chứng minh được là ghi được:
# token chỉ-đọc vẫn qua được bước tạo repo.
Path("kiem_tra_ghi_ablation.txt").write_text("ok", encoding="utf-8")
api.upload_file(path_or_fileobj="kiem_tra_ghi_ablation.txt",
                path_in_repo="kiem_tra_ghi_ablation.txt",
                repo_id=REPO_HUB, token=TOKEN,
                commit_message="thử quyền ghi trước khi chạy ablation")
print(f"\nGHI THỬ LÊN HUB: THÀNH CÔNG -> {REPO_HUB}")
'''),

    md("## Cell 4 — Dữ liệu và tokenizer"),
    code(r'''
# CỐ Ý dùng base.yaml ở hai lệnh này. Dữ liệu và tokenizer DÙNG CHUNG cho mọi
# thí nghiệm — đó là điều kiện để ablation so sánh được. Mỗi cấu hình tự train
# một tokenizer riêng thì token ID khác nhau và mọi so sánh mất ý nghĩa.
if not Path("data/processed/train.en").exists():
    chay("python scripts/prepare_data.py --config configs/base.yaml", "chuẩn bị dữ liệu")
else:
    print("Dữ liệu đã có, bỏ qua bước tải.")

if not Path("artifacts/tokenizer/tokenizer.json").exists():
    chay("python scripts/train_tokenizer.py --config configs/base.yaml", "train tokenizer")
else:
    print("Tokenizer đã có, bỏ qua.")

for f in ["data/processed/train.en", "data/processed/tst2012.en",
          "data/processed/tst2013.en", "artifacts/tokenizer/tokenizer.json"]:
    p = Path(f)
    print(f"  {'OK ' if p.exists() else 'THIẾU'} {f}"
          f"{f'  ({p.stat().st_size / 1024:.0f} KB)' if p.exists() else ''}")
'''),

    md(r"""
## Cell 5 — Chạy TOÀN BỘ ablation

Một lệnh chạy hết 14 lượt. Xong lượt nào là ghi kết quả và đẩy lên Hub ngay,
rồi **xoá checkpoint cục bộ** — `/kaggle/working` chỉ có 20 GB mà 14 lượt
checkpoint là 15,4 GB (mục 1.10 Sưu tập lỗi).

Hết giờ hoặc notebook crash thì **Run All lại**: những lượt đã xong tự được bỏ qua.
"""),
    code(r'''
lenh = (f"python scripts/chay_ablation.py --so-buoc {NGAN_SACH_BUOC} "
        f"--repo-hub {REPO_HUB} --gio-toi-da {GIO_TOI_DA}")
if SMOKE_TEST:
    lenh += " --smoke"
if CHI_THI_NGHIEM:
    lenh += " --chi-thi-nghiem " + " ".join(CHI_THI_NGHIEM)

chay(lenh, "toàn bộ ablation")
'''),

    md(r"""
## Cell 6 — Dựng bảng, hình và báo cáo

Sinh `docs/ablation.md` cùng các file hình gốc. Hình theo phong cách nghiên cứu
khoa học: **trên hình chỉ có dữ liệu**, mọi câu kết luận nằm ở phần chữ bên dưới.
Xuất cả PNG 300 dpi (dán Word) lẫn PDF vector (dán slide, phóng to không vỡ).
"""),
    code(r'''
from nmt.eval.bao_cao_ablation import sinh_bao_cao

duong_dan = sinh_bao_cao(Path.cwd(), so_buoc=NGAN_SACH_BUOC)
print(f"Đã sinh: {duong_dan}\n")

for p in sorted(Path("results/ablation/hinh").glob("*")):
    print(f"  {p}  ({p.stat().st_size / 1024:.0f} KB)")

print("\n" + "=" * 70)
print(duong_dan.read_text(encoding="utf-8"))
'''),

    md("## Cell 7 — Đẩy kết quả lên Hub và tổng kết"),
    code(r'''
import pandas as pd
from nmt.training.checkpoint import CHE_DO_SMOKE, CHE_DO_THAT
from nmt.training.hub_sync import day_thu_muc_len_hub, liet_ke_file

CHE_DO = CHE_DO_SMOKE if SMOKE_TEST else CHE_DO_THAT

# Smoke rơi vào `smoke/...` nhờ _tien_to_theo_che_do, nên chạy smoke bao nhiêu
# lần cũng không đụng kết quả thật (mục 1.8).
day_thu_muc_len_hub("results/ablation", REPO_HUB, "ablation", che_do=CHE_DO)
day_thu_muc_len_hub("results/logs", REPO_HUB, "logs", che_do=CHE_DO)
day_thu_muc_len_hub("docs", REPO_HUB, "docs", che_do=CHE_DO)

bang = Path("results/ablation/ket_qua.csv")
if bang.exists():
    df = pd.read_csv(bang)
    print(f"\nĐÃ XONG {len(df)}/14 lượt\n")
    print(df[["ma_thi_nghiem", "seed", "so_buoc", "loss_dev",
              "bleu_test", "chrf_test"]].to_string(index=False))
    con_thieu = 14 - len(df)
    if con_thieu > 0:
        print(f"\nCÒN {con_thieu} LƯỢT CHƯA CHẠY. Run All lại ở phiên sau — "
              "những lượt đã xong sẽ tự được bỏ qua.")
    else:
        print("\nĐỦ 14 LƯỢT. Ablation hoàn tất.")

print(f"\nFile trên Hub ({REPO_HUB}):")
for f in sorted(liet_ke_file(REPO_HUB)):
    if "ablation" in f or f.endswith(".md"):
        print("  ", f)
'''),
]


def main() -> None:
    notebook = {
        "cells": CAC_CELL,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DUONG_DAN.parent.mkdir(parents=True, exist_ok=True)
    DUONG_DAN.write_text(json.dumps(notebook, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    so_code = sum(1 for c in CAC_CELL if c["cell_type"] == "code")
    print(f"Đã sinh {DUONG_DAN.relative_to(GOC)}")
    print(f"  {len(CAC_CELL)} cell ({so_code} cell mã, "
          f"{len(CAC_CELL) - so_code} cell chữ)")


if __name__ == "__main__":
    main()
