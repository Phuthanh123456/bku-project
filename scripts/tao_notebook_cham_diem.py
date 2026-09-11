"""Sinh notebook Kaggle CHẤM ĐIỂM — đóng TASK 16 và TASK 19.

Vì sao tách hẳn khỏi notebook ablation: chấm điểm KHÔNG cần huấn luyện. Nó chỉ
nạp checkpoint đã có trên Hub rồi dịch, nên tốn khoảng 20 phút GPU thay vì 2,5
giờ. Gộp chung vào notebook ablation là bắt người chạy phải đợi cả lượt huấn
luyện chỉ để lấy vài con số.

    TASK 16  Greedy + BLEU/chrF++ trên tst2013, ghi nguyên văn chuỗi chữ ký
             XONG KHI: BLEU >= 19 (tốt từ 22)
    TASK 19  Beam search 4 + KV cache
             XONG KHI: beam cao hơn greedy ÍT NHẤT 0,5 BLEU

Notebook chạy đủ bốn tổ hợp (greedy/beam x dev/test) rồi tự đối chiếu với hai
ngưỡng trên, và in ra đúng khối LaTeX để dán vào slide. Không phải chép tay số
từ log — chép tay là chỗ sinh ra sai lệch giữa slide và bảng kết quả.

Chạy:  python scripts/tao_notebook_cham_diem.py
"""

from __future__ import annotations

import json
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
DUONG_DAN = GOC / "notebooks" / "05_kaggle_cham_diem.ipynb"


def md(nguon: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": nguon.strip().splitlines(keepends=True)}


def code(nguon: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": nguon.strip().splitlines(keepends=True)}


CAC_CELL = [
    md(r'''
# ENVI-NMT — Chấm điểm: TASK 16 (Greedy) + TASK 19 (Beam + KV cache)

**Notebook này KHÔNG huấn luyện gì cả.** Nó nạp checkpoint đã có sẵn trên Hugging
Face rồi dịch, nên tốn khoảng **20 phút GPU** chứ không phải 2,5 giờ.

## Hai tiêu chí cần đóng

| Task | Việc | XONG KHI |
|---|---|---|
| **16** | Greedy, chấm BLEU + chrF++ trên tst2012 và tst2013, ghi **nguyên văn chuỗi chữ ký** sacrebleu | BLEU trên tst2013 $\ge 19$ (tốt từ 22) |
| **19** | Beam search $=4$ kèm KV cache, so với Greedy cả điểm lẫn thời gian | Beam cao hơn Greedy **ít nhất 0,5 BLEU** |

Notebook chạy đủ **bốn tổ hợp** (greedy/beam × dev/test), tự đối chiếu với hai
ngưỡng trên, rồi in ra **đúng khối LaTeX** để dán vào slide.

> Không chép tay số từ log sang slide. Chép tay chính là chỗ sinh ra sai lệch
> giữa slide và bảng kết quả, mà loại sai đó không ai phát hiện được khi đọc.

## Trước khi bấm Run All

| # | Việc | Chỗ làm |
|---|---|---|
| 1 | GPU T4 | panel phải → Accelerator → GPU T4 x2 |
| 2 | Dataset `bku-project-code` | Add Input → Datasets |
| 3 | Dataset `bku-project-data` | Add Input → Datasets |
| 4 | Secret `HF_TOKEN` (quyền **ghi**) | Add-ons → Secrets |
'''),

    md("## Cell 1 — Cài đặt. TOÀN BỘ thư viện nằm ở đây."),
    code(r'''
# Mọi thư viện cài ở ĐÚNG MỘT CHỖ này. Rải rác giữa notebook thì tới cell cuối
# mới phát hiện thiếu gói, mà lúc đó đã tốn hàng chục phút GPU.
#
# Ghim bản y hệt notebook ablation — bảng điểm của hai notebook phải so được
# với nhau, mà chữ ký sacrebleu có ghi cả số phiên bản vào trong.
!pip install -q "tokenizers==0.21.0" "sacrebleu==2.6.0" "PyYAML>=6.0.2"

import importlib

print("Phiên bản thật đang dùng:")
for ten in ["torch", "numpy", "tokenizers", "sacrebleu", "huggingface_hub",
            "yaml", "pandas"]:
    try:
        m = importlib.import_module(ten)
        print(f"  {ten:18s} {getattr(m, '__version__', '(không có __version__)')}")
    except ImportError as loi:
        print(f"  {ten:18s} THIẾU — {loi}")
        raise

# Không tin lời hứa của số phiên bản — kiểm thẳng mấy hàm mình thật sự gọi.
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

    md("## Cell 2 — CẤU HÌNH. Mọi công tắc nằm ở đây."),
    code(r'''
# ===================== CÔNG TẮC =====================

REPO_HUB = "mgbao/envi-nmt-scratch-transformer"   # ĐỔI THÀNH TÀI KHOẢN HF CỦA CẬU

# Checkpoint đem đi chấm. Mặc định là lượt ĐỐI CHỨNG của ablation — lượt duy
# nhất vừa huấn luyện xong đàng hoàng vừa có số đã lên slide (BLEU 28,74).
TEN_CHAY = "abl3000_base_seed42"

# Cấu hình phải KHỚP với checkpoint. Đối chứng dùng base.yaml; đổi checkpoint
# sang lượt A1 thì phải đổi luôn dòng này sang configs/ablation_a1_layernorm.yaml,
# nếu không mô hình dựng lên sai kiến trúc và nạp trọng số sẽ lỗi.
DUONG_DAN_CAU_HINH = "configs/base.yaml"

BEAM_SIZE = 4          # TASK 19 yêu cầu beam = 4
SO_CAU_VI_DU = 12      # ghi ra bảng câu dịch ví dụ cho báo cáo

# 0 = chấm ĐỦ cả tập (kết quả thật). Đặt 64 để chạy thử đường ống trong 2 phút
# trước khi chấm thật — nhưng điểm lúc đó KHÔNG dùng được.
GIOI_HAN_CAU = 0

# Chạy cổng chặn "học thuộc 50 câu" với cấu hình A0 vanilla.
# Đây là câu hỏi còn treo: A0 không huấn luyện được là do LỖI MÃ trong nhánh
# vanilla (Post-Norm, sin-cos, ReLU), hay chỉ vì Post-Norm khó huấn luyện ở
# ngân sách 3.000 bước? Kiến trúc nào không học thuộc nổi 50 câu là kiến trúc
# còn sai. Tốn thêm khoảng 3 phút GPU, đáng để biết.
CHAY_CONG_CHAN_A0 = True

# ====================================================

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Mục 1.3 Sưu tập lỗi: /kaggle/input CHỈ ĐỌC.
IS_KAGGLE = os.path.isdir("/kaggle/working")
OUT_BASE = Path("/kaggle/working") if IS_KAGGLE else Path.cwd()

# Mục 1.2: độ sâu dataset trong /kaggle/input KHÔNG cố định. Quét 4 cấp thay vì đoán.
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

if IS_KAGGLE and str(REPO).startswith("/kaggle/input"):
    DICH = OUT_BASE / "bku-project"
    if not DICH.exists():
        shutil.copytree(REPO, DICH)
    REPO = DICH

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))


def gop_dataset_du_lieu():
    """Gộp dataset dữ liệu/tokenizer đã gắn vào bản làm việc.

    Thiếu bước này thì notebook phải tải và train lại tokenizer, mất khoảng 10
    phút GPU cho một việc đã làm xong từ lâu. Tệ hơn: tokenizer train lại có thể
    ra token ID khác, và điểm chấm được sẽ KHÔNG so được với bảng ablation.
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


for _m in gop_dataset_du_lieu():
    print(f"  đã gộp {_m}")


def chay(lenh: str, ten_viec: str) -> int:
    """Chạy lệnh con, in thẳng log ra ngoài, ném lỗi nếu thất bại.

    In từng dòng ngay khi có thay vì gom lại cuối: lượt dịch kéo dài nhiều phút,
    không thấy gì chạy thì không biết nó đang làm việc hay đã treo.
    """
    print(f"\n{'=' * 78}\n>>> {ten_viec}\n    {lenh}\n{'=' * 78}", flush=True)
    bat_dau = time.perf_counter()
    tt = subprocess.run(lenh, shell=True)
    giay = time.perf_counter() - bat_dau
    if tt.returncode != 0:
        raise RuntimeError(f"{ten_viec} THẤT BẠI (mã {tt.returncode}) sau "
                           f"{giay / 60:.1f} phút")
    print(f"<<< {ten_viec} xong sau {giay / 60:.1f} phút", flush=True)
    return tt.returncode


print(f"\nRepo: {REPO}")
print(f"Checkpoint sẽ chấm: {TEN_CHAY}")
print(f"Cấu hình: {DUONG_DAN_CAU_HINH}")
if GIOI_HAN_CAU:
    print(f"\n*** CHỈ CHẤM {GIOI_HAN_CAU} CÂU — điểm ra KHÔNG phải kết quả thật ***")
'''),

    md("## Cell 3 — Kiểm GPU và token"),
    code(r'''
import torch

print(f"CUDA có sẵn : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    kha_nang = torch.cuda.get_device_capability()
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"Compute cap : {kha_nang}  "
          f"({'có bf16' if kha_nang[0] >= 8 else 'CHỈ fp16 — T4 là Turing'})")
else:
    print("KHÔNG CÓ GPU. Panel phải > Accelerator > GPU T4 x2, rồi Restart session.")
    print("Chấm điểm trên CPU sẽ mất hàng giờ thay vì 20 phút.")

# Mục 1.12: Kaggle cấp T4 x2 nên thư viện dễ tự bọc DataParallel. Ép rõ một GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from nmt.training.hub_sync import doc_token

TOKEN = doc_token()
print("\nĐã đọc được HF token.")
'''),

    md("## Cell 4 — Dữ liệu và tokenizer"),
    code(r'''
# CỐ Ý dùng base.yaml: dữ liệu và tokenizer DÙNG CHUNG cho mọi lượt. Chấm điểm
# bằng tokenizer khác lượt huấn luyện là ra số vô nghĩa mà không báo lỗi.
if not Path("data/processed/tst2013.en").exists():
    chay("python scripts/prepare_data.py --config configs/base.yaml", "chuẩn bị dữ liệu")
else:
    print("Dữ liệu đã có, bỏ qua bước tải.")

if not Path("artifacts/tokenizer/tokenizer.json").exists():
    chay("python scripts/train_tokenizer.py --config configs/base.yaml", "train tokenizer")
else:
    print("Tokenizer đã có, bỏ qua.")

for f in ["data/processed/tst2012.en", "data/processed/tst2012.vi",
          "data/processed/tst2013.en", "data/processed/tst2013.vi",
          "artifacts/tokenizer/tokenizer.json"]:
    p = Path(f)
    print(f"  {'OK ' if p.exists() else 'THIẾU'} {f}"
          f"{f'  ({p.stat().st_size / 1024:.0f} KB)' if p.exists() else ''}")
'''),

    md("## Cell 5 — Kéo checkpoint từ Hub về"),
    code(r'''
from nmt.training.hub_sync import tai_file, liet_ke_file

DICH_CK = Path("artifacts/checkpoints") / TEN_CHAY
DICH_CK.mkdir(parents=True, exist_ok=True)
CHECKPOINT = DICH_CK / "tot_nhat.pt"

if CHECKPOINT.exists():
    print(f"Checkpoint đã có sẵn: {CHECKPOINT} "
          f"({CHECKPOINT.stat().st_size / 1024**2:.0f} MB)")
else:
    ten_hub = f"checkpoints/{TEN_CHAY}/tot_nhat.pt"

    # LIỆT KÊ TRƯỚC KHI KẾT LUẬN "Hub chưa có gì" — mục 1.7 Sưu tập lỗi. Lần
    # trước kết luận nhầm rồi train lại 3 tiếng cho một thứ đã nằm sẵn trên Hub.
    co_tren_hub = [f for f in liet_ke_file(REPO_HUB) if TEN_CHAY in f]
    if not co_tren_hub:
        raise SystemExit(
            f"Trên Hub không có file nào của {TEN_CHAY}.\n"
            f"Kiểm lại TEN_CHAY ở Cell 2. Các lượt đang có trên Hub:\n  "
            + "\n  ".join(sorted({f.split('/')[1] for f in liet_ke_file(REPO_HUB)
                                  if f.startswith('checkpoints/')}))
        )

    # tai_file GIỮ NGUYÊN cấu trúc thư mục của Hub, nên tải về chỗ tạm rồi chép
    # sang đúng vị trí thay vì đoán đường dẫn nó trả về.
    tam = Path(".hub_tam")
    ve = tai_file(REPO_HUB, ten_hub, tam)
    if not ve or not Path(ve).exists():
        raise SystemExit(f"Kéo {ten_hub} từ Hub thất bại.")
    shutil.copy2(ve, CHECKPOINT)
    shutil.rmtree(tam, ignore_errors=True)
    print(f"Đã kéo về: {CHECKPOINT} "
          f"({CHECKPOINT.stat().st_size / 1024**2:.0f} MB)")

# Bảng điểm cũ có thể lẫn số của lượt khác. Dọn đi để Cell 7 chỉ đọc đúng số
# vừa chấm — lẫn số cũ thì bảng vẫn ra, vẫn trông hợp lý, và không ai phát hiện.
cu = Path("results/diem_chinh.csv")
if cu.exists():
    cu.rename("results/diem_chinh_truoc_khi_cham.csv")
    print("Đã dời bảng điểm cũ sang results/diem_chinh_truoc_khi_cham.csv")
'''),

    md(r'''
## Cell 6 — Chấm bốn tổ hợp

`greedy` và `beam` × `dev` và `test`. Đây là phần tốn thời gian nhất, khoảng
15–20 phút. Beam chậm hơn greedy khoảng 4 lần vì phải giữ 4 giả thuyết cùng lúc.
'''),
    code(r'''
gh = f" --gioi-han-cau {GIOI_HAN_CAU}" if GIOI_HAN_CAU else ""

for split in ["dev", "test"]:
    for cach in ["greedy", "beam"]:
        lenh = (f"python scripts/evaluate.py"
                f" --config {DUONG_DAN_CAU_HINH}"
                f" --checkpoint {CHECKPOINT}"
                f" --split {split}"
                f" --cach {cach}"
                f" --luu-vi-du {SO_CAU_VI_DU}"
                + (f" --beam-size {BEAM_SIZE}" if cach == "beam" else "")
                + gh)
        chay(lenh, f"CHẤM {cach.upper()} · {split}")

print("\nĐã chấm xong cả bốn tổ hợp.")
'''),

    md("## Cell 7 — Bảng so sánh và đối chiếu hai tiêu chí"),
    code(r'''
import pandas as pd

d = pd.read_csv("results/diem_chinh.csv")
d = d[d["Checkpoint"].astype(str).str.contains(TEN_CHAY, regex=False, na=False)]

print("=" * 78)
print("BẢNG ĐIỂM ĐẦY ĐỦ")
print("=" * 78)
print(d[["Tập test", "Cách sinh", "Số câu", "BLEU", "chrF++",
         "Giây dịch"]].to_string(index=False))

print("\nCHỮ KÝ SACREBLEU — phải chép nguyên văn vào báo cáo (TASK 16):")
print(f"  BLEU   {d['Chữ ký BLEU'].iloc[0]}")
print(f"  chrF++ {d['Chữ ký chrF++'].iloc[0]}")


def lay(split, cach_chua):
    """Lấy dòng điểm theo split và cách sinh.

    Lọc theo CHUỖI CON của cột 'Cách sinh' vì cột đó ghi cả beam size
    (ví dụ 'beam=4'), không phải đúng chữ 'beam'.
    """
    k = d[(d["Tập test"] == split)
          & (d["Cách sinh"].astype(str).str.contains(cach_chua, case=False,
                                                     regex=False, na=False))]
    return None if k.empty else k.iloc[-1]


print("\n" + "=" * 78)
print("ĐỐI CHIẾU TIÊU CHÍ")
print("=" * 78)

dat_het = True

# ---- TASK 16 ----
g_test = lay("test", "greedy")
if g_test is None:
    print("TASK 16: KHÔNG CÓ dòng greedy/test — chấm điểm đã hỏng.")
    dat_het = False
else:
    b = float(g_test["BLEU"])
    muc = "rất tốt" if b >= 29 else "tốt" if b >= 22 else "đạt" if b >= 19 else "CHƯA ĐẠT"
    print(f"TASK 16  Greedy BLEU trên tst2013 = {b:.2f}  (ngưỡng 19 · tốt 22 · "
          f"rất tốt 29)  ->  {muc}")
    dat_het &= b >= 19

# ---- TASK 19 ----
b_test = lay("test", "beam")
if g_test is None or b_test is None:
    print("TASK 19: THIẾU dòng greedy hoặc beam trên test.")
    dat_het = False
else:
    chenh = float(b_test["BLEU"]) - float(g_test["BLEU"])
    nhanh = float(g_test["Giây dịch"]) and float(b_test["Giây dịch"]) / float(g_test["Giây dịch"])
    print(f"TASK 19  Beam {float(b_test['BLEU']):.2f} − Greedy "
          f"{float(g_test['BLEU']):.2f} = {chenh:+.2f} BLEU  (cần >= +0,50)  ->  "
          f"{'ĐẠT' if chenh >= 0.5 else 'CHƯA ĐẠT'}")
    print(f"         Beam chậm hơn greedy {nhanh:.1f} lần "
          f"({float(g_test['Giây dịch']):.0f}s -> {float(b_test['Giây dịch']):.0f}s)")
    dat_het &= chenh >= 0.5

    if chenh < 0.5:
        print("\n  Beam không hơn greedy đủ 0,5 điểm KHÔNG phải lúc nào cũng là lỗi.")
        print("  Với mô hình mới huấn luyện 3.000 bước thì phân phối còn phẳng,")
        print("  beam ít có chỗ phát huy. Báo cáo con số thật kèm giải thích này,")
        print("  ĐỪNG chỉnh beam size cho tới khi vượt ngưỡng — đó là bịa kết quả.")

print("\n" + ("=" * 78))
print("ĐỦ CẢ HAI TIÊU CHÍ." if dat_het else "CHƯA ĐỦ — đọc phần trên.")
print("=" * 78)
'''),

    md(r'''
## Cell 8 — Cổng chặn A0: lỗi mã hay chỉ khó huấn luyện?

Câu hỏi còn treo. A0 vanilla đã hỏng ba lượt liên tiếp, loss kẹt ở 6,9 và BLEU
0,00. Có hai khả năng, và chúng cần cách xử lý khác hẳn nhau:

- **Lỗi mã** trong nhánh vanilla (Post-Norm, sin-cos, ReLU) → phải sửa mã
- **Không lỗi**, Post-Norm đơn giản cần warmup dài hơn cả ngân sách 3.000 bước
  → đây là kết quả đáng báo cáo, và nó ủng hộ đúng lựa chọn Pre-Norm của nhóm
  (Xiong và cộng sự, 2020 — tài liệu tham khảo [5])

Cổng chặn học thuộc 50 câu phân biệt được hai thứ đó: **kiến trúc nào không học
thuộc nổi 50 câu là kiến trúc còn sai.**
'''),
    code(r'''
if not CHAY_CONG_CHAN_A0:
    print("Bỏ qua (CHAY_CONG_CHAN_A0 = False).")
else:
    # KHÔNG để cổng chặn trượt làm chết cả notebook.
    #
    # overfit_sanity.py thoát với mã khác 0 khi gate trượt, mà chay() ném lỗi
    # khi gặp mã khác 0 — nên Cell 9 KHÔNG BAO GIỜ CHẠY: không đẩy kết quả lên
    # Hub, không in khối LaTeX. Lượt 11/09 dính đúng chuyện này, và trớ trêu là
    # gate "trượt" vì loss 0,0622 so với ngưỡng 0,05 đúng lúc chạm trần 500
    # bước, trong khi BLEU trên chính 50 câu đó đã 100,00. Tức kiến trúc ĐÚNG,
    # chỉ là hết bước.
    #
    # Cổng chặn ở đây là thông tin chẩn đoán, không phải điều kiện chấm điểm.
    try:
        chay("python scripts/overfit_sanity.py --config configs/ablation_a0_vanilla.yaml",
             "CỔNG CHẶN — A0 vanilla học thuộc 50 câu")
    except RuntimeError as loi:
        print(f"\n[cổng chặn] Thoát với mã lỗi: {loi}")
        print("[cổng chặn] KHÔNG dừng notebook — đọc bảng kết quả ngay trên đây.")

    print("\n" + "=" * 78)
    print("ĐỌC KẾT QUẢ NÀY THẾ NÀO")
    print("=" * 78)
    print("  BLEU > 90 trên chính 50 câu đó  ->  kiến trúc vanilla ĐÚNG, không")
    print("     có lỗi mã. Kể cả khi loss chưa xuống dưới 0,05: nếu đường loss")
    print("     vẫn đang giảm đều tới bước cuối thì đó là CHẠM TRẦN 500 BƯỚC,")
    print("     không phải kiến trúc sai. A0 hỏng ở ablation là do Post-Norm cần")
    print("     warmup dài hơn cả ngân sách 3.000 bước — đúng điều Xiong và cộng")
    print("     sự (2020) chỉ ra, và đó là một kết quả ĐÁNG BÁO CÁO.")
    print()
    print("  loss chững ở 2-3 và BLEU thấp  ->  CÓ LỖI MÃ trong nhánh vanilla.")
    print("     Nghi trước: Post-Norm đặt sai chỗ, bảng sin-cos, hay việc bỏ")
    print("     chuẩn hóa cuối. Phải sửa mã rồi mới chạy lại A0.")
'''),

    md("## Cell 9 — Đẩy lên Hub và in khối LaTeX để dán vào slide"),
    code(r'''
from nmt.training.checkpoint import CHE_DO_THAT
from nmt.training.hub_sync import day_len_hub

for f in sorted(glob.glob("results/diem_chinh.csv")
                + glob.glob("results/vi_du_dich_*.csv")
                + glob.glob("results/overfit_*.csv")):
    try:
        day_len_hub(Path(f), REPO_HUB, f"results/{Path(f).name}",
                    che_do=CHE_DO_THAT, ghi_chu="chấm điểm TASK 16 + 19")
        print(f"  đã đẩy {f}")
    except Exception as loi:
        print(f"  đẩy {f} thất bại: {type(loi).__name__}: {loi}")

# In sẵn khối LaTeX thay vì bắt người dùng chép tay từng số. Chép tay là chỗ
# sinh ra sai lệch giữa slide và bảng kết quả, và loại sai đó không ai phát
# hiện được khi đọc slide.
if g_test is not None and b_test is not None:
    g_dev, b_dev = lay("dev", "greedy"), lay("dev", "beam")
    print("\n" + "=" * 78)
    print("DÁN THẲNG VÀO slide 17 CỦA main.tex")
    print("=" * 78)
    print(r"\begin{tabular}{@{}llcc@{}}")
    print(r"  \toprule")
    print(r"  \textbf{Tập} & \textbf{Sinh câu} & \textbf{BLEU} & \textbf{chrF++} \\")
    print(r"  \midrule")
    for ten, hang in [("tst2012 (dev)", g_dev), ("tst2012 (dev)", b_dev),
                      ("tst2013 (test)", g_test), ("tst2013 (test)", b_test)]:
        if hang is None:
            continue
        cach = str(hang["Cách sinh"]).replace("=", r"$=$")
        bleu = f"{float(hang['BLEU']):.2f}".replace(".", "{,}")
        chrf = f"{float(hang['chrF++']):.2f}".replace(".", "{,}")
        print(f"  {ten} & {cach} & {bleu} & {chrf} \\\\")
    print(r"  \bottomrule")
    print(r"\end{tabular}")
    print()
    print(f"% Chữ ký BLEU:   {g_test['Chữ ký BLEU']}")
    print(f"% Chữ ký chrF++: {g_test['Chữ ký chrF++']}")
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

    # Cùng cửa chặn như bộ sinh notebook ablation: gán trùng một biến cấu hình
    # ở cấp ngoài cùng thì cái dưới đè cái trên, người dùng sửa dòng trên rồi
    # Run All và notebook im lặng chạy theo giá trị khác.
    for so_cell, c in enumerate(CAC_CELL):
        if c["cell_type"] != "code":
            continue
        dem: dict[str, int] = {}
        for dong in "".join(c["source"]).split("\n"):
            if dong[:1].isspace() or "=" not in dong:
                continue
            s = dong.strip()
            if s.startswith("#"):
                continue
            ten = s.split("=")[0].strip()
            if ten.isupper() and ten.isidentifier():
                dem[ten] = dem.get(ten, 0) + 1
        trung = sorted(t for t, n in dem.items() if n > 1)
        if trung:
            raise SystemExit(f"cell {so_cell}: biến cấu hình bị gán nhiều lần: "
                             f"{trung}")

    so_code = sum(1 for c in CAC_CELL if c["cell_type"] == "code")
    print(f"Đã sinh {DUONG_DAN.relative_to(GOC)}")
    print(f"  {len(CAC_CELL)} cell ({so_code} cell mã, "
          f"{len(CAC_CELL) - so_code} cell chữ)")


if __name__ == "__main__":
    main()
