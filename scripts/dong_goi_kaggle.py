"""Đóng gói HAI dataset để upload lên Kaggle. Chạy một lệnh, ra hai file zip.

    python scripts/dong_goi_kaggle.py

Ra hai file ở thư mục `kaggle_upload/` nằm CẠNH repo (không nằm trong repo, để
không lọt vào git):

    bku-project-code.zip   mã nguồn — BẮT BUỘC
    bku-project-data.zip   dữ liệu và tokenizer — nên có, đỡ 10 phút mỗi phiên

VÌ SAO TÁCH LÀM HAI DATASET
Mã nguồn đổi liên tục còn dữ liệu thì gần như không bao giờ đổi. Gộp chung thì
mỗi lần sửa một dòng code phải upload lại 30 MB dữ liệu. Tách ra thì chỉ upload
lại zip mã nguồn (~2 MB), còn dataset dữ liệu giữ nguyên.

BA THỨ CỐ Ý KHÔNG ĐƯA VÀO
  *.pt                 checkpoint thuộc về Hugging Face. Nhét vào dataset vừa
                       nặng vừa dễ lẫn bản smoke với bản thật (mục 1.8, 1.9).
  results/ablation/    chay_ablation.py đọc file này để biết lượt nào đã xong.
                       Mang một bản cũ lên Kaggle là nó BỎ QUA những lượt thật
                       ra chưa chạy — hỏng im lặng, không lỗi nào báo.
  .git/                lịch sử git vô dụng trên Kaggle, chỉ tổ nặng.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
THU_MUC_RA = GOC.parent / "kaggle_upload"

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

# --- dataset 1: mã nguồn ---
THU_MUC_MA = ["src", "configs", "scripts", "tests", "notebooks", "docs"]
FILE_GOC = ["requirements.txt", "pyproject.toml", "README.md", "pytest.ini",
            "setup.py", "setup.cfg"]

# --- dataset 2: dữ liệu ---
THU_MUC_DU_LIEU = ["artifacts/tokenizer", "data/processed"]

# Không bao giờ đưa vào, dù nằm ở đâu.
LOAI_TRU_DUOI = {".pt", ".pth", ".pyc", ".zip"}
LOAI_TRU_THU_MUC = {"__pycache__", ".git", ".pytest_cache", ".venv",
                    ".ipynb_checkpoints", "ablation"}


def _nen_duoc(p: Path) -> bool:
    if p.suffix in LOAI_TRU_DUOI:
        return False
    return not any(phan in LOAI_TRU_THU_MUC for phan in p.parts)


def dong_goi(ten_zip: str, cac_thu_muc: list[str], cac_file: list[str]) -> Path:
    """Nén vào zip, giữ nguyên đường dẫn tương đối so với gốc repo.

    Giữ nguyên cấu trúc là bắt buộc: notebook tìm repo bằng cách dò thư mục nào
    có `src/nmt`. Nén phẳng hoặc bọc thêm một lớp thư mục thì nó không nhận ra,
    và cậu sẽ thấy "KHÔNG TÌM THẤY repo" ở Cell 2.
    """
    THU_MUC_RA.mkdir(parents=True, exist_ok=True)
    dich = THU_MUC_RA / ten_zip
    so_file, tong_byte = 0, 0

    with zipfile.ZipFile(dich, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for ten in cac_thu_muc:
            thu_muc = GOC / ten
            if not thu_muc.is_dir():
                print(f"  BỎ QUA (không có): {ten}")
                continue
            for f in sorted(thu_muc.rglob("*")):
                if f.is_file() and _nen_duoc(f):
                    z.write(f, f.relative_to(GOC).as_posix())
                    so_file += 1
                    tong_byte += f.stat().st_size
        for ten in cac_file:
            f = GOC / ten
            if f.is_file():
                z.write(f, ten)
                so_file += 1
                tong_byte += f.stat().st_size

    mb_goc = tong_byte / 1024 ** 2
    mb_zip = dich.stat().st_size / 1024 ** 2
    print(f"  {ten_zip:26s} {so_file:4d} file · {mb_goc:6.1f} MB → {mb_zip:5.1f} MB")
    return dich


def main() -> None:
    print(f"Đóng gói từ: {GOC}")
    print(f"Ghi ra     : {THU_MUC_RA}\n")

    ma = dong_goi("bku-project-code.zip", THU_MUC_MA, FILE_GOC)
    du_lieu = dong_goi("bku-project-data.zip", THU_MUC_DU_LIEU, [])

    # Kiểm hậu điều kiện thay vì tin là đã nén đúng. Thiếu một trong mấy file này
    # thì notebook chết ở Cell 2 hoặc Cell 4, sau khi đã khởi động phiên GPU.
    with zipfile.ZipFile(ma) as z:
        ten_file = z.namelist()
    for bat_buoc in ("src/nmt/model/transformer.py", "configs/base.yaml",
                     "configs/ablation_a0_vanilla.yaml",
                     "scripts/chay_ablation.py",
                     "notebooks/03_kaggle_ablation.ipynb"):
        if bat_buoc not in ten_file:
            raise SystemExit(f"THIẾU trong zip mã nguồn: {bat_buoc}")
    if any(f.endswith(".pt") for f in ten_file):
        raise SystemExit("Có file .pt lọt vào zip mã nguồn — checkpoint thuộc về Hub.")

    with zipfile.ZipFile(du_lieu) as z:
        ten_du_lieu = z.namelist()
    for bat_buoc in ("artifacts/tokenizer/tokenizer.json",
                     "data/processed/train.en", "data/processed/tst2013.vi"):
        if bat_buoc not in ten_du_lieu:
            raise SystemExit(f"THIẾU trong zip dữ liệu: {bat_buoc}")

    print(f"\n{'=' * 74}")
    print("HAI FILE ĐÃ SẴN SÀNG. Lên kaggle.com/datasets, bấm New Dataset:")
    print("=" * 74)
    print(f"""
  1. Kéo thả  {ma.name}
     Đặt tên  bku-project-code          <- BẮT BUỘC, chứa mã nguồn

  2. Kéo thả  {du_lieu.name}
     Đặt tên  bku-project-data          <- dữ liệu, đỡ 10 phút mỗi phiên

  Cả hai để Private cũng được, miễn cùng tài khoản với notebook.

  Rồi mở notebook: Add Input > Datasets > gắn CẢ HAI.

  Sửa code xong thì chỉ cần đóng gói lại và cập nhật dataset thứ nhất:
  vào trang dataset > New Version > kéo file zip mới vào.
""")
    print(f"Thư mục chứa: {THU_MUC_RA}")


if __name__ == "__main__":
    main()
