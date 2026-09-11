"""Tìm ĐỘ DÀI WARMUP nhỏ nhất mà A0 (Post-Norm) còn sống được.

VÌ SAO PHẢI QUÉT CHỨ KHÔNG ĐOÁN
-------------------------------
Ba lượt A0 đã hỏng, mỗi lượt 70 phút, đều vì chọn warmup bằng cách suy luận rồi
chạy thẳng lượt thật. Số đo từ ba lượt đó đã khoanh được vùng nhưng chưa chỉ ra
điểm:

    warmup 4.000  ->  SỐNG, nhưng lr chưa kịp lên đỉnh nên mới đạt loss 4,91 ở
                      bước 3.000 và BLEU chỉ 3,6 — quá thấp để so với đối chứng
    warmup   120  ->  CHẾT, rơi vào bẫy unigram trong 50 bước đầu

Đáp án nằm ở giữa, và nằm ở đâu thì KHÔNG suy ra được từ lý thuyết: nó phụ thuộc
độ sâu, fp16, ngưỡng cắt gradient và dữ liệu. Warmup càng nhỏ càng tốt, vì phần
ngân sách còn lại được học ở learning rate thật càng nhiều.

CÁI LÀM CHO VIỆC QUÉT ĐỦ RẺ
---------------------------
Cầu dao bẫy unigram trong src/nmt/training/chan_doan.py. Một cấu hình hỏng bị
ngắt ngay khi đo được là mô hình bỏ qua câu nguồn, thay vì chạy hết 3.000 bước.
Không có cầu dao thì quét 3 giá trị mất 3,5 giờ GPU; có cầu dao thì phần lớn
lượt hỏng chết trong khoảng 15 phút.

MÔ HÌNH KHÔNG HỀ BỊ THU NHỎ. Quét chạy đúng kiến trúc thật (d_model 512, 6+6
lớp), đúng dữ liệu thật, chỉ CẮT NGẮN SỐ BƯỚC. Cái rẻ đi là thời gian, không
phải độ trung thực.

Chạy:
    python scripts/quet_warmup_a0.py --repo-hub <user>/<repo>
    python scripts/quet_warmup_a0.py --ung-vien 700 1000 --so-buoc 1200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))

CAU_HINH_A0 = "configs/ablation_a0_vanilla.yaml"

# Đã biết chắc là sống (lượt 10/09 chạy hết 3.000 bước không suy biến). Giữ làm
# lưới an toàn: mọi ứng viên đều chết thì vẫn còn một con số để báo cáo.
WARMUP_AN_TOAN = 4000


def _chay_mot_ung_vien(warmup: int, so_buoc: int, danh_gia_moi: int,
                       canh_tu: int, seed: int, repo_hub: str | None) -> dict:
    """Chạy thử một giá trị warmup. Trả về dict mô tả kết quả."""
    ten = f"quet_a0_warmup{warmup}"
    lenh = [
        sys.executable, "scripts/train.py",
        "--config", CAU_HINH_A0,
        "--seed", str(seed),
        "--so-buoc", str(so_buoc),
        "--so-buoc-warmup", str(warmup),
        "--danh-gia-moi", str(danh_gia_moi),
        "--canh-bay-tu-buoc", str(canh_tu),
        "--ten-thi-nghiem", ten,
        "--tu-dau",
    ]
    if repo_hub:
        lenh += ["--repo-hub", repo_hub]

    print(f"\n{'=' * 78}\n>>> THỬ warmup = {warmup:,}\n    {' '.join(lenh)}\n"
          f"{'=' * 78}", flush=True)
    ket_thuc = subprocess.run(lenh, cwd=GOC)

    duong_log = GOC / "results/logs" / f"{ten}_seed{seed}_tu_dau" / "metrics.csv"
    return {
        "warmup": warmup,
        "song": ket_thuc.returncode == 0,
        **_doc_ket_qua(duong_log),
    }


def _doc_ket_qua(duong_log: Path) -> dict:
    """Lấy loss_dev và chênh lệch phụ thuộc đầu vào ở lần đánh giá cuối."""
    trong = {"loss_dev_cuoi": None, "chenh_cuoi": None, "buoc_cuoi": None}
    if not duong_log.exists():
        return trong

    bang = pd.read_csv(duong_log)
    if "loss_dev" not in bang.columns:
        return trong
    hang_danh_gia = bang[bang["loss_dev"].notna()]
    if hang_danh_gia.empty:
        return trong

    cuoi = hang_danh_gia.iloc[-1]
    chenh = cuoi.get("chenh_phu_thuoc")
    return {
        "loss_dev_cuoi": float(cuoi["loss_dev"]),
        "chenh_cuoi": None if pd.isna(chenh) else float(chenh),
        "buoc_cuoi": int(cuoi["buoc"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ung-vien", type=int, nargs="+", default=[1000, 500, 2000],
                   help="các độ dài warmup cần thử, theo thứ tự thử. Mặc định "
                        "thử 1000 trước (điểm giữa của vùng đã khoanh), rồi 500 "
                        "nếu 1000 sống, rồi 2000 nếu 1000 chết.")
    p.add_argument("--so-buoc", type=int, default=1500,
                   help="ngân sách bước cho MỖI lượt thử. Đủ dài để một cấu hình "
                        "tốt kịp lộ ra là nó đang học.")
    p.add_argument("--danh-gia-moi", type=int, default=250)
    p.add_argument("--canh-bay-tu-buoc", type=int, default=1000,
                   help="từ bước này cầu dao mới có hiệu lực. Đừng hạ xuống 500: "
                        "lượt A0 ngày 10/09 sống nhưng ở bước 500 loss vẫn 7,06 "
                        "nên rất dễ bị ngắt oan; tới bước 1.000 nó đã xuống 6,37 "
                        "và lúc đó mới phân biệt được chậm với chết.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repo-hub", default=None)
    args = p.parse_args()

    print(f"Quét warmup cho A0 — mô hình THẬT (d_model 512, 6+6 lớp), "
          f"chỉ cắt ngắn còn {args.so_buoc:,} bước mỗi lượt.")
    print(f"Ứng viên: {args.ung_vien}   (lưới an toàn: {WARMUP_AN_TOAN:,})\n")

    ket_qua = [_chay_mot_ung_vien(w, args.so_buoc, args.danh_gia_moi,
                                  args.canh_bay_tu_buoc, args.seed, args.repo_hub)
               for w in args.ung_vien]

    print(f"\n{'=' * 78}\nKẾT QUẢ QUÉT\n{'=' * 78}")
    print(f"{'warmup':>8} {'sống':>6} {'bước':>7} {'loss_dev':>10} "
          f"{'phụ thuộc nguồn':>17}")
    for r in ket_qua:
        chenh = "—" if r["chenh_cuoi"] is None else f"{r['chenh_cuoi']:+.3f}"
        loss = "—" if r["loss_dev_cuoi"] is None else f"{r['loss_dev_cuoi']:.4f}"
        buoc = "—" if r["buoc_cuoi"] is None else f"{r['buoc_cuoi']:,}"
        print(f"{r['warmup']:>8,} {'CÓ' if r['song'] else 'KHÔNG':>6} "
              f"{buoc:>7} {loss:>10} {chenh:>17}")

    song = [r for r in ket_qua if r["song"]]
    print()
    if song:
        tot_nhat = min(song, key=lambda r: r["warmup"])
        print(f"KHUYẾN NGHỊ: đặt so_buoc_warmup = {tot_nhat['warmup']:,} trong "
              f"{CAU_HINH_A0}")
        print("Đây là giá trị NHỎ NHẤT còn sống trong số đã thử, tức để lại nhiều "
              "bước học ở learning rate thật nhất.")
        print("Muốn ép sát hơn nữa thì quét tiếp khoảng giữa nó và ứng viên chết "
              "gần nhất.")
    else:
        print(f"Không ứng viên nào sống. Dùng lưới an toàn "
              f"so_buoc_warmup = {WARMUP_AN_TOAN:,} — đã chứng minh là chạy được "
              f"ở lượt 10/09, đổi lại A0 sẽ chưa học hết ở learning rate đỉnh và "
              f"điều đó PHẢI ghi rõ trong báo cáo.")


if __name__ == "__main__":
    main()
