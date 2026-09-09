"""TASK 17 + 18 — Chạy TOÀN BỘ ablation trong một lượt, chịu được đứt phiên.

Gọi lại scripts/train.py và scripts/evaluate.py thay vì cài lại logic huấn luyện.
Nhờ vậy mọi thứ đã kiểm chứng ở TASK 15 (checkpoint, đồng bộ Hub, dừng sớm, báo
cáo tự sinh) dùng lại được nguyên vẹn, và ablation không có đường đi lệch khỏi
lượt chạy chính.

BA RÀNG BUỘC ĐỊNH HÌNH FILE NÀY, đều rút từ Sưu tập lỗi.md:

  1. Phiên Kaggle bị cắt ở 12 giờ (mục 1.5). Cả loạt 14 lượt không chắc chạy hết
     trong một phiên, nên PHẢI chạy tiếp được. Mỗi lượt xong là ghi kết quả và
     đẩy ngay, không đợi tới cuối. Chạy lại thì bỏ qua những lượt đã xong.

  2. /kaggle/working chỉ có 20 GB (mục 1.10). 14 lượt x 2 checkpoint x 549 MB =
     15,4 GB, chưa kể dữ liệu. Phải XOÁ checkpoint cục bộ ngay sau khi đã đẩy
     lên Hub và đã chấm điểm xong.

  3. Smoke test từng đè lên checkpoint thật (mục 1.8). Ở đây smoke đi vào nhánh
     tên khác hẳn, nên chạy smoke bao nhiêu lần cũng không đụng kết quả thật.

Chạy:
    python scripts/chay_ablation.py --so-buoc 3000 --repo-hub <user>/<repo>
    python scripts/chay_ablation.py --smoke        # nhanh, chỉ để bắt lỗi sớm
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))

for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# DANH SÁCH THÍ NGHIỆM — nguồn sự thật DUY NHẤT.
# Thêm bớt thí nghiệm chỉ sửa ở đây; bảng, hình và báo cáo tự bám theo.
# Mục 2.15 Sưu tập lỗi: hằng số viết ở nhiều nơi là nguồn của lỗi im lặng.
#
# Vì sao số seed khác nhau giữa các nhóm:
#   TASK 17 (kiến trúc) BẮT BUỘC tối thiểu 2 seed — không có độ lệch giữa seed
#   thì không biết chênh lệch đo được là thật hay là nhiễu.
#   TASK 18 (kỹ thuật huấn luyện) không đòi 2 seed, nên chạy 1 để dành GPU.
#   A0 được 2 seed vì đây là thí nghiệm chủ đạo thầy đề nghị.
# =============================================================================
THI_NGHIEM = [
    ("doi_chung", "configs/base.yaml",                        [42, 1337]),
    ("a0",        "configs/ablation_a0_vanilla.yaml",         [42, 1337]),
    ("a1",        "configs/ablation_a1_layernorm.yaml",       [42, 1337]),
    ("a4",        "configs/ablation_a4_sincos.yaml",          [42, 1337]),
    ("a5",        "configs/ablation_a5_relu.yaml",            [42, 1337]),
    ("a6",        "configs/ablation_a6_post_norm.yaml",       [42, 1337]),
    ("a2",        "configs/ablation_a2_warmup.yaml",          [42]),
    ("a3",        "configs/ablation_a3_label_smoothing.yaml", [42]),
]

THU_MUC_KET_QUA = GOC / "results" / "ablation"
DUONG_DAN_KET_QUA = THU_MUC_KET_QUA / "ket_qua.csv"

COT_KET_QUA = [
    "ma_thi_nghiem", "ten_chay", "seed", "so_buoc", "loss_dev",
    "bleu_dev", "chrf_dev", "bleu_test", "chrf_test",
    "chu_ky_bleu", "chu_ky_chrf", "so_tham_so", "giay_huan_luyen",
]


def _chay(lenh: list[str], mo_ta: str) -> bool:
    """Gọi một script con, cho log chảy thẳng ra ngoài để theo dõi trên Kaggle."""
    print(f"\n{'=' * 78}\n>>> {mo_ta}\n    {' '.join(lenh)}\n{'=' * 78}", flush=True)
    ket_thuc = subprocess.run(lenh, cwd=GOC)
    if ket_thuc.returncode != 0:
        print(f"[ablation] THẤT BẠI ({ket_thuc.returncode}): {mo_ta}", flush=True)
        return False
    return True


def da_lam_xong(ma: str, seed: int) -> bool:
    """Lượt này đã có kết quả chưa — đây chính là cơ chế chạy tiếp.

    Đọc từ ket_qua.csv, mà file đó được đẩy lên Hub sau mỗi lượt. Phiên mới kéo
    file về là biết ngay còn nợ những lượt nào.
    """
    if not DUONG_DAN_KET_QUA.exists():
        return False
    df = pd.read_csv(DUONG_DAN_KET_QUA)
    if df.empty or "ma_thi_nghiem" not in df.columns:
        return False
    return not df[(df["ma_thi_nghiem"] == ma) & (df["seed"] == seed)].empty


def doc_diem_moi_nhat(ten_chay: str, split: str) -> dict:
    """Lấy dòng điểm mà evaluate.py vừa nối vào results/diem_chinh.csv.

    evaluate.py chỉ biết NỐI THÊM, nên phải lọc theo tên lượt chạy rồi lấy dòng
    cuối. Lọc sai thì bảng ablation lấy nhầm điểm của thí nghiệm khác — vẫn ra
    số, vẫn trông hợp lý, và không ai phát hiện.
    """
    csv = GOC / "results" / "diem_chinh.csv"
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    khop = df[df["Checkpoint"].astype(str).str.contains(ten_chay, regex=False, na=False)]
    khop = khop[khop["Tập test"] == split]
    if khop.empty:
        return {}
    d = khop.iloc[-1]
    return {
        "bleu": float(d["BLEU"]), "chrf": float(d["chrF++"]),
        "chu_ky_bleu": str(d["Chữ ký BLEU"]), "chu_ky_chrf": str(d["Chữ ký chrF++"]),
    }


def ghi_ket_qua(hang: dict) -> None:
    """Nối một dòng vào ket_qua.csv, tạo file kèm đủ cột nếu chưa có."""
    THU_MUC_KET_QUA.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{c: hang.get(c) for c in COT_KET_QUA}])
    df.to_csv(DUONG_DAN_KET_QUA, mode="a", index=False,
              header=not DUONG_DAN_KET_QUA.exists(), encoding="utf-8")


def don_checkpoint(thu_muc: Path) -> int:
    """Xoá checkpoint cục bộ sau khi đã đẩy lên Hub. Trả về số MB thu hồi.

    Mục 1.10 Sưu tập lỗi: hết đĩa giữa lúc lưu model làm mất trắng cả lượt chạy.
    Checkpoint chỉ còn ích cho việc chạy tiếp, mà bản trên Hub đã lo việc đó.
    """
    if not thu_muc.is_dir():
        return 0
    mb = sum(f.stat().st_size for f in thu_muc.rglob("*") if f.is_file()) // (1024 ** 2)
    shutil.rmtree(thu_muc, ignore_errors=True)
    return mb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--so-buoc", type=int, default=3000,
                        help="NGÂN SÁCH BƯỚC dùng chung cho mọi thí nghiệm. "
                             "Phải bằng nhau thì so sánh mới công bằng.")
    parser.add_argument("--smoke", action="store_true",
                        help="chạy cực ngắn để bắt lỗi sớm. Ghi vào nhánh riêng, "
                             "không bao giờ đụng kết quả thật.")
    parser.add_argument("--repo-hub", default=None)
    parser.add_argument("--gio-toi-da", type=float, default=None,
                        help="tự dừng cả loạt khi sắp hết giờ phiên Kaggle")
    parser.add_argument("--chi-thi-nghiem", nargs="*", default=None,
                        help="chỉ chạy vài mã, ví dụ: --chi-thi-nghiem doi_chung a0")
    parser.add_argument("--bo-qua-danh-gia", action="store_true",
                        help="chỉ huấn luyện, không chấm BLEU (để chấm sau)")
    args = parser.parse_args()

    so_buoc = 60 if args.smoke else args.so_buoc
    bat_dau = time.perf_counter()

    ke_hoach = [(ma, cfg, seed)
                for ma, cfg, seeds in THI_NGHIEM
                for seed in seeds
                if args.chi_thi_nghiem is None or ma in args.chi_thi_nghiem]

    print(f"\n{'#' * 78}")
    print(f"# ABLATION — {len(ke_hoach)} lượt · {so_buoc:,} bước mỗi lượt"
          f"{' · SMOKE TEST' if args.smoke else ''}")
    print("# Ngân sách bước BẰNG NHAU giữa mọi thí nghiệm — điều kiện để so sánh")
    print(f"{'#' * 78}", flush=True)

    da_chay, bo_qua, hong = 0, 0, []
    for ma, duong_dan_cfg, seed in ke_hoach:
        # TÊN LƯỢT CHẠY PHẢI TÁCH HẲN KHỎI LƯỢT HUẤN LUYỆN CHÍNH.
        #
        # Bản đầu đặt đối chứng là "iwslt_base_v1_seed42" — trùng đúng tên lượt
        # huấn luyện chính đang nằm trên Hub. Cộng với --tiep-tuc thì hậu quả là:
        #   1. Đối chứng KÉO VỀ checkpoint 11.000 bước rồi chạy tiếp, thay vì
        #      train 3.000 bước mới. Nó thành 14.000 bước còn A1..A6 chỉ có
        #      3.000 — ngân sách lệch nhau nên ablation mất sạch ý nghĩa.
        #   2. Nó GHI ĐÈ checkpoints/iwslt_base_v1_seed42/ trên Hub, phá luôn
        #      model bước 6.000 đã giao.
        # Cả hai đều không ném lỗi nào. Đúng loại hỏng im lặng đắt nhất.
        #
        # Nhét NGÂN SÁCH BƯỚC vào tên là lớp chặn thứ hai: đổi ngân sách thì tên
        # đổi theo, nên kết quả của hai ngân sách khác nhau không thể lẫn vào
        # cùng một bảng.
        ten_goc = ("base" if ma == "doi_chung"
                   else Path(duong_dan_cfg).stem.replace("ablation_", ""))
        ten_chay = f"abl{so_buoc}_{ten_goc}_seed{seed}"
        if args.smoke:
            ten_chay = "smoke_" + ten_chay

        if da_lam_xong(ma, seed):
            print(f"[ablation] BỎ QUA {ma} seed {seed} — đã có kết quả.", flush=True)
            bo_qua += 1
            continue

        if args.gio_toi_da:
            da_dung = (time.perf_counter() - bat_dau) / 3600
            if da_dung >= args.gio_toi_da:
                print(f"\n[ablation] Đã dùng {da_dung:.2f} giờ, dừng để kịp lưu. "
                      "Chạy lại notebook là tiếp tục từ đây.", flush=True)
                break

        moc = time.perf_counter()
        # --ten-thi-nghiem là thứ THẬT SỰ đổi tên lượt chạy. train.py đặt tên từ
        # cfg.thi_nghiem.ten, nên thiếu cờ này thì đối chứng vẫn mang tên
        # iwslt_base_v1_seed42 và đè lên lượt huấn luyện chính.
        #
        # --danh-gia-moi: lượt ablation ngắn hơn lượt chính nhiều lần, mà mặc định
        # 1000 bước mới đánh giá một lần. Để nguyên thì lượt 3.000 bước chỉ đánh
        # giá 3 lần, còn smoke 60 bước không đánh giá lần nào — không sinh ra
        # tot_nhat.pt, nên không có loss_dev, không chấm được BLEU, và bảng kết
        # quả toàn NaN. Chia làm 6 mốc để còn thấy được đường loss.
        lenh = [sys.executable, "scripts/train.py", "--config", duong_dan_cfg,
                "--seed", str(seed), "--so-buoc", str(so_buoc), "--tiep-tuc",
                "--ten-thi-nghiem", f"abl{so_buoc}_{ten_goc}",
                "--danh-gia-moi", str(max(10, so_buoc // 6))]
        if args.smoke:
            lenh.append("--smoke")

        # SMOKE CHỈ ĐẨY CHECKPOINT Ở LƯỢT ĐẦU TIÊN.
        # Mỗi checkpoint là 549 MB. Đẩy cả 14 lượt là 7,7 GB, khiến smoke test
        # mất hơn 40 phút — mà huấn luyện chỉ tốn một phút mỗi lượt, phần còn
        # lại là thời gian upload. Nó cũng bơm 12,7 GB rác vĩnh viễn vào repo Hub.
        # Lượt đầu đã chứng minh xong cơ chế đẩy hoạt động; 13 lượt sau chỉ lặp
        # lại đúng lời chứng minh đó với giá 7 GB băng thông.
        # Bảng ket_qua.csv vẫn được đẩy sau MỖI lượt (vài trăm byte), nên cơ chế
        # chạy tiếp vẫn được kiểm đầy đủ.
        if args.repo_hub and (not args.smoke or da_chay == 0):
            lenh += ["--repo-hub", args.repo_hub]
        elif args.smoke:
            print("[ablation] Smoke: bỏ đẩy checkpoint 549 MB lên Hub "
                  "(lượt đầu đã kiểm xong cơ chế).", flush=True)
        if args.gio_toi_da:
            # Chừa lại phần giờ đã tiêu, để lượt này không ăn lẹm sang giờ lưu.
            con_lai = args.gio_toi_da - (time.perf_counter() - bat_dau) / 3600
            lenh += ["--gio-toi-da", f"{max(con_lai, 0.05):.3f}"]

        if not _chay(lenh, f"HUẤN LUYỆN {ma} · seed {seed} · {so_buoc:,} bước"):
            hong.append(f"{ma}/seed{seed} (huấn luyện)")
            continue
        giay_huan_luyen = time.perf_counter() - moc

        ck = GOC / "artifacts" / "checkpoints" / ten_chay / "tot_nhat.pt"
        hang = {"ma_thi_nghiem": ma, "ten_chay": ten_chay, "seed": seed,
                "so_buoc": so_buoc, "giay_huan_luyen": round(giay_huan_luyen, 1)}

        if ck.exists():
            from nmt.training.checkpoint import doc_thong_tin

            tt = doc_thong_tin(ck)
            hang["loss_dev"] = tt["loss_dev"]

            if not args.bo_qua_danh_gia:
                for split in ("dev", "test"):
                    lenh_cham = [sys.executable, "scripts/evaluate.py",
                                 "--config", duong_dan_cfg, "--seed", str(seed),
                                 "--checkpoint", str(ck), "--split", split]
                    if args.smoke:
                        lenh_cham.append("--cho-phep-smoke")
                    if _chay(lenh_cham, f"CHẤM ĐIỂM {ma} · seed {seed} · {split}"):
                        d = doc_diem_moi_nhat(ten_chay, split)
                        hang[f"bleu_{split}"] = d.get("bleu")
                        hang[f"chrf_{split}"] = d.get("chrf")
                        if split == "test":
                            hang["chu_ky_bleu"] = d.get("chu_ky_bleu")
                            hang["chu_ky_chrf"] = d.get("chu_ky_chrf")
        else:
            print(f"[ablation] Không thấy {ck} — lượt này không có checkpoint tốt nhất.",
                  flush=True)
            hong.append(f"{ma}/seed{seed} (thiếu checkpoint)")

        ghi_ket_qua(hang)
        da_chay += 1

        # Đẩy bảng kết quả lên Hub NGAY, không đợi tới cuối loạt. Phiên đứt giữa
        # chừng thì lần sau vẫn biết đã làm tới đâu (mục 1.5).
        if args.repo_hub:
            try:
                from nmt.training.checkpoint import CHE_DO_SMOKE, CHE_DO_THAT
                from nmt.training.hub_sync import day_len_hub

                # KHÔNG tự bịa tiền tố smoke ở đây. day_len_hub đã có sẵn
                # _tien_to_theo_che_do đẩy mọi thứ của smoke vào `smoke/...`, và
                # đó chính là bản sửa mục 1.8. Đặt thêm một lược đồ tên thứ hai
                # song song là cách chắc chắn nhất để đẻ ra mục 1.7 lần nữa:
                # dữ liệu vẫn nằm trên Hub, chỉ là cái TÊN không khớp.
                day_len_hub(DUONG_DAN_KET_QUA, args.repo_hub, "ablation/ket_qua.csv",
                            che_do=CHE_DO_SMOKE if args.smoke else CHE_DO_THAT,
                            ghi_chu=f"ablation: xong {ma} seed {seed}")
            except Exception as loi:
                print(f"[ablation] Đẩy bảng kết quả thất bại: "
                      f"{type(loi).__name__}: {loi}", flush=True)

        mb = don_checkpoint(GOC / "artifacts" / "checkpoints" / ten_chay)
        if mb:
            print(f"[ablation] Đã xoá checkpoint cục bộ của {ten_chay}, "
                  f"thu hồi {mb:,} MB.", flush=True)

    tong_gio = (time.perf_counter() - bat_dau) / 3600
    print(f"\n{'#' * 78}")
    print(f"# XONG · chạy {da_chay} lượt · bỏ qua {bo_qua} lượt đã có · {tong_gio:.2f} giờ")
    if hong:
        print(f"# LƯỢT HỎNG ({len(hong)}): {', '.join(hong)}")
    print(f"{'#' * 78}", flush=True)

    if DUONG_DAN_KET_QUA.exists():
        print(f"\nBảng kết quả: {DUONG_DAN_KET_QUA.relative_to(GOC)}")
        print(pd.read_csv(DUONG_DAN_KET_QUA).to_string(index=False))


if __name__ == "__main__":
    main()
