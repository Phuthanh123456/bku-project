"""TASK 16 — Chấm BLEU và chrF++ bằng sacrebleu trên dev và test.  Người làm: My.

Ghi lại NGUYÊN VĂN chuỗi chữ ký của sacrebleu vào results/diem_chinh.csv.
Ghi rõ hướng dịch là từ Anh sang Việt, tên tập test và số câu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Chạy được ngay cả khi chưa `pip install -e .` (tiện khi làm trên Kaggle).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Console Windows mặc định dùng bảng mã cp1252, in chữ tiếng Việt ra là vỡ chữ
# hoặc UnicodeEncodeError. Ép về UTF-8 để cả nhóm đọc được log giống nhau.
for _luong in (sys.stdout, sys.stderr):
    if hasattr(_luong, "reconfigure"):
        _luong.reconfigure(encoding="utf-8", errors="replace")

from nmt.utils import dat_seed, luu_config, nap_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=None, help="ghi đè thi_nghiem.seed")
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn tới checkpoint (.pt) đã huấn luyện")
    parser.add_argument("--split", type=str, choices=["dev", "test"], default="test", help="Tập đánh giá (dev hoặc test)")
    parser.add_argument("--cho-phep-ngau-nhien", action="store_true",
                        help="Cho phép chạy KHÔNG có checkpoint. Điểm ra sẽ vô nghĩa, "
                             "chỉ dùng để kiểm script không crash.")
    parser.add_argument("--cho-phep-smoke", action="store_true",
                        help="Cho phép chấm điểm trên checkpoint của smoke test. "
                             "Mặc định TỪ CHỐI, vì điểm đó không phải kết quả thật.")
    parser.add_argument("--cach", choices=["greedy", "beam"], default="greedy",
                        help="Cách sinh câu. Mặc định greedy — điểm baseline của "
                             "TASK 16. Đổi sang beam là phần so sánh của TASK 19.")
    parser.add_argument("--beam-size", type=int, default=None,
                        help="ghi đè sinh_cau.beam_size")
    parser.add_argument("--luu-vi-du", type=int, default=0,
                        help="ghi N câu dịch ra results/vi_du_dich.csv để dựng "
                             "bảng ví dụ trong báo cáo")
    parser.add_argument("--gioi-han-cau", type=int, default=0,
                        help="chỉ chấm N câu đầu. 0 = chấm đủ cả tập. "
                             "Dùng cho smoke test: chấm đủ 1.553 câu dev cộng "
                             "1.268 câu test cho MỖI lượt, nhân 14 lượt, thì "
                             "phần chấm điểm lâu hơn cả phần huấn luyện — smoke "
                             "test thành thứ chậm nhất quy trình, ngược hẳn mục "
                             "đích của nó. ĐIỂM RA KHÔNG PHẢI KẾT QUẢ THẬT, chỉ "
                             "để chứng minh đường chấm điểm chạy được.")
    args = parser.parse_args()

    cfg = nap_config(args.config)
    if args.seed is not None:
        cfg["thi_nghiem"]["seed"] = args.seed
    dat_seed(cfg.thi_nghiem.seed, cfg.thi_nghiem.deterministic)

    import torch
    import pandas as pd
    import os
    from datetime import datetime
    from nmt.data import nap_tokenizer, DuLieuSongNgu, tao_dataloader, PAD_ID, BOS_ID, EOS_ID
    from nmt.model.transformer import TransformerNMT
    from nmt.inference.search import beam_search, greedy_search
    from nmt.eval.metrics import cham_bleu, cham_chrf

    tokenizer = nap_tokenizer(cfg.du_lieu.tokenizer)
    
    split_path = cfg.du_lieu.dev if args.split == "dev" else cfg.du_lieu.test
    duong_dan_en = f"{split_path}.en"
    duong_dan_vi = f"{split_path}.vi"
    
    dataset = DuLieuSongNgu(duong_dan_en, duong_dan_vi, tokenizer)

    if args.gioi_han_cau > 0:
        dataset._src = dataset._src[: args.gioi_han_cau]
        dataset._tgt = dataset._tgt[: args.gioi_han_cau]
        print(f"CHỈ CHẤM {len(dataset)} CÂU ĐẦU — điểm dưới đây KHÔNG phải kết "
              f"quả thật, chỉ để kiểm đường chấm điểm chạy được.")

    # gom_theo_do_dai=True: xếp câu dài gần nhau nên mỗi batch bớt phần đệm.
    # Sinh câu là vòng lặp tự hồi quy chạy tới khi câu DÀI NHẤT trong batch xong,
    # nên một câu 100 token đứng chung với chín câu 10 token bắt cả batch chạy đủ
    # 100 bước. Xếp theo độ dài cắt được phần lãng phí đó.
    # KHÔNG ảnh hưởng tới điểm: câu dịch và câu tham chiếu lấy ra từ cùng một
    # batch nên vẫn khớp cặp; sacrebleu chấm theo cặp chứ không theo thứ tự.
    loader = tao_dataloader(dataset, so_token_moi_batch=cfg.du_lieu.so_token_moi_batch,
                            gom_theo_do_dai=True, tron=False)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = TransformerNMT(cfg).to(device)
    
    # ------------------------------------------------------------------ nạp checkpoint
    #
    # BA LỖI CỦA BẢN CŨ, cả ba đều thuộc loại "chạy được nhưng sai":
    #
    # 1. Nó tìm khóa "model_state_dict", nhưng luu_checkpoint của TASK 12 lưu dưới
    #    khóa "model". Không khớp thì nhánh dự phòng nạp NGUYÊN CẢ dict checkpoint
    #    làm state_dict, và ném RuntimeError vì thừa các khóa phien_ban, che_do,
    #    buoc, rng... Nói cách khác: TASK 16 chạy trên checkpoint thật là chết.
    #
    # 2. Gõ sai đường dẫn thì `os.path.exists` trả False, rơi xuống nhánh else,
    #    IN MỘT CẢNH BÁO RỒI CHẤM ĐIỂM BẰNG TRỌNG SỐ NGẪU NHIÊN — và vẫn nối thêm
    #    một dòng BLEU vào results/diem_chinh.csv. Ai liếc bảng điểm sẽ thấy một
    #    con số trông bình thường.
    #
    # 3. Không kiểm che_do, nên chấm nhầm checkpoint smoke test mà không hay biết.
    #    Đúng mục 1.9 của Sưu tập lỗi.md.
    thong_tin_ck = None
    if args.checkpoint:
        duong_dan_ck = Path(args.checkpoint)
        if not duong_dan_ck.exists():
            raise SystemExit(
                f"Không thấy checkpoint: {duong_dan_ck}\n"
                "Dừng ở đây thay vì lặng lẽ chấm điểm bằng trọng số ngẫu nhiên."
            )

        from nmt.training.checkpoint import CHE_DO_THAT, nap_checkpoint

        try:
            thong_tin_ck = nap_checkpoint(
                duong_dan_ck, model, map_location=device,
                che_do_mong_doi=None if args.cho_phep_smoke else CHE_DO_THAT,
                khoi_phuc_rng=False,      # chấm điểm thì không cần đụng RNG
            )
            print(f"Đã nạp checkpoint {duong_dan_ck} — bước {thong_tin_ck['buoc']:,}, "
                  f"epoch {thong_tin_ck['epoch']}, chế độ {thong_tin_ck['che_do']}")
        except RuntimeError as loi:
            # Checkpoint đời cũ hoặc do người khác sinh ra, không theo định dạng
            # của TASK 12. Thử nạp thẳng, nhưng nói rõ là đang đi đường dự phòng.
            print(f"[eval] Không đọc được theo định dạng TASK 12 ({loi})")
            print("[eval] Thử nạp như state_dict thuần...")
            goi = torch.load(duong_dan_ck, map_location=device, weights_only=False)
            model.load_state_dict(goi.get("model_state_dict", goi))
            print(f"Đã nạp checkpoint từ {duong_dan_ck} (đường dự phòng)")

    elif args.cho_phep_ngau_nhien:
        print("CẢNH BÁO: chạy bằng TRỌNG SỐ NGẪU NHIÊN. Điểm dưới đây vô nghĩa, "
              "chỉ dùng để kiểm script không crash.")
    else:
        raise SystemExit(
            "Chưa truyền --checkpoint.\n"
            "Chấm điểm không có checkpoint thì BLEU chỉ là nhiễu, mà nó vẫn được "
            "ghi vào results/diem_chinh.csv như một kết quả thật.\n"
            "  Chấm thật    : --checkpoint artifacts/checkpoints/<ten_chay>/tot_nhat.pt\n"
            "  Chỉ thử script: thêm cờ --cho-phep-ngau-nhien"
        )

    model.eval()
    
    du_doan = []
    tham_chieu = []
    cau_nguon = []          # để dựng bảng ví dụ trong báo cáo

    beam_size = args.beam_size or cfg.sinh_cau.beam_size
    mo_ta_cach = (f"beam={beam_size}, phạt độ dài {cfg.sinh_cau.he_so_phat_do_dai}"
                  if args.cach == "beam" else "greedy")
    print(f"Bắt đầu dịch tập {args.split} ({len(dataset)} câu) bằng {mo_ta_cach}...")

    import time

    bat_dau = time.perf_counter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            src_ids = batch["src_ids"].to(device)
            src_mask = batch["src_mask"].to(device)
            labels = batch["labels"].to(device)

            if args.cach == "beam":
                preds_ids = beam_search(
                    model, src_ids, src_mask, BOS_ID, EOS_ID,
                    beam_size=beam_size,
                    he_so_phat_do_dai=cfg.sinh_cau.he_so_phat_do_dai,
                    do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich,
                    dung_kv_cache=True,
                )
            else:
                preds_ids = greedy_search(
                    model, src_ids, src_mask, BOS_ID, EOS_ID,
                    do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich,
                    dung_kv_cache=cfg.sinh_cau.dung_kv_cache,
                )

            for p_ids, l_ids, s_ids in zip(preds_ids, labels, src_ids):
                p_list = p_ids.tolist()
                if BOS_ID in p_list: p_list.remove(BOS_ID)
                if EOS_ID in p_list: p_list = p_list[:p_list.index(EOS_ID)]
                du_doan.append(tokenizer.decode(p_list, skip_special_tokens=True))

                l_list = l_ids.tolist()
                if EOS_ID in l_list: l_list = l_list[:l_list.index(EOS_ID)]
                tham_chieu.append(tokenizer.decode(l_list, skip_special_tokens=True))

                cau_nguon.append(
                    tokenizer.decode([t for t in s_ids.tolist() if t != PAD_ID],
                                     skip_special_tokens=True))

            if (i + 1) % 10 == 0:
                print(f" Đã dịch xong batch {i + 1}/{len(loader)}")

    giay_dich = time.perf_counter() - bat_dau

    bleu, bleu_sig = cham_bleu(du_doan, tham_chieu)
    chrf, chrf_sig = cham_chrf(du_doan, tham_chieu)

    print(f"\nKẾT QUẢ TRÊN TẬP {args.split.upper()}  ({mo_ta_cach}):")
    print(f"BLEU:   {bleu:.2f}  (Chữ ký: {bleu_sig})")
    print(f"chrF++: {chrf:.2f}  (Chữ ký: {chrf_sig})")
    print(f"Thời gian dịch: {giay_dich:.1f} giây "
          f"({len(du_doan) / giay_dich:.1f} câu/giây)")

    # ------------------------------------------------------- bảng câu ví dụ
    # docs/bao_cao_danh_gia.md đòi bảng mười câu, và đòi CHỌN CẢ CÂU DỊCH TỐT
    # LẪN CÂU DỊCH TỆ. Lấy mười câu đầu là chọn mẫu thiên vị: tập test thường xếp
    # theo tài liệu nguồn nên mười câu đầu cùng một chủ đề và cùng độ khó.
    # Ở đây xếp theo độ trùng khớp rồi lấy cả hai đầu, để bảng phản ánh thật.
    os.makedirs("results", exist_ok=True)

    if args.luu_vi_du > 0:
        import sacrebleu

        diem_tung_cau = [
            sacrebleu.sentence_bleu(d, [t]).score for d, t in zip(du_doan, tham_chieu)
        ]
        thu_tu = sorted(range(len(du_doan)), key=lambda i: diem_tung_cau[i])
        mot_nua = max(1, args.luu_vi_du // 2)
        chon = thu_tu[:mot_nua] + thu_tu[-mot_nua:]      # tệ nhất + tốt nhất

        pd.DataFrame([{
            "stt": i,
            "nhom": "dịch tệ" if vi_tri < mot_nua else "dịch tốt",
            "cau_anh": cau_nguon[i],
            "ban_dich_chuan": tham_chieu[i],
            "du_doan": du_doan[i],
            "bleu_cau": round(diem_tung_cau[i], 2),
            "cach": args.cach,
            "tap": args.split,
        } for vi_tri, i in enumerate(chon)]).to_csv(
            Path("results") / f"vi_du_dich_{args.split}_{args.cach}.csv",
            index=False, encoding="utf-8")
        print(f"Đã ghi {len(chon)} câu ví dụ (nửa tệ nhất, nửa tốt nhất) vào "
              f"results/vi_du_dich_{args.split}_{args.cach}.csv")
    
    os.makedirs("results", exist_ok=True)
    csv_path = "results/diem_chinh.csv"
    
    row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Tập test": args.split,
        "Hướng dịch": "En-Vi",
        "Số câu": len(dataset),
        # Thiếu cột này thì dòng của Greedy và dòng của Beam nhìn y hệt nhau, và
        # TASK 19 không chứng minh được beam hơn greedy bao nhiêu điểm.
        "Cách sinh": mo_ta_cach,
        "Giây dịch": round(giay_dich, 1),
        "BLEU": bleu,
        "chrF++": chrf,
        "Chữ ký BLEU": bleu_sig,
        "Chữ ký chrF++": chrf_sig,
        "Checkpoint": args.checkpoint or "TRỌNG SỐ NGẪU NHIÊN — không phải kết quả thật",
        # Ghi kèm số bước và chế độ, để bảng điểm nói được nó chấm bản nào. Thiếu
        # hai cột này thì 12 lượt ablation cho ra 12 dòng nhìn y hệt nhau.
        "Bước": thong_tin_ck["buoc"] if thong_tin_ck else "",
        "Chế độ": thong_tin_ck["che_do"] if thong_tin_ck else "",
    }])
    
    # Nối thêm (append) nếu file đã tồn tại
    if os.path.exists(csv_path):
        row.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        row.to_csv(csv_path, index=False)
        
    print(f"Đã lưu kết quả vào {csv_path}")


if __name__ == "__main__":
    main()
