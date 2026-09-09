"""Sinh bảng và biểu đồ phân tích lỗi thủ công cho 30 câu tst2013 đầu tiên."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))


# Nhãn được đọc thủ công từ output checkpoint iwslt_base_v1_seed42, không phải
# heuristic tự động. Một câu có thể mang nhiều nhãn.
NHAN = {
    1: (["Sai nghĩa", "Sai tên riêng", "Từ chưa dịch"], "'best' thành 'người giỏi nhất'; tên bài hát còn từ Envy."),
    2: ([], "Đúng nghĩa; khác tham chiếu nhưng tự nhiên."),
    3: (["Thiếu ý"], "Thiếu quan hệ sở hữu 'kẻ thù của chúng tôi'."),
    4: ([], "Truyền đạt đủ nội dung chính."),
    5: (["Sai nghĩa"], "'public execution' bị dịch thành 'cuộc hành trình công cộng'."),
    6: (["Ngữ pháp/độ trôi chảy"], "'trải nghiệm sự đói kém' cứng và không tự nhiên."),
    7: (["Sai nghĩa"], "'coworker's sister' bị biến thành 'chị dâu của thợ săn'."),
    8: (["Sai đại từ"], "Ngữ cảnh lá thư cần chị/em, model dùng bạn/chúng ta."),
    9: (["Sai nghĩa", "Sai đại từ"], "Mất ý 'sẵn sàng chết' và chọn đại từ chưa đúng ngữ cảnh."),
    10: ([], "Khớp nghĩa."),
    11: (["Ngữ pháp/độ trôi chảy"], "Đúng ý nhưng cụm 'chịu đựng khổ sở' gượng."),
    12: (["Dịch sát chữ", "Ngữ pháp/độ trôi chảy"], "'erase from my memory' bị dịch máy móc thành 'xoá bỏ bộ nhớ'."),
    13: (["Sai nghĩa", "Sai con số", "Sai đại từ"], "Bịa '20 tuổi', sai 'emaciated/lifeless' và quan hệ đại từ."),
    14: (["Dịch sát chữ"], "'focused on' dịch sát thành 'rất tập trung'."),
    15: (["Dịch sát chữ"], "Nạn đói 'hit' bị dịch thành 'xâm nhập'."),
    16: (["Sai nghĩa", "Thiếu ý"], "'bugs' thành 'thiên chúa'; mất 'tree bark'."),
    17: (["Sai nghĩa", "Thiếu ý"], "'power outages' thành 'quyền lực'; 'sea of lights' thành 'biển lửa'."),
    18: (["Dịch sát chữ"], "'we didn't' thành 'chúng tôi không làm thế' thay vì 'không có điện'."),
    19: (["Dịch sát chữ"], "'neighbors' thành người hàng xóm thay vì các nước lân cận."),
    20: (["Sai tên riêng", "Dịch sát chữ"], "Giữ Amrok thay vì tên Việt Áp Lục; 'serves as' dịch cứng."),
    21: (["Thiếu ý"], "Mất 'secretly' và chủ thể 'North Koreans' bị dịch thành quốc gia."),
    22: ([], "Đúng nội dung chính."),
    23: ([], "Đúng và tự nhiên."),
    24: (["Ngữ pháp/độ trôi chảy"], "Mệnh đề 'nạn đói mà tôi...' liên kết chưa chuẩn."),
    25: (["Dịch sát chữ"], "'separated' dịch sát thành 'bị tách rời'."),
    26: (["Sai nghĩa"], "Đảo nghĩa: nguồn nói mất 14 năm mới đoàn tụ, output nói mất 14 năm để sống cùng nhau."),
    27: (["Sai đại từ", "Ngữ pháp/độ trôi chảy"], "Chủ thể 'I' biến thành 'nó'; câu không tự nhiên."),
    28: (["Dịch sát chữ", "Ngữ pháp/độ trôi chảy"], "Sai cấu trúc 'considered in China as' và dịch sát nhiều cụm."),
    29: (["Sai nghĩa"], "'repatriated' bị dịch thành 'bị phạt'."),
    30: (["Sai nghĩa", "Ngữ pháp/độ trôi chảy"], "'came true/interrogation' thành 'đến đúng/tra cứu'."),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/du_doan_test_greedy_30.csv")
    parser.add_argument("--output", default="results/phan_tich_loi_30_cau.csv")
    parser.add_argument("--chart", default="results/phan_tich_loi.png")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    import pandas as pd
    from nmt.eval.metrics import cham_bleu, cham_chrf

    data = pd.read_csv(args.input).head(30)
    if len(data) != 30:
        raise SystemExit(f"Cần đúng 30 câu, file chỉ có {len(data)} câu")

    data["Nhóm lỗi"] = ["; ".join(NHAN[i][0]) or "Không lỗi nghiêm trọng" for i in range(1, 31)]
    data["Nhận xét thủ công"] = [NHAN[i][1] for i in range(1, 31)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False, encoding="utf-8-sig")

    counts = Counter(cat for cats, _ in NHAN.values() for cat in cats)
    categories = [
        "Sai nghĩa", "Thiếu ý", "Lặp từ", "Sai tên riêng", "Sai con số",
        "Sai đại từ", "Dịch sát chữ", "Ngữ pháp/độ trôi chảy", "Từ chưa dịch",
    ]
    values = [counts.get(cat, 0) for cat in categories]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(categories, values, color="#2f6fed")
    ax.bar_label(bars)
    ax.set_ylabel("Số câu mắc lỗi (trên 30 câu)")
    ax.set_title("Phân loại lỗi định tính — iwslt_base_v1_seed42, tst2013")
    ax.tick_params(axis="x", rotation=28)
    fig.tight_layout()
    chart = Path(args.chart)
    chart.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart, dpi=180)
    plt.close(fig)

    hypotheses = data["Dự đoán (VI)"].fillna("").tolist()
    references = data["Tham chiếu (VI)"].fillna("").tolist()
    bleu, bleu_sig = cham_bleu(hypotheses, references, tokenize="none")
    chrf, chrf_sig = cham_chrf(hypotheses, references)
    severe_free = sum(not NHAN[i][0] for i in range(1, 31))

    examples = {}
    for category in categories:
        for i, (cats, note) in NHAN.items():
            if category in cats:
                examples[category] = f"Câu {i}: {note}"
                break
        else:
            examples[category] = "Không ghi nhận trong mẫu 30 câu."

    lines = [
        "# Phân tích lỗi định tính — TASK 21",
        "",
        "> Kết quả này dùng checkpoint thật `iwslt_base_v1_seed42/tot_nhat.pt` "
        "(bước 6.000), greedy + KV cache, trên 30 câu đầu của `tst2013`.",
        "",
        "## Kết quả định lượng của mẫu",
        "",
        f"- BLEU: **{bleu:.2f}** — `{bleu_sig}`",
        f"- chrF++: **{chrf:.2f}** — `{chrf_sig}`",
        f"- Câu không có lỗi nghiêm trọng theo đọc thủ công: **{severe_free}/30**.",
        "- Đây là số trên mẫu phân tích, không thay cho điểm full test 1.268 câu.",
        "",
        "## Bảng phân loại",
        "",
        "| Nhóm lỗi | Số câu | Tỉ lệ | Ví dụ |",
        "|---|---:|---:|---|",
    ]
    for category, value in zip(categories, values):
        lines.append(f"| {category} | {value} | {value / 30 * 100:.1f}% | {examples[category]} |")
    lines.extend([
        "",
        "![Biểu đồ phân loại lỗi](../results/phan_tich_loi.png)",
        "",
        "## Nhận xét",
        "",
        "Lỗi chính là chọn sai nghĩa của từ/cụm đa nghĩa và dịch sát cấu trúc tiếng Anh. "
        "Các ví dụ rõ nhất gồm `public execution`, `power outages`, `tree bark`, "
        "`repatriated` và `interrogation`. Con số nhìn chung được bảo toàn; lỗi số duy "
        "nhất trong mẫu là model tự sinh `20 tuổi` ở câu 13. Không ghi nhận lặp từ.",
        "",
        "## Giới hạn của phân tích",
        "",
        "1. Mẫu 30 câu liên tiếp thuộc cùng một bài TED nên không đại diện toàn bộ chủ đề tst2013.",
        "2. Nhãn do một người đọc thủ công, chưa đo độ đồng thuận giữa nhiều người gán nhãn.",
        "3. Một câu có thể mang nhiều nhãn, vì vậy tổng tỉ lệ các nhóm không cộng thành 100%.",
        "4. Tham chiếu đơn cũng có cách viết/tên riêng chưa thống nhất; khác tham chiếu không tự động là sai.",
        "5. BLEU/chrF++ trên 30 câu có phương sai cao, chỉ dùng mô tả mẫu chứ không kết luận mô hình.",
        "",
        f"Bảng chi tiết từng câu: `{output.as_posix()}`.",
    ])
    (GOC / "docs/phan_tich_loi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã lưu {output}, {chart} và docs/phan_tich_loi.md")


if __name__ == "__main__":
    main()
