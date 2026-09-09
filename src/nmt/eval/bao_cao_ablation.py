"""TASK 17 + 18 — Dựng bảng, hình và diễn giải cho toàn bộ ablation.

Sinh ra từ notebook sau khi chạy xong, để số trong báo cáo luôn khớp số đo được.

QUY TẮC VẼ HÌNH — phong cách nghiên cứu khoa học:
  1. TRÊN HÌNH KHÔNG ĐƯỢC CÓ INSIGHT. Hình chỉ mang dữ liệu: trục, đơn vị, chú
     giải, thanh sai số. Mọi câu kết luận nằm ở phần chữ bên dưới. Lý do không
     phải thẩm mỹ: người đọc phải nhìn được số liệu và tự rút ra kết luận, rồi
     mới đối chiếu với kết luận của tác giả. Nhét sẵn kết luận vào hình là dẫn
     dắt người đọc, và tới lúc số liệu đổi thì chữ trên hình thành sai.
  2. Chú thích hình ("Hình 1. ...") đặt DƯỚI hình, chỉ mô tả hình vẽ cái gì và
     đo thế nào — không nêu kết luận.
  3. Xuất cả PNG 300 dpi (dán vào Word) lẫn PDF vector (dán vào slide, phóng to
     không vỡ).
  4. Bảng màu an toàn cho người mù màu, và khác nhau cả về độ đậm nhạt để in
     đen trắng vẫn phân biệt được.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Bảng màu Okabe–Ito: phân biệt được với mọi dạng mù màu phổ biến, và độ sáng
# khác nhau đủ để in đen trắng vẫn đọc được.
MAU_DOI_CHUNG = "#0072B2"     # xanh dương — bản cải tiến (đối chứng)
MAU_BIEN_THE = "#D55E00"      # cam đỏ    — biến thể đem so
MAU_PHU = "#009E73"           # xanh lá   — đường thứ ba khi cần

# Tên người đọc hiểu được, thay cho mã A0..A6 trong bảng và trên hình.
TEN_THI_NGHIEM = {
    "doi_chung": "Bản cải tiến (đối chứng)",
    "a0": "Vanilla 2017 (cả gói)",
    "a1": "A1 · LayerNorm",
    "a2": "A2 · Warmup",
    "a3": "A3 · Label smoothing 0,1",
    "a4": "A4 · sin-cos",
    "a5": "A5 · ReLU",
    "a6": "A6 · Post-Norm",
}

# Yếu tố mà mỗi thí nghiệm đổi so với đối chứng, để bảng nói rõ đang so cái gì.
DOI_GI = {
    "a0": "LayerNorm + Post-Norm + sin-cos + ReLU + warmup + label smoothing",
    "a1": "RMSNorm → LayerNorm",
    "a2": "cố định → warmup 4000 bước",
    "a3": "label smoothing 0,0 → 0,1",
    "a4": "RoPE → sin-cos",
    "a5": "SwiGLU → ReLU",
    "a6": "Pre-Norm → Post-Norm",
}

NHOM_KIEN_TRUC = ["a1", "a4", "a5", "a6"]      # TASK 17
NHOM_HUAN_LUYEN = ["a2", "a3"]                 # TASK 18


def _dat_kieu_khoa_hoc() -> None:
    """Kiểu vẽ dùng chung: chữ đủ to để chiếu slide, viền gọn, lưới mờ."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")      # không cần màn hình, chạy được trên Kaggle
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "errorbar.capsize": 3,
    })


def _luu_hinh(fig, thu_muc: Path, ten: str) -> dict[str, Path]:
    """Lưu cả PNG (Word) lẫn PDF (slide). Trả về đường dẫn để báo cáo trỏ tới."""
    import matplotlib.pyplot as plt

    thu_muc.mkdir(parents=True, exist_ok=True)
    duong_dan = {}
    for duoi in ("png", "pdf"):
        p = thu_muc / f"{ten}.{duoi}"
        fig.savefig(p)
        duong_dan[duoi] = p
    plt.close(fig)
    return duong_dan


def gom_theo_seed(df: pd.DataFrame) -> pd.DataFrame:
    """Gộp các seed của cùng một thí nghiệm thành trung bình và độ lệch chuẩn.

    Cột độ lệch giữa seed là thứ TASK 17 bắt buộc phải có. Nó quan trọng hơn vẻ
    ngoài: nếu chênh lệch giữa hai cấu hình nhỏ hơn độ lệch giữa hai seed của
    CÙNG một cấu hình, thì chênh lệch đó là nhiễu chứ không phải phát hiện. Báo
    cáo thiếu cột này thì mọi kết luận đều không kiểm chứng được.
    """
    so_cot = ["loss_dev", "bleu_dev", "chrf_dev", "bleu_test", "chrf_test",
              "so_tham_so", "giay_huan_luyen"]
    co_that = [c for c in so_cot if c in df.columns]
    if not co_that:
        raise ValueError(
            "ket_qua.csv không có cột số nào để gộp. Cần ít nhất loss_dev. "
            f"Cột đang có: {list(df.columns)}"
        )

    gom = df.groupby("ma_thi_nghiem")[co_that].agg(["mean", "std", "count"])
    gom.columns = [f"{a}_{b}" for a, b in gom.columns]
    gom = gom.reset_index()
    gom["so_seed"] = gom[f"{co_that[0]}_count"]

    # std của một seed duy nhất là NaN — ghi 0 thì người đọc tưởng đã đo lặp lại
    # và kết quả trùng khít. Để trống thì trung thực hơn.
    for c in co_that:
        gom[f"{c}_std"] = gom[f"{c}_std"].where(gom[f"{c}_count"] > 1)
    return gom


def _so(gia_tri, chu_so: int = 2, mac_dinh: str = "—") -> str:
    """Định dạng số cho bảng markdown, dùng dấu phẩy thập phân kiểu Việt."""
    if gia_tri is None or pd.isna(gia_tri):
        return mac_dinh
    return f"{gia_tri:.{chu_so}f}".replace(".", ",")


def ve_so_sanh_cot(gom: pd.DataFrame, ma_list: list[str], cot: str,
                   nhan_truc_y: str, thu_muc: Path, ten_hinh: str):
    """Biểu đồ cột so đối chứng với từng biến thể, kèm thanh sai số giữa seed.

    Thanh sai số chính là phần nhiều biểu đồ ablation bỏ quên. Không có nó thì
    hai cột cao gần bằng nhau trông như một khác biệt thật, trong khi chúng có
    thể nằm trọn trong dao động giữa các seed.
    """
    import matplotlib.pyplot as plt

    _dat_kieu_khoa_hoc()
    co_that = set(gom["ma_thi_nghiem"])
    thu_tu = [m for m in (["doi_chung"] + ma_list) if m in co_that]
    hang = gom.set_index("ma_thi_nghiem").reindex(thu_tu).dropna(subset=[f"{cot}_mean"])
    if hang.empty:
        return None

    nhan = [TEN_THI_NGHIEM.get(m, m) for m in hang.index]
    gia_tri = hang[f"{cot}_mean"].to_numpy()
    sai_so = hang[f"{cot}_std"].fillna(0).to_numpy()
    mau = [MAU_DOI_CHUNG if m == "doi_chung" else MAU_BIEN_THE for m in hang.index]

    fig, ax = plt.subplots(figsize=(1.6 * len(nhan) + 2.2, 4.0))
    thanh = ax.bar(range(len(nhan)), gia_tri, yerr=sai_so, color=mau,
                   width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(len(nhan)))
    ax.set_xticklabels(nhan, rotation=18, ha="right")
    ax.set_ylabel(nhan_truc_y)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)

    # In số ngay trên cột: người xem slide ở xa không đọc được vạch trục. Đây là
    # DỮ LIỆU chứ không phải insight, nên đặt trên hình là hợp lệ.
    for t, v, e in zip(thanh, gia_tri, sai_so):
        ax.annotate(f"{v:.2f}".replace(".", ","),
                    (t.get_x() + t.get_width() / 2, v + e),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)

    duoi = float(min(gia_tri - sai_so))
    tren = float(max(gia_tri + sai_so))
    bien = (tren - duoi) * 0.18 or 0.1
    ax.set_ylim(max(0.0, duoi - bien), tren + bien * 1.6)
    return _luu_hinh(fig, thu_muc, ten_hinh)


def ve_duong_loss(cac_duong: dict[str, pd.DataFrame], thu_muc: Path, ten_hinh: str,
                  cot_x: str = "buoc", cot_y: str = "loss_dev"):
    """Nhiều đường loss vẽ chồng lên nhau — TASK 18 yêu cầu đúng hình này.

    Bảng số chỉ cho thấy điểm cuối. Hai cấu hình có thể kết thúc ở cùng một chỗ
    mà đi tới đó theo hai đường hoàn toàn khác nhau, và với warmup thì chính
    HÌNH DẠNG đoạn đầu mới là thứ cần nhìn.
    """
    import matplotlib.pyplot as plt

    _dat_kieu_khoa_hoc()
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bang_mau = [MAU_DOI_CHUNG, MAU_BIEN_THE, MAU_PHU]
    kieu_net = ["-", "--", "-."]

    co_du_lieu = False
    for i, (nhan, df) in enumerate(cac_duong.items()):
        if df is None or df.empty or cot_y not in df.columns:
            continue
        d = df.dropna(subset=[cot_y])
        if d.empty:
            continue
        ax.plot(d[cot_x], d[cot_y], label=nhan, linewidth=1.9,
                color=bang_mau[i % len(bang_mau)],
                linestyle=kieu_net[i % len(kieu_net)])
        co_du_lieu = True

    if not co_du_lieu:
        plt.close(fig)
        return None

    ax.set_xlabel("Số bước huấn luyện")
    ax.set_ylabel("Loss trên tập dev")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    return _luu_hinh(fig, thu_muc, ten_hinh)


def dien_giai(gom: pd.DataFrame, ma: str, cot: str = "loss_dev",
              cang_nho_cang_tot: bool = True) -> str:
    """Viết đoạn diễn giải ngắn từ chính con số, đặt dưới bảng và hình.

    Chỗ này quyết định báo cáo trung thực hay không. Quy tắc: chênh lệch giữa
    hai cấu hình chỉ được gọi là khác biệt khi nó LỚN HƠN độ lệch giữa các seed
    của cùng một cấu hình. Không thì phải nói thẳng là chưa phân biệt được —
    đó là một kết quả hợp lệ, không phải thất bại.
    """
    b = gom.set_index("ma_thi_nghiem")
    if ma not in b.index or "doi_chung" not in b.index:
        return "_Chưa có đủ số liệu cho thí nghiệm này._"

    v_bien = b.loc[ma, f"{cot}_mean"]
    v_goc = b.loc["doi_chung", f"{cot}_mean"]
    if pd.isna(v_bien) or pd.isna(v_goc):
        return "_Chưa có đủ số liệu cho thí nghiệm này._"

    def _std(khoa) -> float:
        gt = b.loc[khoa, f"{cot}_std"]
        return 0.0 if pd.isna(gt) else float(gt)

    chenh = float(v_bien) - float(v_goc)
    lech_seed = max(_std(ma), _std("doi_chung"))
    ten = TEN_THI_NGHIEM.get(ma, ma)
    bien_the_tot_hon = (chenh < 0) if cang_nho_cang_tot else (chenh > 0)

    cau = [
        f"Đổi sang **{ten}** làm {cot.replace('_', ' ')} đi từ {_so(v_goc, 4)} "
        f"thành {_so(v_bien, 4)}, tức chênh {_so(abs(chenh), 4)}."
    ]
    if lech_seed == 0:
        cau.append(
            "Thí nghiệm này mới chạy một seed nên chưa biết dao động giữa các seed "
            "là bao nhiêu; chưa kết luận được chênh lệch trên là thật hay là nhiễu."
        )
    elif abs(chenh) <= lech_seed:
        cau.append(
            f"Chênh lệch này NHỎ HƠN dao động giữa các seed ({_so(lech_seed, 4)}), "
            "nên hai cấu hình chưa phân biệt được bằng số liệu hiện có. Đây là kết "
            "quả hợp lệ chứ không phải thí nghiệm hỏng: nó cho biết yếu tố này "
            "không phải thứ quyết định ở quy mô dữ liệu và ngân sách bước đang dùng."
        )
    else:
        ai_hon = "biến thể tốt hơn" if bien_the_tot_hon else "bản đối chứng tốt hơn"
        cau.append(
            f"Chênh lệch lớn hơn dao động giữa các seed ({_so(lech_seed, 4)}) nên "
            f"đây là khác biệt thật: {ai_hon}."
        )
    return " ".join(cau)


def _bang_markdown(gom: pd.DataFrame, ma_list: list[str]) -> str:
    """Bảng đúng các cột TASK 17 đòi: loss dev, BLEU, chrF++, ĐỘ LỆCH GIỮA SEED."""
    b = gom.set_index("ma_thi_nghiem")
    dong = [
        "| Thí nghiệm | Đổi gì so với đối chứng | Số seed | loss dev | ± giữa seed | BLEU (test) | chrF++ (test) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ma in ["doi_chung"] + ma_list:
        if ma not in b.index:
            continue
        h = b.loc[ma]
        dong.append(
            f"| {TEN_THI_NGHIEM.get(ma, ma)} | {DOI_GI.get(ma, '— (đối chứng)')} "
            f"| {int(h['so_seed'])} | {_so(h.get('loss_dev_mean'), 4)} "
            f"| {_so(h.get('loss_dev_std'), 4)} | {_so(h.get('bleu_test_mean'))} "
            f"| {_so(h.get('chrf_test_mean'))} |"
        )
    return "\n".join(dong)


def sinh_bao_cao(goc: Path, so_buoc: int | None = None) -> Path:
    """Dựng docs/ablation.md từ số đo, kèm hình và diễn giải. Trả về đường dẫn.

    Điền đúng khuôn nhóm trưởng đặt sẵn trong docs/ablation.md: bảng kiến trúc
    (TASK 17) có cột độ lệch giữa seed, bảng kỹ thuật huấn luyện (TASK 18) kèm
    diễn giải, và biểu đồ hai đường loss chồng lên nhau.
    """
    goc = Path(goc)
    csv = goc / "results" / "ablation" / "ket_qua.csv"
    if not csv.exists():
        raise SystemExit(
            f"Chưa có {csv}. Chạy scripts/chay_ablation.py trước đã.\n"
            "Dừng ở đây thay vì sinh một báo cáo rỗng trông như đã có kết quả."
        )

    df = pd.read_csv(csv)
    gom = gom_theo_seed(df)
    thu_muc_hinh = goc / "results" / "ablation" / "hinh"

    # --- hình: loss dev và BLEU cho từng nhóm ---
    h_kt_loss = ve_so_sanh_cot(gom, NHOM_KIEN_TRUC, "loss_dev",
                               "Loss trên tập dev (càng thấp càng tốt)",
                               thu_muc_hinh, "task17_kien_truc_loss_dev")
    h_kt_bleu = ve_so_sanh_cot(gom, NHOM_KIEN_TRUC, "bleu_test",
                               "BLEU trên tst2013 (càng cao càng tốt)",
                               thu_muc_hinh, "task17_kien_truc_bleu")
    h_hl_loss = ve_so_sanh_cot(gom, NHOM_HUAN_LUYEN, "loss_dev",
                               "Loss trên tập dev (càng thấp càng tốt)",
                               thu_muc_hinh, "task18_huan_luyen_loss_dev")
    h_a0 = ve_so_sanh_cot(gom, ["a0"], "bleu_test",
                          "BLEU trên tst2013 (càng cao càng tốt)",
                          thu_muc_hinh, "a0_vanilla_vs_cai_tien")

    # --- hình: đường loss chồng nhau, TASK 18 yêu cầu đúng hình này ---
    def _duong(ten_chay: str):
        p = goc / "results" / "logs" / ten_chay / "metrics.csv"
        return pd.read_csv(p) if p.exists() else None

    ten_theo_ma = df.drop_duplicates("ma_thi_nghiem").set_index("ma_thi_nghiem")["ten_chay"]
    h_duong = ve_duong_loss(
        {TEN_THI_NGHIEM[m]: _duong(ten_theo_ma[m])
         for m in ("doi_chung", "a2", "a3") if m in ten_theo_ma.index},
        thu_muc_hinh, "task18_duong_loss",
    )

    def _hinh(kq, so: int, chu_thich: str) -> str:
        if kq is None:
            return "_(chưa có đủ số liệu để vẽ hình này)_\n"
        ten = kq["png"].name
        return (f"![Hình {so}](../results/ablation/hinh/{ten})\n\n"
                f"**Hình {so}.** {chu_thich} "
                f"Bản gốc: `results/ablation/hinh/{kq['png'].name}` (PNG 300 dpi) "
                f"và `{kq['pdf'].name}` (PDF vector, dán slide không vỡ).\n")

    ngan_sach = so_buoc or (int(df["so_buoc"].iloc[0]) if "so_buoc" in df else None)
    phan = [
        "# Ablation",
        "",
        "> **Phụ trách: TASK 17 + 18 — Phú.** File này do "
        "`nmt.eval.bao_cao_ablation.sinh_bao_cao()` TỰ SINH sau khi chạy xong "
        "notebook ablation. Đừng sửa tay: chạy lại là mọi số được cập nhật cùng lúc.",
        "",
        "## Cách đọc",
        "",
        # Dấu phân cách hàng nghìn kiểu Việt là DẤU CHẤM. Để nguyên f"{n:,}" thì
        # "3,000 bước" bị đọc thành ba phẩy không, ngay trong file mà cả phần còn
        # lại đang dùng dấu phẩy làm dấu thập phân.
        f"Mọi thí nghiệm chạy CÙNG một ngân sách {ngan_sach:,} bước".replace(",", ".")
        + ", cùng dữ liệu, "
        "cùng tokenizer. Chỉ khi ngân sách bằng nhau thì chênh lệch đo được mới quy "
        "được cho yếu tố bị đổi." if ngan_sach else "",
        "",
        "Cột **± giữa seed** là thước đo quan trọng nhất trong bảng. Chênh lệch giữa "
        "hai cấu hình chỉ được coi là thật khi nó LỚN HƠN dao động giữa hai seed của "
        "cùng một cấu hình. Nhỏ hơn thì đó là nhiễu, và nói thẳng ra như vậy là một "
        "kết quả hợp lệ chứ không phải thí nghiệm hỏng.",
        "",
        "---",
        "",
        "## A0 — Transformer 2017 nguyên bản vs phiên bản cải tiến",
        "",
        "Thí nghiệm này có theo đề nghị của thầy: *\"Vì đã thêm các cải tiến so với "
        "Transformer 2017, anh đề nghị làm ablation vanilla Transformer vs phiên bản "
        "cải tiến với cùng budget train. Như vậy project sẽ chuyển từ 'code lại "
        "Transformer' thành một thực nghiệm đo được đóng góp của các cải tiến.\"*",
        "",
        _bang_markdown(gom, ["a0"]),
        "",
        _hinh(h_a0, 1, "BLEU trên tst2013 của bản 2017 nguyên bản so với bản cải "
                       "tiến, cùng ngân sách bước. Thanh sai số là độ lệch chuẩn "
                       "giữa các seed."),
        "",
        dien_giai(gom, "a0", "bleu_test", cang_nho_cang_tot=False),
        "",
        GHI_CHU_LY_THUYET["a0"],
        "",
        "---",
        "",
        "## TASK 17 — Ablation nhóm kiến trúc",
        "",
        _bang_markdown(gom, NHOM_KIEN_TRUC),
        "",
        _hinh(h_kt_loss, 2, "Loss trên tập dev của bốn biến thể kiến trúc so với bản "
                            "đối chứng. Thanh sai số là độ lệch chuẩn giữa các seed."),
        "",
        _hinh(h_kt_bleu, 3, "BLEU trên tst2013 của cùng bốn biến thể. Cùng ngân sách "
                            "bước, cùng dữ liệu, mỗi biến thể chỉ đổi một yếu tố."),
        "",
        *[f"- {dien_giai(gom, m)}" for m in NHOM_KIEN_TRUC],
        "",
        "---",
        "",
        "## TASK 18 — Ablation nhóm kỹ thuật huấn luyện",
        "",
        _bang_markdown(gom, NHOM_HUAN_LUYEN),
        "",
        _hinh(h_hl_loss, 4, "Loss trên tập dev khi bật warmup (A2) và khi bật label "
                            "smoothing 0,1 (A3), so với bản đối chứng."),
        "",
        _hinh(h_duong, 5, "Đường loss trên tập dev theo số bước, vẽ chồng lên nhau. "
                          "Bảng số chỉ cho thấy điểm cuối; hình này cho thấy tốc độ "
                          "hội tụ, tức đoạn đầu của đường."),
        "",
        *[f"- {dien_giai(gom, m)}" for m in NHOM_HUAN_LUYEN],
        "",
        GHI_CHU_LY_THUYET["a2"],
        "",
        "---",
        "",
        "## Số liệu gốc",
        "",
        "| File | Nội dung |",
        "|---|---|",
        "| `results/ablation/ket_qua.csv` | một dòng cho mỗi lượt chạy, chưa gộp seed |",
        "| `results/ablation/hinh/*.png` | hình bản gốc 300 dpi |",
        "| `results/ablation/hinh/*.pdf` | bản vector để dán slide |",
        "| `results/logs/<tên lượt>/metrics.csv` | đường loss từng bước |",
        "",
    ]

    ra = goc / "docs" / "ablation.md"
    ra.parent.mkdir(parents=True, exist_ok=True)
    ra.write_text("\n".join(p for p in phan if p is not None), encoding="utf-8")
    return ra


GHI_CHU_LY_THUYET = {
    "a2": (
        "> **Đọc trước khi kết luận.** Pre-Norm vốn đã làm giảm nhu cầu warmup "
        "(*On Layer Normalization in the Transformer Architecture*, arXiv:2002.04745). "
        "Vì vậy A2 chênh lệch rất nhỏ là ĐIỀU LÝ THUYẾT ĐÃ DỰ ĐOÁN, không phải bug "
        "và cũng không phải ablation thất bại. Giải thích được một kết quả bằng lý "
        "thuyết đã công bố thì thuyết phục hơn nhiều so với chỉ đưa ra con số."
    ),
    "a0": (
        "> **Lưu ý về cách đọc A0.** Thí nghiệm này đổi CẢ GÓI cùng lúc nên nó trả "
        "lời câu hỏi *\"toàn bộ cải tiến có đáng không\"*, chứ không quy được công "
        "lao cho từng thành phần. Phần quy công lao là việc của A1, A4, A5, A6. "
        "Hai loại thí nghiệm bổ sung cho nhau và phải đọc cùng nhau."
    ),
}
