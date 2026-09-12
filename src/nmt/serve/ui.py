"""Giao diện Streamlit cho ứng dụng dịch Anh → Việt."""

from __future__ import annotations

import csv
import gc
import html
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
import torch


GOC_DU_AN = Path(__file__).resolve().parents[3]
THU_MUC_SRC = GOC_DU_AN / "src"
if str(THU_MUC_SRC) not in sys.path:
    sys.path.insert(0, str(THU_MUC_SRC))


@dataclass(frozen=True)
class ThongTinBanDich:
    van_ban: str
    so_token_nguon: int
    so_token_dich: int
    giay: float


class BoDich:
    """Nạp mô hình một lần và phục vụ các lượt dịch tiếp theo."""

    def __init__(self, checkpoint: str, tokenizer: str, config: str) -> None:
        from nmt.data import BOS_ID, EOS_ID, PAD_ID, nap_tokenizer
        from nmt.model.transformer import TransformerNMT
        from nmt.utils import nap_config

        checkpoint_path = Path(checkpoint)
        tokenizer_path = Path(tokenizer)
        config_path = Path(config)
        for ten, duong_dan in (
            ("checkpoint", checkpoint_path),
            ("tokenizer", tokenizer_path),
            ("config", config_path),
        ):
            if not duong_dan.is_file():
                raise FileNotFoundError(f"Không tìm thấy {ten}: {duong_dan}")

        self.thiet_bi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = nap_config(config_path)
        self.tokenizer = nap_tokenizer(tokenizer_path)
        self.pad_id, self.bos_id, self.eos_id = PAD_ID, BOS_ID, EOS_ID

        # Checkpoint huấn luyện còn chứa optimizer/scaler nên lớn hơn riêng model
        # nhiều lần. mmap chỉ đọc các trang thật sự dùng; assign=True gắn trực tiếp
        # tensor model thay vì tạo thêm một bản sao, giữ peak RAM đủ thấp cho cloud.
        goi = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        if not isinstance(goi, dict) or "model" not in goi:
            raise ValueError("Checkpoint không đúng định dạng: thiếu khóa 'model'.")

        self.model = TransformerNMT(self.cfg)
        self.model.load_state_dict(goi["model"], strict=True, assign=True)
        self.model.to(self.thiet_bi)
        self.model.eval()
        del goi
        gc.collect()

    @torch.inference_mode()
    def dich(
        self,
        cac_cau: list[str],
        beam_size: int = 4,
        do_dai_toi_da: int = 96,
    ) -> list[ThongTinBanDich]:
        from nmt.inference.search import beam_search, greedy_search

        if not cac_cau:
            return []

        gioi_han_nguon = int(self.cfg.du_lieu.do_dai_toi_da)
        cac_id = [self.tokenizer.encode(cau).ids[:gioi_han_nguon] for cau in cac_cau]
        if any(not ids for ids in cac_id):
            raise ValueError("Câu đầu vào không được để trống.")

        dai_nhat = max(map(len, cac_id))
        src = torch.full(
            (len(cac_id), dai_nhat), self.pad_id, dtype=torch.long, device=self.thiet_bi
        )
        for i, ids in enumerate(cac_id):
            src[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.thiet_bi)
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)

        tham_so_chung = dict(
            model=self.model,
            src_ids=src,
            src_mask=src_mask,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            do_dai_toi_da=do_dai_toi_da,
        )
        bat_dau = time.perf_counter()
        if beam_size > 1:
            dau_ra = beam_search(
                **tham_so_chung,
                beam_size=beam_size,
                he_so_phat_do_dai=float(self.cfg.sinh_cau.he_so_phat_do_dai),
                dung_kv_cache=True,
            )
        else:
            dau_ra = greedy_search(**tham_so_chung, dung_kv_cache=True)
        tong_giay = time.perf_counter() - bat_dau

        ket_qua: list[ThongTinBanDich] = []
        for hang, ids_nguon in zip(dau_ra, cac_id):
            ids_dich = [
                int(token)
                for token in hang
                if int(token) not in (self.pad_id, self.bos_id, self.eos_id)
            ]
            ket_qua.append(
                ThongTinBanDich(
                    van_ban=self.tokenizer.decode(ids_dich).strip(),
                    so_token_nguon=len(ids_nguon),
                    so_token_dich=len(ids_dich),
                    giay=tong_giay / len(cac_cau),
                )
            )
        return ket_qua


@st.cache_resource(show_spinner=False)
def nap_bo_dich(checkpoint: str, tokenizer: str, config: str) -> BoDich:
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    return BoDich(checkpoint, tokenizer, config)


def _duong_dan(ten_bien: str, mac_dinh: str) -> str:
    gia_tri = os.environ.get(ten_bien)
    return gia_tri if gia_tri else str(GOC_DU_AN / mac_dinh)


CHECKPOINT = _duong_dan("DUONG_DAN_CHECKPOINT", "artifacts/checkpoints/tot_nhat.pt")
TOKENIZER = _duong_dan("DUONG_DAN_TOKENIZER", "artifacts/tokenizer/tokenizer.json")
CONFIG = _duong_dan("DUONG_DAN_CONFIG", "configs/abl3000_base_seed42.yaml")


CSS = """
<style>
:root {
  --navy:#132238;
  --ink:#172b3f;
  --muted:#6b7c8f;
  --blue:#3975e9;
  --teal:#19a7a0;
  --cream:#f7f8f5;
  --line:#e2e8ee;
}
html,body,[class*="css"] { font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 8% 3%,rgba(57,117,233,.10),transparent 26rem),
    radial-gradient(circle at 92% 18%,rgba(25,167,160,.10),transparent 24rem),
    var(--cream);
}
[data-testid="stHeader"] { display:none; }
.block-container { max-width:1120px; padding-top:1.45rem; padding-bottom:2rem; }
#MainMenu,footer { visibility:hidden; }
.topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:3.4rem; }
.brand { display:flex; align-items:center; gap:.75rem; font-weight:800; color:var(--navy); font-size:1.12rem; }
.brand-mark { display:grid; place-items:center; width:38px; height:38px; border-radius:12px; color:white; background:linear-gradient(135deg,var(--blue),var(--teal)); box-shadow:0 8px 20px rgba(57,117,233,.25); }
.language-pill { padding:.52rem .85rem; border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.72); color:#526579; font-size:.82rem; font-weight:650; }
.hero { text-align:center; margin:0 auto 2.1rem; max-width:760px; }
.hero-kicker { color:var(--teal); text-transform:uppercase; font-size:.72rem; letter-spacing:.16em; font-weight:800; }
.hero h1 { color:var(--navy); font-size:clamp(2.25rem,5vw,4.15rem); letter-spacing:-.055em; line-height:1.04; margin:.55rem 0 .9rem; }
.hero h1 span { background:linear-gradient(90deg,var(--blue),var(--teal)); -webkit-background-clip:text; color:transparent; }
.hero p { color:var(--muted); font-size:1.02rem; line-height:1.7; margin:auto; max-width:620px; }
.st-key-translation_card { background:rgba(255,255,255,.88); border-radius:28px; box-shadow:0 24px 70px rgba(34,58,83,.11); }
.language-head { display:flex; justify-content:space-between; align-items:center; padding:.35rem .15rem .65rem; color:var(--navy); font-weight:750; }
.language-head span:last-child { color:#93a1af; font-size:.78rem; font-weight:500; }
.output-card { min-height:235px; padding:1.15rem 1.2rem; border-radius:18px; background:linear-gradient(145deg,#f0f5fb,#edf7f5); border:1px solid #dce8ed; color:var(--ink); font-size:1.12rem; line-height:1.7; white-space:pre-wrap; }
.output-placeholder { color:#98a6b4; }
.ready { display:inline-flex; align-items:center; gap:.45rem; color:#34756d; font-size:.78rem; font-weight:700; }
.ready:before { content:""; width:7px; height:7px; border-radius:50%; background:#2bb9a9; box-shadow:0 0 0 4px rgba(43,185,169,.12); }
.section-title { color:var(--navy); font-size:1.35rem; font-weight:800; margin-top:1rem; }
.hint { color:var(--muted); font-size:.88rem; }
.history-card { background:white; border:1px solid var(--line); border-radius:15px; padding:.85rem 1rem; margin:.55rem 0; }
.history-card .source { color:#718195; font-size:.82rem; margin-bottom:.25rem; }
.history-card .target { color:var(--ink); font-weight:650; }
.stButton>button,.stDownloadButton>button { border-radius:13px; min-height:2.85rem; font-weight:700; border:1px solid #d4dde5; }
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"] { border:0; color:white; background:linear-gradient(100deg,var(--blue),var(--teal)); box-shadow:0 10px 24px rgba(57,117,233,.20); }
[data-testid="stTextArea"] textarea { min-height:235px; border-radius:18px; border-color:#dce3e9; background:#fbfcfd; color:var(--ink); font-size:1.12rem; line-height:1.7; padding:1.15rem 1.2rem; }
[data-testid="stTextArea"] textarea:focus { border-color:#6d9cf2; box-shadow:0 0 0 3px rgba(57,117,233,.10); }
[data-testid="stFileUploaderDropzone"] { border-radius:18px; border-color:#d4dfe7; background:rgba(255,255,255,.72); }
[data-baseweb="tab-list"] { gap:1.25rem; }
[data-baseweb="tab"] { height:3.1rem; color:#6b7c8f; font-weight:650; }
.footer-note { text-align:center; color:#98a6b4; font-size:.78rem; margin-top:2.8rem; }
@media (max-width:700px) {
  .block-container { padding:1rem .85rem 2rem; }
  .topbar { margin-bottom:2rem; }
  .language-pill { display:none; }
  .hero h1 { font-size:2.5rem; }
  .st-key-translation_card { border-radius:20px; }
}
</style>
"""


def _tao_csv(cac_cau: list[str], cac_ket_qua: list[ThongTinBanDich]) -> bytes:
    bo_dem = io.StringIO()
    writer = csv.writer(bo_dem)
    writer.writerow(["English", "Tiếng Việt"])
    for cau, ket_qua in zip(cac_cau, cac_ket_qua):
        writer.writerow([cau, ket_qua.van_ban])
    return bo_dem.getvalue().encode("utf-8-sig")


def _dich_an_toan(bo_dich: BoDich, cac_cau: list[str]) -> list[ThongTinBanDich] | None:
    try:
        return bo_dich.dich(cac_cau, beam_size=4, do_dai_toi_da=96)
    except Exception as exc:
        print(f"Lỗi dịch: {exc}", file=sys.stderr)
        st.error("Chưa thể tạo bản dịch lúc này. Vui lòng thử lại sau ít phút.")
        return None


def _xoa_noi_dung() -> None:
    st.session_state["van_ban_nguon"] = ""
    st.session_state.pop("ket_qua_cuoi", None)


def _dat_cau_goi_y(noi_dung: str) -> None:
    st.session_state["van_ban_nguon"] = noi_dung
    st.session_state.pop("ket_qua_cuoi", None)


def main() -> None:
    st.set_page_config(
        page_title="ENVI Translate",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    with st.spinner("Đang chuẩn bị trình dịch…"):
        try:
            bo_dich = nap_bo_dich(CHECKPOINT, TOKENIZER, CONFIG)
        except Exception as exc:
            print(f"Lỗi khởi động: {exc}", file=sys.stderr)
            st.error("Ứng dụng chưa sẵn sàng. Vui lòng kiểm tra lại dữ liệu và khởi động lại.")
            st.stop()

    st.markdown(
        """
        <nav class="topbar">
          <div class="brand"><span class="brand-mark">✦</span> ENVI Translate</div>
          <div class="language-pill">English&nbsp;&nbsp;→&nbsp;&nbsp;Tiếng Việt</div>
        </nav>
        <section class="hero">
          <div class="hero-kicker">Dịch thuật đơn giản và tự nhiên</div>
          <h1>Hiểu nhau, không còn <span>khoảng cách.</span></h1>
          <p>Chuyển câu tiếng Anh thành tiếng Việt rõ ràng chỉ trong vài giây. Nhập nội dung hoặc tải lên một tệp để bắt đầu.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    tab_van_ban, tab_tep = st.tabs(["Dịch văn bản", "Dịch từ tệp"])

    with tab_van_ban:
        with st.container(border=True, key="translation_card"):
            cot_nguon, cot_dich = st.columns(2, gap="medium")

            with cot_nguon:
                st.markdown('<div class="language-head"><span>English</span><span>Ngôn ngữ nguồn</span></div>', unsafe_allow_html=True)
                cau = st.text_area(
                    "Nội dung tiếng Anh",
                    height=235,
                    placeholder="Nhập nội dung tiếng Anh…",
                    label_visibility="collapsed",
                    max_chars=2000,
                    key="van_ban_nguon",
                )

            with cot_dich:
                st.markdown('<div class="language-head"><span>Tiếng Việt</span><span class="ready">Sẵn sàng</span></div>', unsafe_allow_html=True)
                ket_qua_cuoi = st.session_state.get("ket_qua_cuoi")
                if ket_qua_cuoi:
                    noi_dung = html.escape(ket_qua_cuoi.van_ban)
                    st.markdown(f'<div class="output-card">{noi_dung}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="output-card output-placeholder">Bản dịch sẽ xuất hiện tại đây…</div>', unsafe_allow_html=True)

            khoang_trong, cot_nut, cot_xoa, khoang_trong_2 = st.columns([1.15, 1.5, .72, 1.15])
            nut_dich = cot_nut.button("Dịch sang tiếng Việt  →", type="primary", use_container_width=True)
            cot_xoa.button("Xóa", use_container_width=True, on_click=_xoa_noi_dung)

        if nut_dich:
            if not cau.strip():
                st.warning("Hãy nhập nội dung tiếng Anh trước khi dịch.")
            else:
                with st.spinner("Đang tạo bản dịch tự nhiên nhất…"):
                    cac_ket_qua = _dich_an_toan(bo_dich, [cau.strip()])
                if cac_ket_qua:
                    ket_qua = cac_ket_qua[0]
                    st.session_state["ket_qua_cuoi"] = ket_qua
                    st.session_state.setdefault("lich_su", []).insert(
                        0, (cau.strip(), ket_qua.van_ban)
                    )
                    st.session_state["lich_su"] = st.session_state["lich_su"][:5]
                    st.rerun()

        st.markdown('<div class="section-title">Thử một câu gợi ý</div>', unsafe_allow_html=True)
        st.markdown('<div class="hint">Chọn một câu để xem bản dịch ngay.</div>', unsafe_allow_html=True)
        vi_du = [
            ("Giao tiếp", "Language helps us understand each other."),
            ("Giáo dục", "Education can change the future of a community."),
            ("Công nghệ", "Technology should make everyday life easier."),
        ]
        cac_cot = st.columns(3)
        for cot, (nhan, noi_dung) in zip(cac_cot, vi_du):
            cot.button(
                nhan,
                key=f"goi_y_{nhan}",
                use_container_width=True,
                on_click=_dat_cau_goi_y,
                args=(noi_dung,),
            )

        if st.session_state.get("lich_su"):
            with st.expander("Các bản dịch gần đây"):
                for en, vi in st.session_state["lich_su"]:
                    st.markdown(
                        f'<div class="history-card"><div class="source">{html.escape(en)}</div>'
                        f'<div class="target">{html.escape(vi)}</div></div>',
                        unsafe_allow_html=True,
                    )

    with tab_tep:
        st.markdown('<div class="section-title">Dịch nhiều câu cùng lúc</div>', unsafe_allow_html=True)
        st.markdown('<div class="hint">Tải lên tệp TXT mã hóa UTF-8, mỗi dòng là một câu tiếng Anh.</div>', unsafe_allow_html=True)
        tep = st.file_uploader("Chọn tệp văn bản", type=["txt"])
        cac_cau: list[str] = []
        if tep is not None:
            try:
                cac_cau = [
                    dong.strip()
                    for dong in tep.getvalue().decode("utf-8-sig").splitlines()
                    if dong.strip()
                ]
            except UnicodeDecodeError:
                st.error("Không đọc được tệp. Hãy lưu lại tệp ở định dạng UTF-8.")
            if len(cac_cau) > 100:
                st.info("Ứng dụng sẽ dịch 100 dòng đầu tiên trong tệp.")
                cac_cau = cac_cau[:100]
            if cac_cau:
                st.success(f"Đã nhận {len(cac_cau)} câu.")

        if st.button("Dịch tệp", type="primary", disabled=not cac_cau):
            tien_do = st.progress(0, text="Đang chuẩn bị…")
            tat_ca: list[ThongTinBanDich] = []
            thanh_cong = True
            for bat_dau in range(0, len(cac_cau), 2):
                batch = cac_cau[bat_dau : bat_dau + 2]
                ket_qua_batch = _dich_an_toan(bo_dich, batch)
                if ket_qua_batch is None:
                    thanh_cong = False
                    break
                tat_ca.extend(ket_qua_batch)
                da_xong = min(bat_dau + len(batch), len(cac_cau))
                tien_do.progress(da_xong / len(cac_cau), text=f"Đã dịch {da_xong}/{len(cac_cau)} câu")
            tien_do.empty()
            if thanh_cong:
                st.session_state["ket_qua_tep"] = (cac_cau, tat_ca)
                st.success("Tệp đã được dịch xong.")

        if "ket_qua_tep" in st.session_state:
            cau_goc, ban_dich = st.session_state["ket_qua_tep"]
            bang = [
                {"English": cau, "Tiếng Việt": ket_qua.van_ban}
                for cau, ket_qua in zip(cau_goc, ban_dich)
            ]
            st.dataframe(bang, use_container_width=True, hide_index=True)
            st.download_button(
                "Tải bản dịch",
                data=_tao_csv(cau_goc, ban_dich),
                file_name="ban_dich_en_vi.csv",
                mime="text/csv",
                type="primary",
            )

    st.markdown(
        '<div class="footer-note">ENVI Translate · Đồ án dịch máy Anh–Việt · 2026</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
