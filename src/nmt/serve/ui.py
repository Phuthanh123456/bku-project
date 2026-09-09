"""Demo Streamlit dịch Anh→Việt — TASK 20."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from time import perf_counter

import torch

GOC = Path(__file__).resolve().parents[3]
if str(GOC / "src") not in sys.path:
    sys.path.insert(0, str(GOC / "src"))

HF_REPO_MAC_DINH = "mgbao/envi-nmt-scratch-transformer"
HF_CHECKPOINT = "checkpoints/iwslt_base_v1_seed42/tot_nhat.pt"
HF_TOKENIZER = "artifacts/tokenizer/tokenizer.json"


def _tim_hoac_tai(env_name: str, mac_dinh: Path, hf_filename: str) -> Path:
    duong_dan = Path(os.getenv(env_name, str(mac_dinh)))
    if duong_dan.exists():
        return duong_dan
    from huggingface_hub import hf_hub_download

    repo = os.getenv("HF_MODEL_REPO", HF_REPO_MAC_DINH)
    return Path(hf_hub_download(repo_id=repo, filename=hf_filename))


def _nap_tai_nguyen():
    from nmt.data import nap_tokenizer
    from nmt.model.transformer import TransformerNMT
    from nmt.training.checkpoint import CHE_DO_THAT, nap_checkpoint
    from nmt.utils import nap_config

    config_path = Path(os.getenv(
        "DUONG_DAN_CONFIG", str(GOC / "configs/iwslt_base_v1_seed42.yaml")
    ))
    if not config_path.exists():
        config_path = GOC / "configs/base.yaml"
    checkpoint_path = _tim_hoac_tai(
        "DUONG_DAN_CHECKPOINT", GOC / HF_CHECKPOINT, HF_CHECKPOINT
    )
    tokenizer_path = _tim_hoac_tai(
        "DUONG_DAN_TOKENIZER", GOC / HF_TOKENIZER, HF_TOKENIZER
    )

    cfg = nap_config(config_path)
    tokenizer = nap_tokenizer(tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNMT(cfg).to(device)
    info = nap_checkpoint(
        checkpoint_path, model, map_location=device,
        che_do_mong_doi=CHE_DO_THAT, khoi_phuc_rng=False,
    )
    model.eval()
    return cfg, tokenizer, model, device, info


def _dich_mot_cau(cfg, tokenizer, model, device, text: str, search: str, beam_size: int):
    from nmt.data import BOS_ID, EOS_ID, PAD_ID
    from nmt.inference.search import beam_search, greedy_search
    from nmt.model.masking import tao_causal_mask, tao_padding_mask

    token_ids = tokenizer.encode(text.strip()).ids[: cfg.du_lieu.do_dai_toi_da]
    if not token_ids:
        raise ValueError("Câu nguồn đang trống")
    src_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    src_mask = tao_padding_mask(src_ids, PAD_ID)

    bat_dau = perf_counter()
    if search == "Beam":
        output = beam_search(
            model, src_ids, src_mask, BOS_ID, EOS_ID,
            beam_size=beam_size,
            he_so_phat_do_dai=cfg.sinh_cau.he_so_phat_do_dai,
            do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich,
            dung_kv_cache=True,
        )
    else:
        output = greedy_search(
            model, src_ids, src_mask, BOS_ID, EOS_ID,
            do_dai_toi_da=cfg.sinh_cau.do_dai_toi_da_khi_dich,
            dung_kv_cache=True,
        )
    elapsed = perf_counter() - bat_dau

    output_ids = output[0].tolist()
    if EOS_ID in output_ids:
        output_ids = output_ids[: output_ids.index(EOS_ID) + 1]
    decoded = tokenizer.decode(
        [i for i in output_ids if i not in (PAD_ID, BOS_ID, EOS_ID)],
        skip_special_tokens=True,
    )

    # Một lượt teacher-forced chỉ để lấy cross-attention của layer cuối. Lượt
    # này không được tính vào độ trễ dịch ở trên.
    attention = None
    if len(output_ids) > 1:
        captured = []
        hook = model.decoder_layers[-1].cross_attn.register_forward_hook(
            lambda module, inputs, result: captured.append(result[1].detach().cpu())
        )
        try:
            memory = model.encode(src_ids, src_mask)
            tgt_input = torch.tensor([output_ids[:-1]], dtype=torch.long, device=device)
            tgt_mask = tao_causal_mask(tgt_input.size(1), device=device)
            model.decode(tgt_input, memory, tgt_mask, src_mask)
        finally:
            hook.remove()
        if captured:
            attention = captured[-1].mean(dim=1)[0].numpy()

    src_tokens = [tokenizer.id_to_token(i) or str(i) for i in token_ids]
    tgt_tokens = [tokenizer.id_to_token(i) or str(i) for i in output_ids[1:]]
    return decoded, elapsed, attention, src_tokens, tgt_tokens


def _ve_attention(attention, src_tokens, tgt_tokens):
    import matplotlib.pyplot as plt

    width = max(7, len(src_tokens) * 0.45)
    height = max(3, len(tgt_tokens) * 0.32)
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(attention[:len(tgt_tokens), :len(src_tokens)], aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(src_tokens)), src_tokens, rotation=55, ha="right")
    ax.set_yticks(range(len(tgt_tokens)), tgt_tokens)
    ax.set_xlabel("Token nguồn (EN)")
    ax.set_ylabel("Token sinh (VI)")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    return fig


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="EN→VI NMT", page_icon="🌐", layout="wide")
    st.title("Dịch máy Anh → Việt")
    st.caption("Transformer tự xây dựng · RoPE · RMSNorm · SwiGLU")

    @st.cache_resource(show_spinner="Đang nạp checkpoint 550 MB...")
    def load_cached():
        return _nap_tai_nguyen()

    try:
        cfg, tokenizer, model, device, info = load_cached()
    except Exception as error:
        st.error(f"Không nạp được model: {error}")
        st.stop()

    st.sidebar.success(
        f"Checkpoint bước {info['buoc']:,} · seed {info['seed']} · {device.type}"
    )
    search = st.sidebar.radio("Cách sinh câu", ["Greedy", "Beam"], horizontal=True)
    beam_size = st.sidebar.slider("Beam size", 2, 8, 4, disabled=search != "Beam")

    text = st.text_area(
        "Câu tiếng Anh",
        "When I was little, I thought my country was the best on the planet.",
        height=110,
    )
    if st.button("Dịch", type="primary"):
        with st.spinner("Đang dịch..."):
            try:
                translation, elapsed, attention, src_tokens, tgt_tokens = _dich_mot_cau(
                    cfg, tokenizer, model, device, text, search, beam_size
                )
                st.subheader("Bản dịch")
                st.write(translation)
                st.caption(f"{search} · {elapsed * 1000:.0f} ms · KV cache bật")
                if attention is not None:
                    st.subheader("Cross-attention layer cuối")
                    st.pyplot(_ve_attention(attention, src_tokens, tgt_tokens))
            except Exception as error:
                st.error(str(error))

    st.divider()
    uploaded = st.file_uploader("Dịch file .txt (mỗi dòng một câu)", type=["txt"])
    if uploaded is not None and st.button("Dịch file"):
        lines = uploaded.getvalue().decode("utf-8").splitlines()
        if len(lines) > 200:
            st.warning("Demo giới hạn 200 dòng mỗi lượt; chỉ dịch 200 dòng đầu.")
            lines = lines[:200]
        progress = st.progress(0.0)
        translations = []
        for index, line in enumerate(lines):
            if line.strip():
                translations.append(_dich_mot_cau(
                    cfg, tokenizer, model, device, line, search, beam_size
                )[0])
            else:
                translations.append("")
            progress.progress((index + 1) / max(len(lines), 1))
        content = "\n".join(translations)
        st.download_button("Tải bản dịch", content, "ban_dich_vi.txt", "text/plain")


if __name__ == "__main__":
    main()
