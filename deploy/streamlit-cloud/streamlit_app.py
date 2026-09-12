"""Entrypoint dành riêng cho Streamlit Community Cloud."""

from __future__ import annotations

import os
import traceback

import streamlit as st
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))

os.environ["HF_MODEL_REPO"] = "phudeeptry/envi-translate-model"
os.environ["HF_MODEL_CHECKPOINT"] = "inference_fp16.pt"
os.environ["HF_MODEL_TOKENIZER"] = "tokenizer.json"
os.environ["DUONG_DAN_CONFIG"] = str(GOC / "configs/abl3000_base_seed42.yaml")

try:
    from nmt.serve.ui import main  # noqa: E402
except BaseException as exc:
    st.set_page_config(page_title="ENVI Translate")
    st.error(f"Startup error: {type(exc).__name__}: {exc}")
    st.code(traceback.format_exc())
    st.stop()


main()
