"""Entrypoint dành riêng cho Streamlit Community Cloud."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))

MODEL_REPO = "phudeeptry/envi-translate-model"
os.environ["DUONG_DAN_CHECKPOINT"] = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="inference_state.pt",
)
os.environ["DUONG_DAN_TOKENIZER"] = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="tokenizer.json",
)
os.environ["DUONG_DAN_CONFIG"] = str(GOC / "configs/abl3000_base_seed42.yaml")

from nmt.serve.ui import main  # noqa: E402


main()
