"""Khởi động demo ở local hoặc Hugging Face Spaces.

Ở local, Docker Compose gắn checkpoint/tokenizer vào ``/app/artifacts``.
Trên Spaces, hai tệp này được lấy từ model repository và dùng trực tiếp từ
Hugging Face cache, vì vậy không cần đưa checkpoint lớn vào repository code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


MODEL_REPO_MAC_DINH = "phudeeptry/envi-translate-model"
CHECKPOINT_TREN_HUB = "inference_state.pt"
TOKENIZER_TREN_HUB = "tokenizer.json"


def _lay_artifact(ten_bien: str, mac_dinh_local: str, ten_file_hub: str) -> str:
    duong_dan = Path(os.environ.get(ten_bien, mac_dinh_local))
    if duong_dan.is_file():
        return str(duong_dan)

    from huggingface_hub import hf_hub_download

    repo_id = os.environ.get("HF_MODEL_REPO", MODEL_REPO_MAC_DINH)
    return hf_hub_download(repo_id=repo_id, filename=ten_file_hub)


def main() -> None:
    os.environ["DUONG_DAN_CHECKPOINT"] = _lay_artifact(
        "DUONG_DAN_CHECKPOINT",
        "/app/artifacts/checkpoints/tot_nhat.pt",
        CHECKPOINT_TREN_HUB,
    )
    os.environ["DUONG_DAN_TOKENIZER"] = _lay_artifact(
        "DUONG_DAN_TOKENIZER",
        "/app/artifacts/tokenizer/tokenizer.json",
        TOKENIZER_TREN_HUB,
    )
    os.environ.setdefault(
        "DUONG_DAN_CONFIG", "/app/configs/abl3000_base_seed42.yaml"
    )

    port = os.environ.get("PORT", "8501")
    lenh = [
        "streamlit",
        "run",
        "src/nmt/serve/ui.py",
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.fileWatcherType=none",
    ]
    os.execvp(lenh[0], lenh)


if __name__ == "__main__":
    main()
