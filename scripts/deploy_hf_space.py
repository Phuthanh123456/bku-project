"""Tạo/cập nhật Hugging Face Docker Space cho demo ENVI Translate."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


GOC = Path(__file__).resolve().parents[1]


def _them_thu_muc(
    operations: list[CommitOperationAdd], thu_muc: Path, dich: str
) -> None:
    for tep in sorted(thu_muc.rglob("*")):
        if tep.is_file() and "__pycache__" not in tep.parts and tep.suffix != ".pyc":
            tuong_doi = tep.relative_to(thu_muc).as_posix()
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{dich}/{tuong_doi}", path_or_fileobj=tep
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space-id", help="Ví dụ: phudeeptry/envi-translate")
    args = parser.parse_args()

    api = HfApi()
    tai_khoan = api.whoami()["name"]
    space_id = args.space_id or f"{tai_khoan}/envi-translate"
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )

    operations: list[CommitOperationAdd] = [
        CommitOperationAdd(
            path_in_repo="Dockerfile",
            path_or_fileobj=GOC / "deploy/huggingface-space/Dockerfile",
        ),
        CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=GOC / "deploy/huggingface-space/README.md",
        ),
        CommitOperationAdd(
            path_in_repo="scripts/start_demo.py",
            path_or_fileobj=GOC / "scripts/start_demo.py",
        ),
        CommitOperationAdd(
            path_in_repo="configs/abl3000_base_seed42.yaml",
            path_or_fileobj=GOC / "configs/abl3000_base_seed42.yaml",
        ),
        CommitOperationAdd(
            path_in_repo=".streamlit/config.toml",
            path_or_fileobj=GOC / ".streamlit/config.toml",
        ),
    ]
    _them_thu_muc(operations, GOC / "src", "src")

    commit = api.create_commit(
        repo_id=space_id,
        repo_type="space",
        operations=operations,
        commit_message="Deploy ENVI Translate Docker demo",
    )
    print(f"Space: https://huggingface.co/spaces/{space_id}")
    print(f"Commit: {commit.oid}")


if __name__ == "__main__":
    main()
