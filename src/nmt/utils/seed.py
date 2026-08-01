"""Cố định random seed — TASK 01 (Phú), theo nhận xét 2 của mentor.

Vì sao việc này quan trọng với đồ án này chứ không phải chuyện làm cho đẹp:

TASK 17 và 18 có 6 thí nghiệm ablation, mỗi thí nghiệm chỉ đổi đúng một yếu tố
rồi so BLEU. Nếu hai lần chạy cùng cấu hình đã lệch nhau 0,4 BLEU vì seed khác
nhau, mà chênh lệch do đổi RMSNorm sang LayerNorm cũng chỉ 0,3 BLEU, thì cả
bảng ablation không kết luận được gì. Đó là lý do mỗi ablation phải chạy tối
thiểu 2 seed và báo cáo cả độ lệch giữa các seed.

Tiêu chí "Xong khi" của TASK 01: chạy lại cùng seed cho ra kết quả giống hệt.
Bài test chứng minh nằm ở tests/test_reproducibility.py.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def dat_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Cố định seed cho toàn bộ nguồn ngẫu nhiên.

    Args:
        seed: giá trị seed.
        deterministic: bật chế độ tất định của cuDNN. Chậm hơn khoảng 5-10 phần
            trăm nhưng đổi lại chạy lại là ra kết quả giống hệt. Với đồ án này
            thì đánh đổi đó đáng, vì cả phần ablation dựa vào nó.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PYTHONHASHSEED ảnh hưởng tới thứ tự duyệt set và dict trong vài trường hợp.
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Một số kernel CUDA không có bản tất định; cờ này bắt PyTorch báo lỗi
        # rõ ràng thay vì lặng lẽ chạy bản bất định.
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Bắt buộc cho cuBLAS khi dùng CUDA >= 10.2, nếu thiếu thì phép nhân ma
        # trận vẫn có thể bất định.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_cho_worker(worker_id: int) -> None:
    """Truyền vào ``DataLoader(worker_init_fn=seed_cho_worker)``.

    Không có hàm này thì mỗi worker của DataLoader tự sinh seed riêng, nên thứ
    tự trộn dữ liệu đổi theo số worker. Hậu quả là chạy lại với num_workers khác
    nhau ra kết quả khác nhau, mà lỗi này rất khó nghĩ ra.
    """
    seed_worker = torch.initial_seed() % 2**32
    np.random.seed(seed_worker)
    random.seed(seed_worker)


def sinh_generator(seed: int = 42) -> torch.Generator:
    """Generator truyền vào ``DataLoader(generator=...)`` để cố định thứ tự trộn."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
