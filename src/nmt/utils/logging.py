"""Ghi log TensorBoard + CSV — TASK 01 (Phú).

Ghi song song hai định dạng, có lý do:

- **TensorBoard** để xem trong lúc train, kéo chuột phóng to đoạn nghi ngờ.
- **CSV** để vẽ lại hình cho báo cáo. Không thể lấy số từ file event của
  TensorBoard một cách gọn gàng, mà đồ án này cần vẽ tay khá nhiều hình:
  hai đường loss chồng lên nhau của thí nghiệm giết phiên, đường learning rate,
  BLEU theo epoch, và hai bảng ablation.

Điểm quan trọng cho TASK 14 (giết phiên): file CSV mở ở chế độ **nối thêm**
(append). Khi phiên Kaggle bị rớt rồi mở phiên mới, log cũ vẫn còn nguyên nên
vẽ được một đường liền mạch qua các lần bị giết. Đó chính là hình có giá trị
nhất của cả đồ án. Vì vậy phải đẩy cả thư mục log lên Hugging Face Hub cùng
checkpoint, đừng chỉ đẩy mỗi trọng số.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class BoGhiLog:
    """Ghi số liệu ra TensorBoard và CSV cùng lúc.

    Ví dụ:
        >>> log = BoGhiLog("results/logs/iwslt_base_v1")
        >>> log.ghi(buoc=100, loss_train=4.21, learning_rate=7e-4)
        >>> log.dong()
    """

    def __init__(self, thu_muc: str | Path, dung_tensorboard: bool = True) -> None:
        self.thu_muc = Path(thu_muc)
        self.thu_muc.mkdir(parents=True, exist_ok=True)

        self.duong_dan_csv = self.thu_muc / "metrics.csv"
        self._cot: list[str] | None = None
        self._doc_cot_da_co()

        self.tb = None
        if dung_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.thu_muc / "tensorboard"))
            except ImportError:
                # Thiếu tensorboard thì vẫn phải train được, chỉ mất phần xem trực quan.
                print("[log] Không có tensorboard, chỉ ghi CSV.")

    def _doc_cot_da_co(self) -> None:
        """Đọc lại header khi nối thêm vào file CSV đã có (trường hợp resume)."""
        if self.duong_dan_csv.exists() and self.duong_dan_csv.stat().st_size > 0:
            with self.duong_dan_csv.open("r", encoding="utf-8", newline="") as f:
                dong_dau = f.readline().strip()
            if dong_dau:
                self._cot = dong_dau.split(",")

    def ghi(self, buoc: int, **so_lieu: float) -> None:
        """Ghi một hàng số liệu tại bước huấn luyện `buoc`.

        Args:
            buoc: số bước huấn luyện toàn cục (global step).
            **so_lieu: ví dụ loss_train=4.21, loss_dev=4.55, bleu_dev=18.3
        """
        hang: dict[str, Any] = {"buoc": buoc, **so_lieu}

        if self._cot is None:
            self._cot = list(hang.keys())
            with self.duong_dan_csv.open("w", encoding="utf-8", newline="") as f:
                csv.DictWriter(f, fieldnames=self._cot).writeheader()

        with self.duong_dan_csv.open("a", encoding="utf-8", newline="") as f:
            # extrasaction="ignore" để một lần ghi thiếu cột không làm hỏng cả file.
            csv.DictWriter(f, fieldnames=self._cot, extrasaction="ignore").writerow(hang)

        if self.tb is not None:
            for ten, gia_tri in so_lieu.items():
                if isinstance(gia_tri, (int, float)):
                    self.tb.add_scalar(ten, gia_tri, buoc)

    def ghi_van_ban(self, ten: str, noi_dung: str, buoc: int = 0) -> None:
        """Ghi văn bản, ví dụ vài câu dịch mẫu ở mỗi mốc đánh giá."""
        if self.tb is not None:
            self.tb.add_text(ten, noi_dung, buoc)
        with (self.thu_muc / f"{ten}.txt").open("a", encoding="utf-8") as f:
            f.write(f"=== bước {buoc} ===\n{noi_dung}\n\n")

    def dong(self) -> None:
        if self.tb is not None:
            self.tb.flush()
            self.tb.close()

    def __enter__(self) -> "BoGhiLog":
        return self

    def __exit__(self, *_: object) -> None:
        self.dong()
