"""TASK 13 — Training Loop.  Người làm: Quân.  [Training • Bắt buộc]

Xong khi: huấn luyện liên tục 1000 bước không xuất hiện NaN.

Thành phần: AdamW, cộng dồn gradient, cắt gradient theo norm 1.0,
độ chính xác hỗn hợp fp16 kèm GradScaler.

HAI CHI TIẾT DỄ SAI KHI GHÉP GradScaler — cả hai đều KHÔNG báo lỗi gì:

1. Phải gọi scaler.unscale_(optimizer) TRƯỚC khi cắt gradient theo norm.
   Quên thì đang cắt gradient đã bị nhân hệ số giãn, ngưỡng 1.0 trở nên vô nghĩa.

2. Khi cộng dồn gradient thì chia loss cho số bước cộng dồn, và CHỈ gọi
   scaler.step() cùng scaler.update() ở bước cuối của mỗi chu kỳ.

Thứ tự đúng trong một chu kỳ cộng dồn:
    for i in range(so_buoc_cong_don):
        with autocast(dtype=float16):
            loss = tinh_loss(...) / so_buoc_cong_don
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
"""

from __future__ import annotations


class Trainer:
    def __init__(self, cfg, model, train_loader, dev_loader, logger) -> None:
        raise NotImplementedError("TASK 13 — Quân")

    def train(self) -> None:
        raise NotImplementedError("TASK 13 — Quân")

    def danh_gia(self) -> dict:
        """Loss trên tập dev. Trả về dict để ghi log."""
        raise NotImplementedError("TASK 13 — Quân")
