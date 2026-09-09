# Hướng dẫn giao một lượt ablation cho Bảo

Notebook mẫu: `notebooks/03_kaggle_ablation_mau.ipynb`.

## Phú làm trước khi gửi

1. Push commit chứa notebook và toàn bộ mã Phase 4 lên branch GitHub.
2. Nhắn Bảo tên branch hiện tại: `feat/ablation-baseline`.
3. Chốt một cặp cấu hình/seed chưa ai nhận. Mẫu đầu tiên là **A1 / seed 42**.
4. Không gửi token qua chat và không ghi token vào notebook.

## Bảo thao tác trên Kaggle

1. Kaggle → Code → New Notebook → File → Import Notebook, chọn
   `notebooks/03_kaggle_ablation_mau.ipynb`.
2. Settings → Accelerator → chọn GPU T4. Internet phải bật.
3. Add-ons → Secrets → thêm `HF_TOKEN` loại Write, bật công tắc cho notebook,
   sau đó Restart session.
4. Kiểm cell cấu hình đang là:

   ```python
   CONFIG = "configs/ablation_a1_layernorm.yaml"
   SEED = 42
   REPO_HUB = "mgbao/envi-nmt-scratch-transformer"
   GIT_REF = "feat/ablation-baseline"
   CHI_SMOKE = True
   ```

5. Run All lần 1. Chỉ khi hiện `SMOKE ĐÃ QUA` mới sang bước tiếp.
6. Đổi duy nhất `CHI_SMOKE = False`, Run All lần 2.
7. Nếu cuối phiên chưa đủ `6000/6000`, mở phiên mới và Run All lại với đúng
   `CONFIG` + `SEED`. Cờ `--tiep-tuc` tự tải checkpoint trên Hub.
8. Khi đủ bước, notebook tự chấm tst2013, đẩy CSV lên Hub và tạo file
   `<tên_lượt>_ket_qua.zip` trong Output.
9. Bảo gửi lại tên lượt, BLEU, chrF++, link checkpoint Hub và file zip.

## Chia 15 lượt còn thiếu

Không chạy lại `improved / seed 42`. Các lượt còn lại là:

| mã | seed cần chạy |
|---|---|
| improved | 1337 |
| vanilla | 42, 1337 |
| a1 | 42, 1337 |
| a2 | 42, 1337 |
| a3 | 42, 1337 |
| a4 | 42, 1337 |
| a5 | 42, 1337 |
| a6 | 42, 1337 |

Mỗi người nhận lượt nào phải báo trước trong nhóm. Không đổi ngân sách 6.000 bước,
tokenizer, tập test hay cách chấm; nếu đổi thì kết quả không còn so sánh công bằng.

## Khi Bảo trả kết quả

Tải checkpoint vào đúng đường dẫn ghi trong `results/ke_hoach_ablation.csv`, gộp
`diem_chinh.csv`, rồi chạy:

```bash
python scripts/run_ablations.py
python scripts/summarize_ablations.py
```

Chỉ kết luận mean/std sau khi mỗi cấu hình có đủ hai seed thật.
