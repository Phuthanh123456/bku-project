# Việc tiếp theo — trạng thái 08/09/2026

## Đã hoàn thành và kiểm bằng số thật được

- Dữ liệu: train 131.339, dev 1.553, test 1.268; đã loại 21 cặp rò rỉ.
- Checkpoint thật seed 42 bước 6.000 đã tải/đọc metadata.
- Full evaluation: dev Greedy 26,24; test Greedy 29,74; Beam-4 30,69 BLEU.
- Beam Search + KV-cache: cache trùng decode đầy đủ; Beam tăng +0,95 BLEU.
- Benchmark CPU 200 câu: cache tăng throughput 0,77→2,41 câu/giây.
- Phân tích lỗi thủ công 30 câu, CSV và biểu đồ.
- Streamlit + Docker: image build được, Compose healthy tại localhost:8501,
  model dịch thật trong container và sinh được attention heatmap.
- Báo cáo tổng hợp và slide `.pptx` 15 trang đã sinh.
- Test không-slow: 104 passed, 1 skipped, 1 deselected.

## Việc còn thiếu bắt buộc cần GPU/tài khoản

### 1. Huấn luyện 15 lượt còn lại

Đã có 1/16 lượt (cải tiến seed 42). Còn:

- cải tiến seed 1337;
- vanilla 2017 seed 42, 1337;
- A1–A6, mỗi cấu hình seed 42 và 1337.

Tất cả đã khóa đúng 6.000 optimizer step, không early stopping sớm. Trên Kaggle
T4 chạy:

```bash
python scripts/run_ablations.py --execute \
  --repo-hub <tai-khoan/envi-nmt-scratch-transformer>
```

Nếu phiên sắp hết giờ, dùng `--resume` ở lượt kế tiếp. Manifest đầy đủ nằm tại
`results/ke_hoach_ablation.csv`. Sau khi kéo điểm về:

```bash
python scripts/summarize_ablations.py
```

Chỉ khi đủ hai seed mới điền mean/std và kết luận cải tiến vs vanilla/A1–A6.

### 2. Tạo URL Hugging Face Space

Code và root Dockerfile đã sẵn; máy hiện tại không có `HF_TOKEN`, nên chưa thể
tạo/push Space thay mặt nhóm. Tạo Docker Space rồi làm theo
`docs/huong_dan_chay.md`; chép URL thật vào README và báo cáo.

### 3. Phần mở rộng tùy chọn

- Export/benchmark ONNX và ONNX INT8.
- Gán nhãn lỗi bởi người thứ hai để đo độ đồng thuận.
- Chạy lại test slow overfit 50 câu trên GPU; artifact lần chạy trước đã có,
  còn lượt CPU hiện tại quá lâu nên không dùng làm bằng chứng mới.

## Không được làm

- Không điền số ablation giả hoặc dùng smoke test làm kết quả.
- Không so trực tiếp 29,74 với paper nếu chưa khớp toàn bộ tokenization.
- Không đổi budget giữa hai vế, không lặng lẽ bật warmup cho A6 khi phân kỳ.
- Không commit Hugging Face token/checkpoint `.pt` vào Git.
