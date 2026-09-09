# Hướng dẫn chạy demo — TASK 20

Demo nhận câu tiếng Anh, dịch sang tiếng Việt bằng Greedy hoặc Beam, báo thời
gian suy luận, vẽ cross-attention layer cuối và nhận file `.txt` tối đa 200 dòng.

## Cách nhanh nhất: Docker Compose

Yêu cầu Docker Desktop còn ít nhất 3 GB dung lượng trống. Tại root repo chạy:

```bash
docker compose up --build
```

Mở <http://localhost:8501>. Compose dùng checkpoint và tokenizer local nếu đã
có. Nếu chưa có checkpoint local, bỏ biến `DUONG_DAN_CHECKPOINT` trong compose
để giao diện tải bản public từ `mgbao/envi-nmt-scratch-transformer`.

## Chạy trực tiếp bằng Python

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py --config configs/base.yaml
streamlit run src/nmt/serve/ui.py
```

Checkpoint/tokenizer public được tải tự động khi file local chưa có. Có thể ghi
đè đường dẫn bằng ba biến môi trường:

```text
DUONG_DAN_CONFIG
DUONG_DAN_CHECKPOINT
DUONG_DAN_TOKENIZER
```

## Đưa lên Hugging Face Docker Space

1. Tạo Space mới, chọn SDK **Docker** và để chế độ public/private tùy nhu cầu.
2. Chép `README_SPACE.md` thành `README.md` ở repo Space.
3. Đẩy root `Dockerfile`, `requirements-demo.txt`, `src/` và `configs/` lên Space.
4. Space chạy cổng 7860 và tự tải checkpoint public lúc khởi động đầu tiên.

Root `Dockerfile` đã được chuẩn bị đúng cổng 7860. URL công khai chưa thể điền
trong repo này vì cần tài khoản/quyền tạo Hugging Face Space của nhóm; việc đó
không được giả là đã deploy.

## Lỗi thường gặp

- **Lần đầu đứng ở “Đang nạp checkpoint”**: file khoảng 550 MB đang được tải;
  xem log container trước khi kết luận treo.
- **Không đủ RAM**: demo CPU cần khoảng 2–3 GB; tắt ứng dụng khác hoặc tăng RAM
  Docker Desktop.
- **Không thấy checkpoint khi dùng Compose**: kiểm tra file
  `checkpoints/iwslt_base_v1_seed42/tot_nhat.pt` và mount `./checkpoints`.
- **Port 8501 đã bận**: đổi vế trái của `8501:8501`, ví dụ `8502:8501`.
- **Checkpoint smoke bị từ chối**: đây là chủ ý; demo chỉ nhận checkpoint có
  metadata `che_do=that`.
