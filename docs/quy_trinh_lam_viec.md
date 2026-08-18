# Quy trình làm việc — đọc trước khi viết dòng code đầu tiên

Tài liệu này là bản rút gọn để dùng hằng ngày. Bản đầy đủ nằm ở
`ENVI-NMT_21_Task.pdf` và đề cương bản thảo lần 3.

---

## 1. Cài môi trường (làm một lần)

```bash
git clone <url-repo>
cd english-vietnamese-nmt-scratch-transformer-pytorch

python -m venv .venv
# Windows:   .venv\Scripts\activate
# Ubuntu:    source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

pytest -q          # phải xanh. Nếu đỏ thì báo Phú trước khi làm tiếp.
```

Nếu `pytest -q` đỏ ngay từ đầu thì môi trường có vấn đề, đừng bắt đầu code.

---

## 2. Nhánh git

Mỗi Chain một nhánh. **Không ai push thẳng vào `main`.**

| Chain | Người | Nhánh | Task |
|---|---|---|---|
| 0 — Khởi động | Phú | `main` | 01 |
| A — Dữ liệu | My | `feat/data` | 02, 03, 04, 10, 16 |
| B — Mô hình | Quân (05, 06, 09), Bảo (07, 08) | `feat/model` | 05–09 |
| C — Huấn luyện | Quân (11, 13), Bảo (12, 14, 15) | `feat/training` | 11–15 |
| D — Đánh giá | My | `eval-baseline` | 16 |
| E — Ablation & Đóng gói | Phú | `feat/ablation`, `feat/deploy` | 17–21 |

```bash
git checkout main
git pull
git checkout -b feat/data
# ... làm việc ...
git add -A
git commit -m "TASK 02: lam sach du lieu IWSLT"
git push -u origin feat/data
```

Quy ước commit: bắt đầu bằng `TASK NN:` để sau này `git log` đọc ra được ai làm gì lúc nào.

---

## 3. Luồng bàn giao — ai chờ ai

```
TASK 01 (Phú, main)
   |
   +--> Chain A: My  02 -> 03 -> 04 ------> push feat/data
   |
   +--> Chain B: Quân 05, 06 | Bảo 07, 08   (bốn task ĐỘC LẬP, làm thứ tự nào cũng được)
   |         xong cả bốn -> Quân làm 09 -> push feat/model, gắn tag model-ready
   |
   +--> Chain E: Phú có thể bắt đầu TASK 20 sớm ngay sau TASK 09

   ĐIỂM HỢP LƯU 1 — TASK 10 (My):
       pull feat/model của Quân + gộp với feat/data của chính mình
       chạy 12 bài test + học thuộc 50 câu
       ĐẠT -> push nhánh verified-model     <- CỔNG CHẶN cuối Tuần 2

   Chain C: Quân 11 (pull verified-model) | Bảo 12 (làm song song, không phụ thuộc)
            -> 13 Quân (cần 09 + 12) -> 14 Bảo (cần 12 + 13) -> 15 Bảo

   ĐIỂM HỢP LƯU 2 — TASK 15 (Bảo):
       đẩy checkpoint tốt nhất lên HUGGING FACE HUB (không phải git)
       My pull checkpoint đó về làm TASK 16

   TASK 16 (My) -> push eval-baseline
       -> Phú pull eval-baseline + feat/training để làm 17, 18, 19
   TASK 21 (Phú) — task cuối, cần 16 đến 20 đều xong
```

**Ba điểm chờ duy nhất của cả dự án:** TASK 01 → TASK 10 → TASK 15.
Ngoài ba chỗ đó ra thì mọi người làm song song được.

---

## 4. Bốn mốc kiểm soát tiến độ

| Mốc | Điều kiện | Nếu không đạt |
|---|---|---|
| **Cuối Tuần 2** | TASK 10 đạt 12/12 test + học thuộc 50 câu | Chưa qua thì **chưa** làm TASK 11 hay 15 |
| **Đầu Tuần 3** | TASK 11 chốt cấu hình bằng số liệu | Đây là căn cứ trả lời mentor về số layer / số head |
| **Cuối Tuần 3** | TASK 15 có mô hình train xong | Cắt TASK 19, thu gọn TASK 17 còn mỗi A1 |
| **Tuần 4** | TASK 17 + 18 đủ 6 thí nghiệm | Ưu tiên A1 vì mentor nêu trực tiếp |

Quy tắc của anh Huy: **ưu tiên số một là nộp một kết quả hoàn chỉnh dù đơn giản.**
Trễ thì cắt ablation, đừng cắt phần chạy được.

---

## 5. Sáu thí nghiệm ablation — chạy như thế nào

Không sửa `base.yaml`. Mỗi thí nghiệm là một file riêng kế thừa base và ghi đè **đúng một yếu tố**.

```bash
# baseline
python scripts/train.py --config configs/base.yaml --seed 42
python scripts/train.py --config configs/base.yaml --seed 1337

# A1 — RMSNorm vs LayerNorm  (ưu tiên cao nhất, mentor hỏi trực tiếp)
python scripts/train.py --config configs/ablation_a1_layernorm.yaml --seed 42
python scripts/train.py --config configs/ablation_a1_layernorm.yaml --seed 1337
```

| Mã | File | Đổi cái gì | Task |
|---|---|---|---|
| A1 | `ablation_a1_layernorm.yaml` | RMSNorm → LayerNorm | 17 |
| A2 | `ablation_a2_warmup.yaml` | Warmup tắt → bật | 18 |
| A3 | `ablation_a3_label_smoothing.yaml` | Label Smoothing 0 → 0.1 | 18 |
| A4 | `ablation_a4_sincos.yaml` | RoPE → sin-cos | 17 |
| A5 | `ablation_a5_relu.yaml` | SwiGLU → ReLU (d_ff 688 → 1024) | 17 |
| A6 | `ablation_a6_post_norm.yaml` | Pre-Norm → Post-Norm | 17 |

**Ba quy tắc bắt buộc:**

1. Mỗi thí nghiệm chạy **tối thiểu 2 seed**, báo cáo cả độ lệch giữa các seed.
2. Mọi cấu hình chạy **cùng một ngân sách bước**, không chạy tới hội tụ.
   (Chạy tới hội tụ thì mỗi cấu hình tốn thời gian khác nhau và không so được.)
3. Chỉ đổi **đúng một yếu tố**. A5 đổi cả `d_ff` là ngoại lệ có chủ ý, để hai
   phương án có số tham số tương đương — nếu không thì đang so hai mô hình khác cỡ.

**Hai kết quả cần biết trước để không tưởng nhầm là bug:**

- **A2 có thể gần như không thấy khác biệt.** Pre-Norm vốn đã làm giảm nhu cầu
  warmup (arxiv 2002.04745). Đó là điều bài báo dự đoán. Viết thẳng vào báo cáo
  kèm trích dẫn — giải thích được một kết quả bằng lý thuyết đã công bố thì
  thuyết phục hơn nhiều so với chỉ đưa ra con số.
- **A6 (Post-Norm) có thể phân kỳ** khi chạy với `scheduler: co_dinh`, vì
  Post-Norm gần như bắt buộc phải có warmup. **Đó chính là kết quả**, hãy ghi
  lại chứ đừng lặng lẽ bật warmup lên rồi báo cáo như thể mọi thứ bình thường.

---

## 6. Bốn cái bẫy fp16 — dán lên màn hình

GPU T4 của Kaggle là kiến trúc Turing, compute capability 7.5, **không có phần cứng bf16**.
Chốt cứng `fp16` trong YAML. Không dùng `torch.cuda.is_bf16_supported()` để tự chọn — hàm
này tính cả trường hợp giả lập nên trả về `True` ngay trên T4, chạy được nhưng chậm hơn cả
fp32, khiến nhóm đo tốc độ ở TASK 11 rồi kết luận nhầm là T4 không đủ sức.
Muốn kiểm tự động thì dùng `torch.cuda.get_device_capability()[0] >= 8`.

| Ở đâu | Phải làm | Nếu sai |
|---|---|---|
| Che mask trong attention | `torch.finfo(scores.dtype).min` | `-1e9` → âm vô cùng ở fp16 → hàng bị mask toàn bộ ra **NaN** |
| Bảng góc quay RoPE | tính ở **float32** rồi mới ép về | từ vị trí ~100 mất độ phân giải, **RoPE hỏng âm thầm**, bài test 4 vẫn báo đạt |
| Trung bình bình phương RMSNorm | tính ở **float32** rồi mới ép về | bình phương vượt 65504 → tràn → NaN cả mô hình |
| GradScaler + clip norm | `scaler.unscale_(optimizer)` **trước** khi clip | đang clip gradient đã bị nhân hệ số giãn, ngưỡng 1.0 vô nghĩa, **không báo lỗi** |

Ba lỗi đầu **không bị 11 bài test đầu phát hiện** vì 11 bài đó chạy ở float32.
Chỉ bài test 12 (chạy trong `torch.autocast` fp16) mới bắt được.

---

## 7. Ba điểm thiết kế kiến trúc không được quên

1. **RoPE chỉ áp cho self-attention.** Không áp cho cross-attention, không áp cho
   tensor value. (Bài test 11 và bài test 4)
2. **Pre-Norm là `x + SubLayer(Norm(x))`**, không phải `Norm(x + SubLayer(x))`. (Bài test 10)
3. **Bắt buộc có một RMSNorm cuối** ngay trước lớp Linear xuất từ vựng khi dùng
   Pre-Norm. Quên bước này **không làm chương trình báo lỗi**. (Bài test 10)

---

## 8. Bảo mật Hugging Face token

- **Không bao giờ** viết token vào code hay commit lên GitHub.
- Trên Kaggle: **Add-ons → Secrets**, tên `HF_TOKEN`, token loại **Write**.
- Trên máy cá nhân: biến môi trường `HF_TOKEN`, hoặc copy `.env.example` thành `.env`.
- `tests/test_khong_dung_lop_dung_san.py` có một bài quét token lỡ commit — nhưng
  nó chỉ là lưới an toàn cuối, đừng dựa vào nó.
- Lỡ commit token rồi thì: **thu hồi token trên Hugging Face ngay**, tạo token mới,
  rồi mới xử lý lịch sử git. Xóa khỏi commit mà không thu hồi là vô nghĩa.

Nhắc lại nhận xét 2 của anh Huy: **không gom nhiều tài khoản Kaggle để lách quota** —
vi phạm điều khoản, dễ bị ban. Mỗi thành viên dùng một tài khoản cá nhân của mình.
Cơ chế checkpoint qua HF Hub vẫn giữ, nhưng đừng ghi chiến lược multi-account vào báo cáo.

---

## 9. Trước khi push — kiểm ba thứ

```bash
pytest -q                 # test phải xanh
git status                # không có file .pt, .json tokenizer, hay data/ lọt vào
git diff --cached         # đọc lại lần cuối, tìm token lỡ dán
```

---

## 10. Dữ liệu

IWSLT 2015 English–Vietnamese:

- `train` ~133.318 cặp câu
- `tst2012` 1.269 cặp — **tập dev**, theo dõi trong lúc train
- `tst2013` 1.269 cặp — **tập test**, chỉ dùng chấm điểm cuối cùng

**Không đụng PhoMT** trong khóa học. 3 triệu câu là quá sức với Kaggle
(nhận xét 1 của anh Huy). PhoMT để dành cho hướng phát triển sau khóa học.

Ghi lại nguồn tải và số dòng của từng file để sau này người khác tái lập được.
