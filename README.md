# ENVI-NMT — English→Vietnamese Neural Machine Translation

**Dịch máy Nơ-ron Anh → Việt bằng Transformer hiện đại tự viết từ đầu**
(RoPE • Pre-Norm • RMSNorm • SwiGLU) + Checkpoint chịu lỗi qua Hugging Face Hub

> Toàn bộ kiến trúc mạng do nhóm tự viết bằng PyTorch thuần.
> **Không** dùng `nn.Transformer`, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`
> hay bất kỳ mô hình pretrained nào **trong `src/`**.
> Các lớp tham chiếu của PyTorch chỉ được phép xuất hiện trong `tests/`.

---

## Bảng kết quả

| Tập | BLEU | chrF++ | Cách sinh câu |
|---|---|---|---|
| tst2012 (dev) | _chưa có_ | _chưa có_ | Greedy |
| tst2013 (test) | _chưa có_ | _chưa có_ | Greedy |
| tst2013 (test) | _chưa có_ | _chưa có_ | Beam = 4 |

Chuỗi chữ ký sacrebleu: _điền sau khi chạy TASK 16_

Mức đánh giá: **≥ 19** tối thiểu chấp nhận · **≥ 22** tốt · **≥ 29** rất tốt (trên tst2013).

---

## Cài đặt nhanh

```bash
git clone <url-repo>
cd english-vietnamese-nmt-scratch-transformer-pytorch

python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .          # để `import nmt` chạy được từ mọi nơi

pytest -q                 # phải xanh ngay từ đầu (test tái lập)
```

Trên **Kaggle** thì đã có sẵn PyTorch, chỉ cần:

```bash
pip install -q tokenizers sacrebleu huggingface_hub pyyaml
```

## Chạy thử pipeline

```bash
python scripts/prepare_data.py    --config configs/base.yaml
python scripts/train_tokenizer.py --config configs/base.yaml
python scripts/overfit_sanity.py  --config configs/base.yaml   # cổng chặn Phase 2
python scripts/benchmark_speed.py --config configs/base.yaml
python scripts/train.py           --config configs/base.yaml
python scripts/evaluate.py        --config configs/base.yaml
```

---

## Nguyên tắc xuyên suốt của nhóm

> **Mọi lựa chọn kiến trúc đều phải chứng minh được bằng số liệu thực nghiệm,
> không lấy mặc định từ bài báo làm kết luận.**

Vì vậy **mỗi khối kiến trúc đều được cài hai phương án**, đổi qua lại bằng **đúng một dòng
trong file YAML**, không sửa code:

| Khối | Phương án A (mặc định) | Phương án B (đối chứng) | Khóa cấu hình | Ablation |
|---|---|---|---|---|
| Chuẩn hóa | `rmsnorm` | `layernorm` | `mo_hinh.kieu_chuan_hoa` | **A1** |
| Mã hóa vị trí | `rope` | `sinusoidal` | `mo_hinh.ma_hoa_vi_tri` | **A4** |
| Feed-Forward | `swiglu` | `relu` | `mo_hinh.kieu_ffn` | **A5** |
| Vị trí chuẩn hóa | `pre` | `post` | `mo_hinh.vi_tri_chuan_hoa` | **A6** |
| Warmup | **tắt** | bật | `toi_uu.scheduler` | **A2** |
| Label Smoothing | **0.0** | 0.1 | `toi_uu.label_smoothing` | **A3** |

Warmup và Label Smoothing **mặc định TẮT** (theo nhận xét 1 của mentor). Giữ hay bỏ là do
kết quả TASK 18 quyết định, không phải do bài báo quyết định.

Mỗi thí nghiệm ablation chạy **tối thiểu 2 seed** và báo cáo cả độ lệch giữa các seed.

---

## Phân công — 21 Task / 5 Phase / 4 Tuần

| # | Task | Người | Chain | Phase | Ưu tiên |
|---|---|---|---|---|---|
| 01 | Lập repo, môi trường & thiết lập tái lập | **Phú** | 0 — Khởi động | 1 | Must |
| 02 | Tải & làm sạch dữ liệu IWSLT | **My** | A — Dữ liệu | 1 | Must |
| 03 | Huấn luyện BPE Tokenizer 32k | **My** | A | 1 | Must |
| 04 | Viết Dataset, DataLoader & Mask | **My** | A | 1 | Must |
| 05 | Viết Attention & Multi-Head Attention | **Bảo** | B — Mô hình | 2 | Must |
| 06 | Viết mã hóa vị trí (RoPE + sin-cos) | **Bảo** | B | 2 | Must |
| 07 | Viết chuẩn hóa (RMSNorm + LayerNorm) | **Quân** | B | 2 | Must |
| 08 | Viết Feed-Forward (SwiGLU + ReLU) | **Quân** | B | 2 | Must |
| 09 | Ghép Encoder–Decoder thành mô hình | **Bảo** | B | 2 | Must |
| 10 | Kiểm chứng tính đúng đắn kiến trúc | **My** | A | 2 | Must |
| 11 | Khảo sát cấu hình & ngân sách GPU | **Bảo** | C — Huấn luyện | 3 | Cao |
| 12 | Cơ chế Checkpoint & đồng bộ HF Hub | **Quân** | C | 3 | Must |
| 13 | Viết Training Loop | **Bảo** | C | 3 | Must |
| 14 | Thí nghiệm giết phiên & phục hồi | **Quân** | C | 3 | Must |
| 15 | Huấn luyện chính thức trên IWSLT | **Quân** | C | 3 | Must |
| 16 | Greedy Search & chấm điểm baseline | **My** | D — Đánh giá | 4 | Must |
| 17 | Ablation nhóm kiến trúc (A1 A4 A5 A6) | **Phú** | E — Ablation & Đóng gói | 4 | Cao |
| 18 | Ablation nhóm kỹ thuật huấn luyện (A2 A3) | **Phú** | E | 4 | Cao |
| 19 | Beam Search & KV Cache | **Phú** | E | 4 | **Bonus** |
| 20 | Demo, Docker & Hugging Face Spaces | **Phú** | E | 5 | Must |
| 21 | Báo cáo, phân tích lỗi & Slide | **Phú** | E | 5 | Must |

Chi tiết đầy đủ từng task, tiêu chí "Xong khi" và luồng bàn giao: **[docs/quy_trinh_lam_viec.md](docs/quy_trinh_lam_viec.md)**

## Bốn mốc kiểm soát tiến độ

| Mốc | Điều kiện cần đạt | Nếu không đạt |
|---|---|---|
| **Cuối Tuần 2** | TASK 10 đạt 12/12 bài test + học thuộc 50 câu | Chưa qua thì **chưa** khảo sát cấu hình hay train chính thức |
| **Đầu Tuần 3** | TASK 11 chốt cấu hình cuối bằng số liệu | Đây là căn cứ trả lời mentor về số layer / số head |
| **Cuối Tuần 3** | TASK 15 có mô hình train xong | Cắt TASK 19, thu gọn TASK 17 còn mỗi A1 |
| **Tuần 4** | TASK 17 + 18 đủ 6 thí nghiệm | Ưu tiên A1 (RMSNorm vs LayerNorm) vì mentor nêu trực tiếp |

---

## Cấu hình mô hình dự kiến

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `d_model` | 512 | chuẩn Transformer base |
| số head | 8 (mỗi head 64 chiều) | **TASK 11 khảo sát 4 vs 8** rồi chốt |
| số lớp | 6 enc + 6 dec | **TASK 11 khảo sát 4 vs 6** rồi chốt |
| `d_ff` | **688** (SwiGLU) / 1024 (ReLU) | 1024 × 2/3 ≈ 682.67 → làm tròn 688 cho chia hết 16 |
| dropout | 0.3 | cao hơn mặc định 0.1 vì dữ liệu ít (133k câu) |
| vocab | 32.000 dùng chung En + Vi | |
| chia sẻ embedding | encoder = decoder = lớp xuất | |
| kiểu số thực | **fp16 + GradScaler** | T4 là Turing (CC 7.5), **KHÔNG có bf16** |
| tổng tham số | ~48 triệu | |

### Ba điểm thiết kế bắt buộc nhớ

1. **RoPE chỉ áp cho self-attention**, không áp cho cross-attention — query ở câu Việt, key ở
   câu Anh là hai chuỗi khác nhau nên khoảng cách giữa chúng vô nghĩa. (Bài test 11)
2. **Pre-Norm** là `x + SubLayer(Norm(x))`, không phải `Norm(x + SubLayer(x))`. (Bài test 10)
3. **Bắt buộc có một RMSNorm cuối** ngay trước lớp Linear xuất từ vựng khi dùng Pre-Norm.
   Quên bước này **không làm chương trình báo lỗi**, chỉ khiến giá trị phình dần rồi mô hình
   mất ổn định. (Bài test 10)

### Bốn cái bẫy fp16 — đọc trước khi code

| Chỗ | Phải làm | Nếu sai |
|---|---|---|
| Che mask trong attention | `torch.finfo(scores.dtype).min` | `-1e9` → `-inf` ở fp16 (max 65504) → hàng bị mask toàn bộ ra **NaN** |
| Bảng góc quay RoPE (`inv_freq`, `cos`, `sin`) | tính ở **float32** rồi mới ép về dtype đầu vào | từ vị trí ~100 trở đi mất độ phân giải, **RoPE hỏng âm thầm** |
| Trung bình bình phương trong RMSNorm | tính ở **float32** rồi mới ép về | bình phương vượt 65504 → tràn → NaN cả mô hình |
| GradScaler + grad clipping | gọi `scaler.unscale_(optimizer)` **trước** khi clip norm | đang clip gradient đã bị nhân hệ số giãn, ngưỡng 1.0 vô nghĩa, **không báo lỗi** |

Không dùng `torch.cuda.is_bf16_supported()` để tự chọn dtype — hàm này tính cả trường hợp
giả lập nên trả về `True` ngay trên T4. Chốt cứng `fp16` trong YAML.

---

## Cấu trúc thư mục

```
english-vietnamese-nmt-scratch-transformer-pytorch/
├── configs/            # base.yaml + 6 file ablation A1..A6
├── src/nmt/
│   ├── utils/          # seed, config, logging   ← TASK 01, đã xong
│   ├── data/           # cleaning, tokenizer, dataset, collate
│   ├── model/          # attention, positional, normalization, layers, masking, transformer
│   ├── training/       # trainer, checkpoint, hub_sync, scheduler, loss
│   ├── inference/      # search (greedy, beam, kv-cache)
│   ├── eval/           # metrics (BLEU, chrF++)
│   └── serve/          # ui.py (Streamlit)
├── tests/              # 12 bài kiểm tra kiến trúc + checkpoint + beam + tái lập
├── scripts/            # prepare_data, train_tokenizer, overfit_sanity, benchmark_*, train, evaluate, export_model
├── notebooks/          # 00 kiểm tra dữ liệu, 01 giải thích kiến trúc, 02 kaggle train
├── docker/             # Dockerfile (PyTorch bản CPU, multi-stage)
├── docs/               # báo cáo từng phần + hướng dẫn chạy
├── results/            # bảng số, hình vẽ do script sinh ra (không commit file nặng)
├── data/               # dữ liệu thô + đã xử lý (KHÔNG commit)
└── artifacts/          # tokenizer, checkpoint (KHÔNG commit)
```

---

## Bảo mật — đọc kỹ

- **Tuyệt đối không commit Hugging Face token** vào repo.
- Trên Kaggle: lưu token loại **Write** trong **Add-ons → Secrets**, tên `HF_TOKEN`.
- Trên máy cá nhân: đặt biến môi trường `HF_TOKEN`, hoặc dùng file `.env` (đã trong `.gitignore`).
- `src/nmt/training/hub_sync.py` đọc token theo thứ tự: Kaggle Secrets → biến môi trường → báo lỗi rõ ràng.
- Thư mục `data/` và `artifacts/` đã bị `.gitignore` chặn, đừng dùng `git add -f`.

## Tech stack

PyTorch · tokenizers (HF) · huggingface_hub · sacrebleu · pytest · PyYAML ·
TensorBoard + CSV · Streamlit · Docker + docker compose · ONNX Runtime (mở rộng)

## Dữ liệu

IWSLT 2015 English–Vietnamese — `train` ~133.318 cặp · `tst2012` 1.269 cặp (dev) ·
`tst2013` 1.269 cặp (test, **chỉ dùng chấm điểm cuối**).

PhoMT **không** dùng trong khóa học (quá sức Kaggle), để dành cho hướng phát triển sau.
