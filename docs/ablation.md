# Ablation

> **Phụ trách: TASK 17 + 18 — Phú**
>
> Cập nhật theo góp ý cuối: ngoài sáu ablation đơn biến, phải có một baseline
> Transformer 2017 rõ ràng để chứng minh đóng góp của cả gói cải tiến.

## 1. Câu hỏi nghiên cứu, benchmark và target

### Câu hỏi nghiên cứu

Với cùng dữ liệu và cùng ngân sách huấn luyện, gói kiến trúc
**RoPE + Pre-Norm + RMSNorm + SwiGLU** có cải thiện chất lượng dịch Anh–Việt
so với kiến trúc **Transformer 2017** dùng
**sin-cos + Post-Norm + LayerNorm + ReLU** hay không?

Đây là phép so **tổng hợp**. Bốn ablation A1/A4/A5/A6 ở mục 3 mới dùng để
quy đóng góp cho từng thành phần.

### Benchmark cố định

- Bộ dữ liệu: IWSLT 2015 English→Vietnamese.
- Train: `data/processed/train` (131.339 cặp sau làm sạch và loại rò rỉ).
- Dev: toàn bộ `tst2012` (1.553 cặp), chỉ dùng chọn checkpoint tốt nhất.
- Test: toàn bộ `tst2013` (1.268 cặp), chỉ chấm sau khi đã chốt cấu hình.
- Sinh câu: greedy cho phép so chính; mọi hệ thống phải dùng cùng cấu hình sinh.
- Thước đo chính: tokenized SacreBLEU (`tokenize=none`, vì corpus đã tách
  token); thước đo phụ: chrF++ (`word_order=2`).
- Bắt buộc lưu nguyên chuỗi signature của cả hai metric.

Các con số 1.553/1.268 và cách chia dev/test trùng với thiết lập IWSLT'15
En–Vi được mô tả trong [Jones et al. (2020), mục 4.3](https://aclanthology.org/2020.tacl-1.53/).

### Target đã chốt

1. **Mức hoàn thành kỹ thuật:** BLEU test ≥ **22,0** theo cổng TASK 16 hiện có.
2. **Target đóng góp:** trung bình BLEU của hệ thống cải tiến cao hơn baseline
   vanilla ít nhất **+1,0 BLEU** trên cùng `tst2013`, tính trên hai seed 42 và
   1337. Mốc +1,0 là giả thuyết thực nghiệm được đặt trước khi chạy, gần với
   mức tăng trung bình +1,1 BLEU của các thay đổi chuẩn hóa trên năm cặp ngôn
   ngữ ít tài nguyên trong [Nguyen & Salazar (2019)](https://aclanthology.org/2019.iwslt-1.17/).
3. **Mốc tham chiếu ngoài:** **26,4 tokenized BLEU** của Luong & Manning được
   tổng hợp trên IWSLT'15 En–Vi `tst2013` trong
   [Jones et al. (2020), Bảng 5](https://aclanthology.org/2020.tacl-1.53/).

Mốc 26,4 chỉ là đối chiếu với công bố, **không được tuyên bố so sánh trực tiếp**
nếu signature/tokenization khác. Kết luận chính của đồ án phải dựa trên hai hệ
thống chạy trong chính repo này với cùng protocol.

## 2. Baseline vanilla cùng ngân sách — phần đã chuẩn bị

Hai vế dùng các file:

| Vế so sánh | Cấu hình | Kiến trúc |
|---|---|---|
| Hệ thống cải tiến | `configs/baseline_cai_tien_6000.yaml` | RoPE, Pre-Norm, RMSNorm, SwiGLU |
| Vanilla 2017 khớp quy mô | `configs/baseline_vanilla_architecture_6000.yaml` | sin-cos, Post-Norm, LayerNorm, ReLU |

Transformer gốc dùng residual rồi LayerNorm, FFN ReLU và positional encoding
sin-cos; xem [Vaswani et al. (2017), mục 3.1, 3.3, 3.5](https://arxiv.org/abs/1706.03762).
Baseline trong repo giữ `d_ff=1024` thay vì 2048 để số tham số FFN gần bằng
SwiGLU `d_ff=688`; đây là đối chứng khớp quy mô cho IWSLT, không phải tái lập
nguyên xi kết quả WMT của bài báo.

Ngân sách được đóng băng trước khi chạy:

- đúng **6.000 bước optimizer**;
- mỗi bước gồm 4 micro-batch, mỗi micro-batch tối đa 4.096 token;
- không early stopping trước mốc 6.000;
- cùng tokenizer, train/dev/test, seed, dropout, optimizer và cách chấm;
- chạy hai seed **42** và **1337**.

Lệnh huấn luyện bốn lượt:

```bash
python scripts/train.py --config configs/baseline_cai_tien_6000.yaml --seed 42 --tu-dau
python scripts/train.py --config configs/baseline_cai_tien_6000.yaml --seed 1337 --tu-dau
python scripts/train.py --config configs/baseline_vanilla_architecture_6000.yaml --seed 42 --tu-dau
python scripts/train.py --config configs/baseline_vanilla_architecture_6000.yaml --seed 1337 --tu-dau
```

Khi đánh giá phải dùng `tot_nhat.pt`. `moi_nhat.pt` là checkpoint để resume,
không mặc định là checkpoint có loss dev tốt nhất. Checkpoint seed 42 hiện tại
nằm tại [Hugging Face Hub](https://huggingface.co/mgbao/envi-nmt-scratch-transformer),
nhưng chưa đủ thay cho bốn lượt cố định phía trên.

### Bảng kết quả tổng hợp — chưa chạy

| Hệ thống | Seed | Bước | loss dev | BLEU | chrF++ | Signature |
|---|---:|---:|---:|---:|---:|---|
| Cải tiến | 42 | 6.000 | 2,2364 | **29,74** | **48,73** | `nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.6.0` |
| Cải tiến | 1337 | 6.000 | — | — | — | — |
| Vanilla 2017 | 42 | 6.000 | — | — | — | — |
| Vanilla 2017 | 1337 | 6.000 | — | — | — | — |

Không điền số vào bảng cho tới khi có checkpoint thật và file kết quả do
`scripts/evaluate.py` sinh ra.

## 3. TASK 17 — ablation kiến trúc chưa chạy

Mỗi ablation dưới đây vẫn chỉ đổi đúng một yếu tố so với baseline cải tiến
6.000 bước. Tất cả file đã kế thừa `baseline_cai_tien_6000.yaml`, nên không còn
nguy cơ A1..A6 chạy nhầm ngân sách 60.000 bước/early stopping từ `base.yaml`:

| Mã | So sánh | File đối chứng | Ưu tiên |
|---|---|---|---|
| A1 | RMSNorm vs LayerNorm | `configs/ablation_a1_layernorm.yaml` | cao nhất, mentor hỏi trực tiếp |
| A4 | RoPE vs sin-cos | `configs/ablation_a4_sincos.yaml` | sau A1 |
| A5 | SwiGLU vs ReLU | `configs/ablation_a5_relu.yaml` | sau A4 |
| A6 | Pre-Norm vs Post-Norm | `configs/ablation_a6_post_norm.yaml` | sau A5 |

Mỗi thí nghiệm: tối thiểu 2 seed, cùng ngân sách bước, báo cáo loss dev, BLEU,
chrF++ và độ lệch giữa các seed. A5 đổi thêm `d_ff` là ngoại lệ có chủ ý để
giữ số tham số FFN chênh dưới 1%.

A6 có thể phân kỳ khi giữ scheduler cố định. Nếu xảy ra thì ghi là kết quả,
không âm thầm bật warmup rồi làm mất tính đơn biến.

## 4. TASK 18 — ablation kỹ thuật huấn luyện chưa chạy

- A2: Warmup tắt vs bật (`configs/ablation_a2_warmup.yaml`).
- A3: Label Smoothing 0,0 vs 0,1 (`configs/ablation_a3_label_smoothing.yaml`).
- Cần vẽ chồng đường loss để so tốc độ hội tụ và dùng kết quả quyết định giữ
  hay bỏ hai kỹ thuật khỏi cấu hình cuối.

A2 có thể chênh rất nhỏ vì Pre-Norm làm giảm nhu cầu warmup; đây là giả thuyết
có cơ sở từ [Nguyen & Salazar (2019)](https://aclanthology.org/2019.iwslt-1.17/),
không mặc định là bug.

## 5. Trạng thái thực tế

Đã làm:

- khóa đúng 6.000 bước, tắt early stopping sớm cho cả baseline và A1..A6;
- sinh manifest đủ 16 lượt (8 cấu hình × 2 seed) bằng
  `python scripts/run_ablations.py`;
- có checkpoint thật cải tiến seed 42 và điểm full tst2013 29,74 BLEU;
- cài/kiểm thử Beam Search + KV cache; Beam-4 đạt 30,69 BLEU, tăng +0,95;
- hoàn thiện demo, Docker và phân tích lỗi 30 câu.

Chưa có số liệu:

- cải tiến seed 1337;
- vanilla seed 42/1337;
- A1..A6 seed 42/1337.

Đây là 15 lượt huấn luyện 48 triệu tham số còn thiếu. Máy hiện tại dùng
`torch+cpu`, không có CUDA; script chủ động từ chối `--execute` trên CPU để
không biến nhiều ngày chạy thành một kết quả không khả thi. Chạy trên Kaggle T4:

```bash
python scripts/run_ablations.py --execute --repo-hub <tai-khoan/repo>
```

Sau khi tải điểm về, chạy `python scripts/summarize_ablations.py` để sinh bảng
mean/std; ô chưa đo luôn để trống thay vì điền số giả.
