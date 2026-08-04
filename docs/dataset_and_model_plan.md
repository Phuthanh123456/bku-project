# Dataset and Model Plan — ENVI-NMT

> Người viết: My (Data Engineer, Chain A)
> Trạng thái: 🟢 **TASK 02 + TASK 03 + TASK 04 đã hoàn tất** — số liệu, tokenizer, Dataset/DataLoader/Mask
> đều đã kiểm tra và pass toàn bộ. Tài liệu phản ánh đúng kết quả thực tế tính đến 2026-08-04.
> Mục đích: tài liệu tham chiếu chung cho cả nhóm về dữ liệu và kế hoạch mô hình,
> tránh mỗi người hiểu một kiểu khi làm các Chain khác nhau.

---

## 1. Tổng quan bộ dữ liệu

**Tên**: IWSLT 2015 English–Vietnamese
**Nguồn gốc**: các bài nói chuyện TED Talks, phụ đề song ngữ Anh–Việt
**Nguồn tải**: Stanford NLP Group (`nlp.stanford.edu/projects/nmt/`), mirror ổn định qua
GitHub `stefan-it/nmt-en-vi`
**Lý do chọn**: dữ liệu vừa đủ nhỏ để train trên GPU Kaggle (T4) trong thời gian khoá học,
nhưng vẫn đủ khó để phản ánh các vấn đề thật của NMT (câu dài, văn nói, tên riêng...)

| Split | Vai trò | Số cặp câu gốc (trước lọc) | Số cặp câu sau lọc (dùng thực tế) | Ghi chú |
|---|---|---|---|---|
| `train` | Huấn luyện | 133.317 | **131.400** | |
| `tst2012` | Dev (theo dõi trong lúc train, chọn checkpoint) | 1.553 | **1.548** | |
| `tst2013` | Test (chỉ dùng chấm điểm cuối cùng, KHÔNG dùng để tune) | 1.268 | **1.255** | |

Chi tiết quá trình lọc và lý do loại từng split xem mục 2.2 bên dưới.

⚠️ **Quy tắc bắt buộc cho cả nhóm**: `tst2013` chỉ được chạm tới đúng 1 lần ở TASK 16
(chấm baseline) và khi báo cáo kết quả cuối. Không dùng `tst2013` để so sánh ablation
hay chọn hyperparameter — việc đó dùng `tst2012`. Vi phạm quy tắc này làm số BLEU
cuối cùng mất ý nghĩa (data leakage vào tập test).

**Không dùng PhoMT** trong khoá học này (README đã ghi rõ — dữ liệu quá lớn so với
ngân sách GPU Kaggle), để dành cho hướng phát triển sau nếu nhóm muốn mở rộng.

---

## 2. Quy trình xử lý dữ liệu (TASK 02)

### 2.1. Pipeline làm sạch

Đã cài đặt trong `scripts/prepare_data.py`, chạy với `configs/base.yaml`
(tham số thực tế đã dùng lưu tại `results/config_da_dung.yaml`).

| Bước | Hàm / đoạn code chính | Số cặp bị loại (train / tst2012 / tst2013) | Trạng thái |
|---|---|---|---|
| Strip + giải mã HTML entity | `html.unescape(cau)` | — (bước biến đổi) | ✅ |
| Xoá tag HTML còn sót | `re.sub(r'<[^>]+>', ' ', cau)` | — (bước biến đổi) | ✅ |
| Chuẩn hoá Unicode NFC | `unicodedata.normalize('NFC', cau)` | — (bước biến đổi) | ✅ |
| Chuẩn hoá khoảng trắng | `re.sub(r'\s+', ' ', cau).strip()` | — (bước biến đổi) | ✅ |
| Bỏ câu rỗng | `if not en or not vi` | 151 / 0 / 0 | ✅ |
| Bỏ cặp trùng lặp | `if (en, vi) in da_gap` | 1.006 / 2 / 7 | ✅ |
| Lọc câu quá ngắn | `if n_en < 1 or n_vi < 1` | 0 / 0 / 0 | ✅ |
| Lọc câu quá dài | `if n_en > do_dai_toi_da or n_vi > do_dai_toi_da` (ngưỡng = 100 token, từ `configs/base.yaml`) | 760 / 3 / 6 | ✅ |

> Tổng cộng bị loại: train **1.917** / tst2012 **5** / tst2013 **13**. Xem `results/thong_ke_loc.csv` để tra cứu lại.

### 2.2. Thống kê trước/sau khi làm sạch

✅ Đã kiểm tra bất biến: **EN = VI về số dòng cho mỗi split** sau khi lọc (không bị
lệch hàng giữa 2 file song ngữ) — đây là điều kiện bắt buộc, sai điều kiện này thì
toàn bộ dataset vô nghĩa.

| Split | Số cặp câu gốc | Số cặp câu sau khi lọc | Số cặp bị loại | % bị loại | Rỗng | Trùng | Quá dài |
|---|--:|--:|--:|--:|--:|--:|--:|
| `train` | 133,317 | **131,400** | 1,917 | ~1.44% | 151 | 1,006 | 760 |
| `tst2012` (dev) | 1,553 | **1,548** | 5 | ~0.32% | 0 | 2 | 3 |
| `tst2013` (test) | 1,268 | **1,255** | 13 | ~1.03% | 0 | 7 | 6 |

Bảng chi tiết đầy đủ (cấp câu, không chỉ tổng hợp) lưu tại `results/thong_ke_loc.csv`.
Báo cáo mô tả quy trình lọc đầy đủ lưu tại `docs/bao_cao_du_lieu.md`.

### 2.3. Vị trí file sau xử lý

```
data/
├── raw/                        # file .tgz và file .en/.vi gốc, chưa lọc (KHÔNG commit)
└── processed/                  # dữ liệu đã làm sạch, dùng cho tokenizer & training (KHÔNG commit)
    ├── train.en, train.vi           # 131.400 cặp câu huấn luyện
    ├── tst2012.en, tst2012.vi       # 1.548 cặp câu dev
    └── tst2013.en, tst2013.vi       # 1.255 cặp câu test
```

> Cập nhật so với bản kế hoạch ban đầu: file được đặt **phẳng trực tiếp trong
> `data/processed/`** (`train.en`, `train.vi`...), không tách thư mục con theo từng
> split như dự kiến ban đầu. Các script tiếp theo (tokenizer TASK 03, Dataset/DataLoader
> TASK 04) cần đọc đúng path này qua `configs/base.yaml`, không hardcode.

Đường dẫn cụ thể lấy từ khoá `du_lieu.*` trong `configs/base.yaml` — mọi script khác
(tokenizer, dataset, train) đều phải đọc từ config, không hardcode path.

**Các artifact khác sinh ra từ TASK 02** (dùng để tham chiếu/kiểm tra lại nếu cần):

| File | Nội dung |
|---|---|
| `results/thong_ke_loc.csv` | Thống kê lọc chi tiết theo từng split |
| `docs/bao_cao_du_lieu.md` | Báo cáo mô tả đầy đủ quy trình làm sạch dữ liệu |
| `results/config_da_dung.yaml` | Bản chụp lại đúng config đã dùng khi chạy `prepare_data.py`, phục vụ tái lập |

---

## 3. Tokenizer (TASK 03) — ✅ Hoàn tất

### 3.1. Cấu hình

Script: `scripts/train_tokenizer.py`, chạy với `configs/base.yaml`.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Loại | BPE | `tokenizers.models.BPE` của Hugging Face |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False)` | Tách theo byte — không mất thông tin Unicode tiếng Việt |
| Decoder | `ByteLevel()` | Tương ứng với pre-tokenizer để decode không lỗi |
| Vocab size | **32,000** (đúng như config) | Đọc từ `du_lieu.vocab_size` trong `configs/base.yaml` |
| `min_frequency` | 2 | Subword phải xuất hiện ≥ 2 lần mới vào vocab |
| Corpus huấn luyện | `train.en` + `train.vi` = **262,800 dòng** | KHÔNG dùng dev/test — tránh leakage |
| Chia sẻ embedding | Có — encoder = decoder = lớp xuất vocab | 1 tokenizer duy nhất cho cả 2 ngôn ngữ |

### 3.2. Special tokens (thứ tự cố định — KHÔNG đổi)

| ID | Token | Vai trò |
|--:|---|---|
| 0 | `<pad>` | Padding — attention mask = 0 tại vị trí này |
| 1 | `<unk>` | Unknown — không nên xuất hiện sau BPE 32k |
| 2 | `<bos>` | Beginning of sequence — input đầu tiên của decoder |
| 3 | `<eos>` | End of sequence — dùng để dừng sinh câu |

> ⚠️ Thứ tự 4 token đặc biệt này phải nhất quán với TASK 04 (Dataset/mask), TASK 05–09 (Model), TASK 11 (test kiến trúc). Không đổi thứ tự.

### 3.3. Thống kê thực tế (chạy xong lúc 2026-08-04, seed=42)

| Metric | Giá trị | Nhận xét |
|---|---|---|
| Vocab size thực tế | **32,000** | Khớp đúng mục tiêu |
| Fertility EN (train, mẫu 5k câu) | **1.0665** subword/từ | Rất tốt — gần 1.0, nghĩa là hầu hết từ tiếng Anh là 1 token |
| Fertility VI (train, mẫu 5k câu) | **1.0137** subword/từ | Xuất sắc — tiếng Việt được tokenize hiệu quả hơn EN |
| UNK rate (train) | **0.0000%** | Không có token lạ — vocab 32k đủ rộng |
| UNK rate (dev / tst2012) | **0.0000%** | Không có OOV trên tập dev |
| Thời gian train | ~7 giây | Trên MacBook, corpus 262,800 dòng |

> **Nhận xét Fertility**: Giá trị 1.01–1.07 là lý tưởng (kỳ vọng cho BPE tốt: 1.0–1.5).
> Con số này có nghĩa là với vocab 32k, gần như mọi từ thông dụng En/Vi đều là 1 token
> đơn — không bị phân mảnh. Tokenizer BPE ByteLevel xử lý tiếng Việt tốt mà không
> cần bước tiền xử lý tách từ riêng.

### 3.4. Kiểm tra encode/decode (vòng lặp đóng)

```
Input EN : "The quick brown fox jumps over the lazy dog."
Tokens   : ['The', 'Ġquick', 'Ġbrown', 'Ġfo', 'x', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog', '.']
Decoded  : "The quick brown fox jumps over the lazy dog."   ← khớp 100%

Input VI : "Xin chào, tôi đang học xây dựng mô hình dịch máy."
Tokens   : ['Xin', 'ĠchÃło', ',', 'ĠtÃ´i', 'ĠÄĳang', 'Ġhá»įc', ...]  (ByteLevel encoding)
Decoded  : "Xin chào, tôi đang học xây dựng mô hình dịch máy."   ← khớp 100%
```

> `Ġ` là ký hiệu ByteLevel cho khoảng trắng đứng trước. Token trông lạ mắt nhưng
> decode hoàn toàn chính xác — đây là cách ByteLevel BPE biểu diễn, không phải lỗi.

### 3.5. Đầu ra sinh ra

| File | Kích thước | Dùng để |
|---|---|---|
| `artifacts/tokenizer/tokenizer.json` | 2,288,456 bytes (~2.2 MB) | Load bằng `Tokenizer.from_file(...)` trong TASK 04, 05, 16 |
| `artifacts/tokenizer/vocab.json` | 517,520 bytes | Tra cứu token → id |
| `artifacts/tokenizer/merges.txt` | 304,038 bytes | BPE merge rules — debug |
| `results/thong_ke_tokenizer.csv` | — | Fertility, UNK rate, top-30 token phổ biến |
| `results/config_tokenizer.yaml` | — | Bản chụp config đã dùng, phục vụ tái lập |

```python
# Cách load tokenizer trong các script tiếp theo (TASK 04, 05, 16):
from tokenizers import Tokenizer
tok = Tokenizer.from_file("artifacts/tokenizer/tokenizer.json")
ids = tok.encode("Hello world").ids    # → list[int]
text = tok.decode(ids)                 # → "Hello world"
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
```

---

## 4. Dataset, DataLoader & Mask (TASK 04) — ✅ Hoàn tất

### 4.1. Cấu trúc module

| File | Nội dung |
|---|---|
| `src/nmt/data/dataset.py` | `DuLieuSongNgu(Dataset)` — đọc 2 file text, encode, trả dict |
| `src/nmt/data/collate.py` | `GomBatchTheoTokenSampler`, `ghep_batch`, `tao_dataloader`, `ti_le_o_dem` |
| `src/nmt/data/tokenizer.py` | `nap_tokenizer`, `kiem_tra_ma_hoa_giai_ma`, `ti_le_token_la` |
| `src/nmt/data/__init__.py` | Export toàn bộ API công khai |

### 4.2. Dataset — `DuLieuSongNgu`

Mỗi mẫu `ds[i]` trả về:

```python
{
  "src_ids": Tensor[int64, shape=(S,)],   # EN token IDs, KHÔNG BOS/EOS
  "tgt_ids": Tensor[int64, shape=(T+2,)], # VI token IDs, CÓ BOS đầu + EOS cuối
}
```

- Encode **toàn bộ corpus khi khởi tạo** (eager) — tránh overhead mỗi lần `__getitem__`
- Câu vượt `do_dai_toi_da=100` bị **truncate** (không bỏ) — TASK 02 đã lọc bằng từ, đây lọc bằng subword
- Kiểm tra bất biến `len(EN) == len(VI)` ngay khi `__init__`, raise `RuntimeError` nếu lệch

### 4.3. Token-Bucket Batching — `GomBatchTheoTokenSampler`

Gom batch theo **tổng token chứ không theo số câu**:

```
max_len_trong_batch × so_cau ≤ so_token_moi_batch (4096)
```

Thuật toán:
1. Sắp xếp câu theo độ dài src (`gom_theo_do_dai=True`)
2. Gom tuần tự: khi `max_len × n_câu > 4096` → đóng batch, mở batch mới
3. Xáo trộn **thứ tự batch** (không xáo câu trong batch) mỗi epoch

Kết quả đo được:
| Metric | Giá trị |
|---|--:|
| Số batch trên train (131,400 câu, 4096 token/batch) | **689** |
| Số câu trung bình mỗi batch | **190.7** |
| Số câu min/max | 37 / 1,024 |
| **Tỉ lệ padding trung bình** (50 batch giữa corpus) | **21.77%** ✅ < 25% |

### 4.4. Mask — `ghep_batch` (collate_fn)

`ghep_batch` trả về dict đầy đủ cho Transformer:

| Key | Shape | Nội dung |
|---|---|---|
| `src_ids` | `[B, S]` | Token IDs nguồn, đã padding |
| `tgt_input` | `[B, T]` | Decoder input = `tgt[:-1]` — có BOS, không EOS |
| `labels` | `[B, T]` | Nhãn = `tgt[1:]` — không BOS, có EOS (teacher forcing) |
| `src_mask` | `[B, 1, 1, S]` | `True` tại vị trí THẬT, `False` tại PAD |
| `tgt_mask` | `[B, 1, T, T]` | **Causal mask AND padding mask** — tam giác dưới, PAD bị che |

Convention mask: `True = được nhìn`, `False = bị che` (dùng `score.masked_fill(~mask, -inf)`).

```
tgt_mask[b, 0] =
  [ T  F  F  F ]   ← vị trí 0 chỉ nhìn được chính nó
  [ T  T  F  F ]   ← vị trí 1 nhìn được 0 và 1
  [ T  T  T  F ]
  [ F  F  F  F ]   ← vị trí này là PAD, toàn False
```

### 4.5. Kết quả kiểm tra (2026-08-04)

| Kiểm tra | Kết quả |
|---|---|
| Special token IDs PAD=0, UNK=1, BOS=2, EOS=3 | ✅ PASS |
| `tgt_ids[0] == BOS` và `tgt_ids[-1] == EOS` | ✅ PASS |
| Teacher forcing: cột đầu `tgt_input` là BOS | ✅ PASS |
| Causal mask: không nhìn về phía sau (vị trí `t` không nhìn `t' > t`) | ✅ PASS |
| Padding mask: PAD bị che trong `src_mask` | ✅ PASS |
| Padding TB < 25% (đo 50 batch giữa corpus) | ✅ **21.77% PASS** |
| Keys batch đủ 5 key | ✅ PASS |

### 4.6. Cách dùng trong các script tiếp theo

```python
from nmt.data import nap_tokenizer, DuLieuSongNgu, tao_dataloader
from nmt.utils import sinh_generator

tok = nap_tokenizer("artifacts/tokenizer/tokenizer.json")
ds  = DuLieuSongNgu("data/processed/train.en", "data/processed/train.vi",
                     tok, do_dai_toi_da=100)
gen    = sinh_generator(seed=42)
loader = tao_dataloader(ds, so_token_moi_batch=4096,
                        gom_theo_do_dai=True, so_worker=2, generator=gen)
for batch in loader:
    src    = batch["src_ids"]    # [B, S]
    inp    = batch["tgt_input"]  # [B, T]
    labels = batch["labels"]     # [B, T]
    smask  = batch["src_mask"]   # [B, 1, 1, S]
    tmask  = batch["tgt_mask"]   # [B, 1, T, T]
```

---

## 5. Kế hoạch kiến trúc mô hình (tham chiếu Chain B, để nhóm cùng nắm)

> Phần này do Bảo/Quân trực tiếp cài đặt (TASK 05–09), My ghi lại ở đây để tài liệu
> đầy đủ một chỗ cho cả nhóm tra cứu, không cần copy lại toàn bộ chi tiết kỹ thuật.

| Tham số | Giá trị dự kiến | Ghi chú |
|---|---|---|
| `d_model` | 512 | chuẩn Transformer base |
| Số head | 8 (mỗi head 64 chiều) | TASK 11 khảo sát 4 vs 8 rồi chốt |
| Số lớp | 6 encoder + 6 decoder | TASK 11 khảo sát 4 vs 6 rồi chốt |
| `d_ff` | 688 (SwiGLU) / 1024 (ReLU) | |
| Dropout | 0.3 | cao hơn mặc định 0.1 vì dữ liệu ít (133k câu) |
| Kiểu số thực | fp16 + GradScaler | T4 không có bf16 |
| Tổng tham số | ~48 triệu | |

**Các khối kiến trúc có 2 phương án đổi qua config** (chi tiết xem README, mục
"Nguyên tắc xuyên suốt của nhóm"): chuẩn hoá (RMSNorm/LayerNorm), mã hoá vị trí
(RoPE/sinusoidal), FFN (SwiGLU/ReLU), vị trí chuẩn hoá (Pre/Post-Norm), Warmup
(tắt/bật), Label Smoothing (0.0/0.1). Quyết định giữ phương án nào dựa trên kết quả
ablation TASK 17–18, không dựa theo mặc định của bài báo gốc.

---

## 5. Rủi ro / vấn đề cần lưu ý khi dùng bộ dữ liệu này

- **Văn nói, không phải văn viết**: Dữ liệu TED Talks có nhiều câu ngắn, ngắt câu
  theo nhịp nói, dùng đại từ rút gọn — có thể khiến mô hình dịch tốt văn nói nhưng
  kém hơn với văn bản học thuật. Cần lưu ý khi đánh giá lỗi dịch ở TASK 21.
- **Kích thước nhỏ (~131k câu train)**: Dropout 0.3 (cao hơn mặc định 0.1) đã được
  cài sẵn trong `configs/base.yaml` để bù đắp. Warmup và Label Smoothing chưa bật mặc
  định — quyết định sau kết quả ablation TASK 17–18.
- **Bất biến EN = VI**: Script `prepare_data.py` tự kiểm tra sau mỗi split bằng cách
  đếm dòng 2 file và `raise RuntimeError` nếu lệch. Đã xác nhận thủ công bằng `wc -l`:
  train 131,400 / tst2012 1,548 / tst2013 1,255 — cả 3 split đều khớp 100%.
- **Tỉ lệ câu trùng lặp trong train cao nhất**: 1,006 cặp trùng (0.75% tổng train) —
  đây là con số bình thường với dữ liệu TED (nhiều bài nói dùng lại câu giới thiệu
  giống nhau). Đã bỏ toàn bộ trùng lặp theo cặp (en, vi).
- **HTML entity và tag sót**: Một số câu có `&amp;`, `&lt;`, `<i>`, `<br />` sót từ
  quá trình xuất phụ đề — đã xử lý trong `_lam_sach_cau()` bằng `html.unescape()` và
  regex loại tag. Kiểm tra thủ công 20 dòng ngẫu nhiên sau lọc không còn thấy artifact.
- **Unicode tiếng Việt**: Một số file có ký tự tổ hợp (combining characters) thay vì
  precomposed — đã chuẩn hoá NFC bằng `unicodedata.normalize('NFC', cau)`. Nếu bỏ
  bước này, tokenizer BPE sẽ tạo ra subword khác nhau cho cùng một từ.

---

## 6. Tham khảo

- Stanford NLP Group — Neural Machine Translation project page
- Luong, Minh-Thang & Manning, Christopher D. — nguồn gốc bộ IWSLT En-Vi dùng trong
  nghiên cứu NMT
- `stefan-it/nmt-en-vi` (GitHub) — mirror tải dữ liệu dùng trong `scripts/prepare_data.py`

---

*Cập nhật lần cuối: **2026-08-04** bởi My — sau khi hoàn tất TASK 02 + TASK 03 + TASK 04 và xác nhận toàn bộ số liệu từ `results/thong_ke_loc.csv` (dữ liệu), `results/thong_ke_tokenizer.csv` (tokenizer), và kết quả smoke test Dataset/DataLoader/Mask. Mọi thay đổi về ngưỡng lọc / pipeline cần cập nhật song song tài liệu này với code.*
