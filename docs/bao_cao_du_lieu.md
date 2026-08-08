# Báo cáo dữ liệu — TASK 02

**Người làm:** My  |  **Seed:** 42

## Nguồn

IWSLT 2015 English–Vietnamese, mirror: `https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data`

| File tgz | URL |
|---|---|
| `train-en-vi.tgz` | https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/train-en-vi.tgz |
| `dev-2012-en-vi.tgz` | https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/dev-2012-en-vi.tgz |
| `test-2013-en-vi.tgz` | https://raw.githubusercontent.com/stefan-it/nmt-en-vi/master/data/test-2013-en-vi.tgz |

## Cấu trúc thư mục sau khi chạy

```
data/
├── raw/
│   ├── train.en  train.vi
│   ├── tst2012.en  tst2012.vi
│   └── tst2013.en  tst2013.vi
└── processed/
    ├── train.en  train.vi
    ├── tst2012.en  tst2012.vi
    └── tst2013.en  tst2013.vi
```

## Thống kê lọc

> Ngưỡng lọc: `do_dai_toi_da = 100` token, `ti_le_lech_toi_da = 3.0` (đọc từ `configs/base.yaml`)  
> Tập dev/test **không bị lọc** — chỉ chuẩn hóa NFC để giữ nguyên số câu.

| Split | Trước lọc | Sau lọc | Bị loại | Rỗng | Trùng | Quá ngắn | Quá dài | Lệch độ dài | Rò rỉ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| train | 133,317 | 131,339 | 1,978 | 151 | 1006 | 0 | 760 | 40 | 21 |
| tst2012 | 1,553 | 1,553 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| tst2013 | 1,268 | 1,268 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Kiểm tra rò rỉ test → train

- **tst2012**: 12 cặp câu trùng nguyên si → đã xóa khỏi train
- **tst2013**: 12 cặp câu trùng nguyên si → đã xóa khỏi train

**Tổng số câu đã xóa khỏi train:** 24 cặp

> Câu bị xóa khỏi **train** (không xóa khỏi dev/test) để tránh rò rỉ làm đội điểm BLEU.

## Xác nhận special tokens

| ID | Token | Ghi chú |
|---|---|---|
| 0 | `<pad>` | Padding |
| 1 | `<unk>` | Unknown |
| 2 | `<bos>` | Beginning of sequence (decoder input) |
| 3 | `<eos>` | End of sequence (decoder output / stop signal) |
| 4 | `<2en>` | Language token English (dành sẵn cho TASK 03) |
| 5 | `<2vi>` | Language token Vietnamese (dành sẵn cho TASK 03) |

> Chốt dùng `<bos>` / `<eos>` (không dùng `<s>` / `</s>`).
> Quân biết khi làm TASK 09.

## Lưu ý tái lập

Chạy lại từ đầu bằng:
```bash
python scripts/prepare_data.py --config configs/base.yaml
```

Sau khi chạy xong, kiểm tra bất biến quan trọng:
```bash
wc -l data/processed/train.en data/processed/train.vi
wc -l data/processed/tst2012.en data/processed/tst2012.vi
wc -l data/processed/tst2013.en data/processed/tst2013.vi
# tst2013.en phải ra 1268, tst2012.en phải ra 1553
```
Số dòng EN và VI của mỗi split **phải bằng nhau** (script đã tự kiểm tra và báo lỗi nếu lệch).
