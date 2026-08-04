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

> Ngưỡng lọc: `do_dai_toi_da = 100` token (đọc từ `configs/base.yaml`)

| Split | Trước lọc | Sau lọc | Bị loại | Rỗng | Trùng | Quá ngắn | Quá dài |
|---|--:|--:|--:|--:|--:|--:|--:|
| train | 133,317 | 131,400 | 1,917 | 151 | 1006 | 0 | 760 |
| tst2012 | 1,553 | 1,548 | 5 | 0 | 2 | 0 | 3 |
| tst2013 | 1,268 | 1,255 | 13 | 0 | 7 | 0 | 6 |

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
```
Số dòng EN và VI của mỗi split **phải bằng nhau** (script đã tự kiểm tra và báo lỗi nếu lệch).
