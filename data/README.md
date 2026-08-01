# Thư mục dữ liệu

Nội dung thư mục này **không được commit** (xem `.gitignore`).

```
data/
├── raw/            # tải về từ nguồn, giữ nguyên không sửa
│   ├── train.en  train.vi
│   ├── tst2012.en  tst2012.vi
│   └── tst2013.en  tst2013.vi
└── processed/      # do scripts/prepare_data.py sinh ra
    ├── train.en  train.vi
    ├── tst2012.en  tst2012.vi
    └── tst2013.en  tst2013.vi
```

## Nguồn

IWSLT 2015 English-Vietnamese, khoảng 133.318 cặp câu train.

| Tập | Số cặp | Dùng làm gì |
|---|---|---|
| train | ~133.318 | huấn luyện |
| tst2012 | 1.269 | dev — theo dõi trong lúc train |
| tst2013 | 1.269 | test — **chỉ dùng chấm điểm cuối cùng** |

**Ghi lại nguồn tải và số dòng của từng file** vào `docs/bao_cao_du_lieu.md`
để sau này người khác tái lập được.

**Không đụng PhoMT** trong khóa học (nhận xét 1 của anh Huy: 3 triệu câu là quá
sức với Kaggle). PhoMT để dành cho hướng phát triển sau khóa học.

Sau khi chạy `scripts/prepare_data.py`, kiểm ngay: với mỗi tập thì số dòng file
tiếng Anh phải bằng **đúng** số dòng file tiếng Việt. Lệch một dòng là toàn bộ
dữ liệu bị lệch cặp và mô hình học rác.
