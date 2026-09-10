# Kết quả ablation

Cập nhật 10/09/2026. Số lấy từ `ablation/ket_qua.csv` trên Hugging Face
(`mgbao/envi-nmt-scratch-transformer`), lượt chạy trên Kaggle Tesla T4.

Mọi lượt dùng **cùng ngân sách 3.000 bước**, mỗi cấu hình **2 seed** (42 và 1337),
chỉ đổi **đúng một yếu tố**. Chấm bằng sacrebleu, sinh câu greedy.

Chữ ký để người khác đối chiếu được:

```
BLEU   nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF++ nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0
```

## Bảng kết quả

| Cấu hình | seed | loss dev | BLEU dev | BLEU test | chrF++ test | giây/bước |
|---|---|---|---|---|---|---|
| Đối chứng (RMSNorm) | 42 | 2,2767 | 25,41 | **28,74** | 47,62 | 1,514 |
| Đối chứng (RMSNorm) | 1337 | 2,2959 | 25,45 | **28,37** | 47,50 | 1,508 |
| A1 (LayerNorm) | 42 | 2,2779 | 25,88 | **28,87** | 47,71 | 1,596 |
| A1 (LayerNorm) | 1337 | 2,2882 | 25,60 | **28,32** | 47,33 | 1,600 |
| A0 (vanilla 2017) | 42 | 4,7695 | 3,59 | 3,64 | 19,77 | 1,472 |
| A0 (vanilla 2017) | 1337 | 4,7677 | 3,62 | 3,61 | 19,71 | 1,469 |

> **Hai hàng A0 KHÔNG dùng được.** Xem mục cuối. Đang chạy lại.

## A1 — RMSNorm hay LayerNorm?

Đây là câu mentor hỏi trực tiếp.

|  | BLEU tst2013 | giây/bước |
|---|---|---|
| RMSNorm | 28,56 ± 0,18 | **1,511** |
| LayerNorm | 28,60 ± 0,27 | 1,598 |

Chênh lệch trung bình là **0,04 BLEU**. Con số đó **nhỏ hơn dao động giữa hai
seed của chính từng nhóm** (0,18 và 0,27). Nói cách khác, nếu chỉ chạy một seed
thì tùy may rủi mà bên nào cũng có thể "thắng" — đó là lý do phải chạy 2 seed
và báo cáo độ lệch, chứ không phải chạy một lượt rồi kết luận.

**Kết luận: về chất lượng dịch, hai cách chuẩn hóa không phân biệt được.**
Nhưng RMSNorm **nhanh hơn 5,7%** (1,511 so với 1,598 giây/bước), đúng như kỳ
vọng vì nó bỏ hẳn phép trừ trung bình.

Vậy lý do giữ RMSNorm là **tốc độ, không phải điểm số**. Đây là số đo của nhóm
trên chính bài toán này, không phải trích kết luận của Zhang & Sennrich (2019).

## So với các công bố IWSLT 2015

Hai bài trong `2015.iwslt-evaluation.12.pdf` và `2015.iwslt-evaluation.15.pdf`:

| Hệ | Anh→Việt | Ghi chú |
|---|---|---|
| Moses (Tran Hong Viet và cs.) | 23,15 | thêm 1GB tin tức tiếng Việt |
| Hiero (Trieu Hai Long và cs.) | 21,48 | |
| Phrase-based in-domain (Trieu) | 26,57 | |
| baseline của workshop | 27,01 | |
| **Phrase-based out-of-domain (Trieu)** | **28,17** | thêm dữ liệu ngoài miền |
| **Của nhóm** | **28,74** | **chỉ dữ liệu trong ràng buộc** |

**Không được kết luận "đã tốt hơn".** Ba lý do:

1. **Khác tập test.** Họ chấm trên tst2015 (1.046 câu), nhóm chấm trên tst2013
   (1.268 câu). Hai bộ câu khác nhau.
2. **Khác cách tính BLEU.** Bài 12 ghi rõ họ tách từ tiếng Việt bằng VnTokenizer
   rồi mới chấm. Nhóm chấm sacrebleu trên văn bản chưa tách. Hai thang đo khác
   nhau — đúng thứ Post (2018) viết cả bài báo để cảnh báo, và bài đó đang là
   tài liệu tham khảo [9] của nhóm.
3. **Khác điều kiện dữ liệu**, và chỗ này có lợi cho nhóm: cả hai bài đều dùng
   dữ liệu ngoài (1GB tin tức crawl, Europarl, News Commentary). Nhóm chỉ dùng
   131.339 cặp câu trong ràng buộc.

Câu nói được một cách trung thực:

> Transformer tự viết, chỉ dùng dữ liệu trong ràng buộc, ở 17% ngân sách bước
> của lượt huấn luyện chính, đạt 28,74 BLEU (sacrebleu, tst2013) — **ngang tầm
> hệ tốt nhất của chiến dịch IWSLT 2015**, vốn có thêm dữ liệu ngoài.

Muốn nói thẳng "tốt hơn" thì phải chấm trên đúng tst2015 theo đúng cách họ chấm.

Cũng nên nói rõ: cả hai bài đều là **SMT dựa trên cụm từ**, thế hệ trước nơ-ron.
Transformer thắng SMT là chuyện đã biết từ 2017. Điểm đáng nói không phải "hơn
họ", mà là **đạt mức đó với ít dữ liệu hơn và kiến trúc tự viết từ đầu**.

## Một quan sát ngoài dự tính

Đối chứng ở **3.000 bước** đạt loss dev **2,2767**. Lượt huấn luyện chính ở
**17.000 bước** chỉ đạt **2,3866**.

Chạy dài hơn 5,7 lần mà **tệ hơn**. Nguyên nhân là quá khớp: lượt chính có loss
train 0,998 so với loss dev 2,387, và cơ chế dừng sớm đã kích hoạt. Với 131k cặp
câu thì 3.000 bước không hề thiếu — đây là bằng chứng cho phần chống quá khớp,
không phải chuyện may.

## Vì sao A0 phải chạy lại

`configs/ablation_a0_vanilla.yaml` từng đặt `so_buoc_warmup: 4000` y như bài báo
2017, trong khi ngân sách ablation chỉ có **3.000 bước**.

Hệ quả: **A0 chưa bao giờ chạy hết warmup.** Suốt cả lượt learning rate vẫn đang
tăng dần, tới lúc dừng mới đạt 75% đỉnh, trung bình cả lượt khoảng 37% đỉnh.

Nhìn bảng thì tưởng "công thức 2017 kém hơn 25 BLEU". Thật ra là **A0 chưa từng
được học ở learning rate thật**. Không có gì báo lỗi, không có gì trong log gợi
ý, và 2,5 giờ GPU đi thẳng vào một con số không dùng được.

Warmup 4.000 của bài báo là cho lượt **100.000 bước**, tức 4% ngân sách. Giữ
đúng tỉ lệ đó ở ngân sách 3.000 bước thì ra **120**. A0 vẫn là công thức 2017
(vẫn warmup, vẫn label smoothing 0,1), chỉ là warmup được quy đổi cho vừa ngân
sách — **và chuyện quy đổi này phải ghi trong báo cáo**, không được giấu.

`scripts/chay_ablation.py` giờ có cửa chặn từ chối chạy khi warmup lớn hơn hoặc
bằng ngân sách bước, in luôn con số nên dùng. Đặt cửa chặn ở đó vì đó là **chỗ
duy nhất biết cả cấu hình lẫn ngân sách bước** — hai thứ nằm ở hai file khác
nhau nên không chỗ nào khác nhìn thấy được mâu thuẫn.
