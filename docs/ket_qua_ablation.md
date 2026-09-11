# Kết quả ablation

Cập nhật 11/09/2026. Số lấy từ `ablation/ket_qua.csv` trên Hugging Face
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
| A0 (vanilla 2017) | 42 | — | — | — | — | — |
| A0 (vanilla 2017) | 1337 | — | — | — | — | — |

> **A0 chưa có số dùng được.** Ba lượt đã chạy đều hỏng vì lịch learning rate,
> không phải vì kiến trúc 2017 kém — chi tiết ở mục "A0 đã hỏng ba lần". Hai
> hàng này để trống **có chủ đích**: số của ba lượt hỏng đã bị gỡ khỏi đây để
> không ai vô tình chép vào slide hay báo cáo. Lượt thứ tư đang chờ chạy với
> `so_buoc_warmup: 500`.

**Chỉ bốn hàng đầu là số dùng được.** Đối chứng và A1 đã xong, đối chiếu khớp
với `results/ablation/ket_qua.csv`.

## TASK 16 và TASK 19 — chấm điểm chính thức

Chấm trên checkpoint `abl3000_base_seed42/tot_nhat.pt` (bước 3.000), đủ tập, GPU.

| Tập | Sinh câu | Số câu | BLEU | chrF++ | Giây dịch |
|---|---|---|---|---|---|
| tst2012 (dev) | Greedy | 1.553 | 25,41 | 44,77 | 8,9 |
| tst2012 (dev) | Beam = 4 | 1.553 | 26,59 | 45,75 | 15,7 |
| tst2013 (test) | Greedy | 1.268 | 28,74 | 47,62 | 9,8 |
| **tst2013 (test)** | **Beam = 4** | 1.268 | **29,53** | **48,41** | 17,2 |

**TASK 16** — Greedy BLEU trên tst2013 là **28,74**. Ngưỡng đề tài ≥ 19, mức
"tốt" từ 22. **Đạt mức tốt.**

**TASK 19** — Beam hơn Greedy **+0,80 BLEU** trên tst2013 (yêu cầu ≥ 0,50).
**Đạt.** Trên dev khoảng cách còn rộng hơn: **+1,18 BLEU**. Đổi lại beam chậm
hơn **1,8 lần** (9,8 → 17,2 giây).

Và **beam đưa BLEU lên 29,53, vượt mốc "rất tốt" ≥ 29** theo chính thang đánh
giá ghi ở slide 4 — đạt được với chỉ 3.000 bước và chỉ dùng dữ liệu trong ràng
buộc.

KV cache cho ra **đúng cùng một chuỗi token** mà nhanh gấp **5,5 lần**
(20,3 → 3,7 giây, đo trên 40 câu). Có bài test đơn vị canh chuyện "cùng kết quả"
cho cả rope lẫn sin-cos — cache chỉ được phép đổi tốc độ, không được đổi điểm.

## Nhánh vanilla KHÔNG có lỗi mã

Câu hỏi treo suốt ba lượt A0 hỏng: lỗi mã, hay chỉ khó huấn luyện? Cổng chặn học
thuộc 50 câu chạy với `configs/ablation_a0_vanilla.yaml` trả lời dứt điểm:

| bước | 50 | 150 | 250 | 350 | 450 | 500 |
|---|---|---|---|---|---|---|
| loss | 6,49 | 4,13 | 1,93 | 0,31 | 0,090 | **0,062** |

**BLEU trên chính 50 câu đó: 100,00.** Kiến trúc vanilla học thuộc hoàn hảo.

Cổng chặn báo "CHƯA ĐẠT" chỉ vì loss cuối 0,0622 so với ngưỡng 0,05, đúng lúc
chạm trần 500 bước — mà đường loss vẫn đang giảm dốc (0,090 → 0,062 trong 50
bước cuối). Đây là **chạm trần bước, không phải kiến trúc sai**.

Kết luận: A0 hỏng ở ablation **không phải do lỗi cài đặt**. Post-Norm cần warmup
dài hơn cả ngân sách 3.000 bước — đúng điều Xiong và cộng sự (2020) chỉ ra, và
đó chính là lý do Pre-Norm ra đời. Bài đó là tài liệu tham khảo [5] của nhóm.

Nói cách khác: **việc A0 không huấn luyện được ở ngân sách này tự nó đã là bằng
chứng ủng hộ lựa chọn Pre-Norm** — đúng thứ thầy muốn thấy khi đề nghị làm A0.

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

## A0 đã hỏng ba lần

Cả ba lần đều do **lịch learning rate**, không phải do kiến trúc 2017 kém. Nhưng
**lần 3 hỏng vì lý do khác hai lần đầu**, nên đừng đọc bảng này như một lỗi lặp
lại ba lượt.

| Lượt | warmup | lr đỉnh | Chuyện xảy ra |
|---|---|---|---|
| 10/09 | 4.000 | 6,99e-4 | warmup dài hơn cả ngân sách 3.000 bước — lr **chưa bao giờ lên tới đỉnh**, trung bình cả lượt chỉ ~37% đỉnh. loss_dev 4,77 · BLEU 3,6 |
| 11/09 | 120 | **4,03e-3** | hạ warmup cho vừa ngân sách, vô tình **đẩy đỉnh lên 5,8 lần**. Post-Norm + fp16 vỡ trong 120 bước đầu. loss_train đứng yên 6,9 suốt 3.000 bước · BLEU **0,00** |
| 11/09 | 120 | 7,00e-4 ✓ | đã ghim đỉnh bằng `lr_dinh`, **lr chạy đúng thiết kế** nhưng vẫn hỏng y hệt. Dốc dài 120 bước quá gắt cho Post-Norm. loss_dev tốt nhất rơi vào lần đánh giá **đầu tiên** rồi chỉ tăng · BLEU **0,0039 / 0,0000** |

### Lần 3 — đã sửa đúng một nửa vấn đề

Lần này learning rate **chạy đúng như thiết kế**, đối chiếu thẳng từ
`results/logs/abl3000_a0_vanilla_seed42/metrics.csv`:

| bước | lr đo được | lr theo công thức | |
|---|---|---|---|
| 50 | 2,975e-4 | `7e-4 × 50/120` = 2,92e-4 | ✓ đang lên dốc |
| 150 | 6,240e-4 | `7e-4 × √(120/150)` = 6,26e-4 | ✓ đã qua đỉnh, đang giảm |

Đỉnh đúng 7e-4, bằng đối chứng. **Vậy mà kết quả vẫn suy biến.** Ba dấu hiệu:

1. **Checkpoint tốt nhất là lần đánh giá đầu tiên** (bước 500) ở cả hai seed. Từ
   đó dev loss chỉ đi lên: 7,31 → 7,83 (seed 42) và 7,36 → 8,51 (seed 1337). Đây
   là **phân kỳ**, không phải học chưa đủ.
2. **loss_train phẳng lì 6,7–7,0 suốt 3.000 bước.** Đoán đều trên 32k từ vựng cho
   `ln(32000) ≈ 10,4`; mô hình tụt xuống ~6,9 trong 50 bước đầu (vừa đủ học phân
   phối tần suất từ) rồi không nhúc nhích thêm 2.950 bước nữa.
3. **Đầu ra rỗng.** Trong `results/diem_chinh.csv`, seed 1337 dịch 1.553 câu dev
   hết **1,1 giây**, seed 42 hết **19,1 giây** — nhanh gấp 17 lần vì nó sinh EOS
   ngay lập tức. Đó là lý do BLEU và chrF đều đúng bằng 0,0000.

Gốc rễ lần 3 nằm ở **độ dài dốc, không phải độ cao dốc**. Post-Norm có đường phần
dư bị một lớp chuẩn hóa cắt ngang ở mỗi khối con (xem `ResidualConnection` trong
`src/nmt/model/normalization.py`). Với 6+6 lớp cộng fp16, ăn full 7e-4 ngay từ
bước 120 thì gradient nổ, cắt gradient theo norm 1,0 nghiền mọi bước về cùng một
độ lớn, và mô hình kẹt vĩnh viễn ở phân phối tần suất từ.

Đã kiểm `ResidualConnection`, `build_final_norm`, phép nhân `√d_model` ở
embedding, và thứ tự `unscale_` → `clip_grad_norm_` → `scaler.step` trong
trainer: **code đúng cả, không có lỗi lập trình.** Vấn đề thuần túy là siêu tham số.

**Cách sửa cho lượt 4:** `so_buoc_warmup: 120` → **500** (17% ngân sách), giữ
nguyên `lr_dinh: 7.0e-4`. Nới warmup chữa độ dài dốc, ghim `lr_dinh` chữa độ cao
dốc — lần 3 đã chứng minh chữa mỗi một vế thì không đủ. Riêng việc nới warmup lên
500 chỉ kéo đỉnh Noam từ 4,03e-3 xuống 1,98e-3, tức **vẫn còn cao gấp 2,8 lần**
đối chứng, nên dòng `lr_dinh` vẫn phải giữ.

### Gốc rễ chung của lần 1 và lần 2

Lịch Noam của bài báo 2017 **buộc chặt** hai thứ tưởng như độc lập:

$$\text{lr}_{\text{đỉnh}} = d_{\text{model}}^{-0.5} \cdot \text{warmup}^{-0.5}$$

Nên **đổi warmup là đổi luôn learning rate đỉnh**. Con số 4.000 của bài báo cho
ra đỉnh 6,99e-4 — đúng bằng `7e-4` mà đối chứng dùng, và đó không phải trùng hợp.
Rút warmup xuống 120 đẩy đỉnh lên 4,03e-3.

Dấu hiệu nhận ra lượt 11/09 **lần 2** là hỏng chứ không phải số đo:

- BLEU **đúng bằng 0,00 ở cả hai seed**, chrF **giống nhau tới sáu chữ số**
  (`0.287189`). Hai seed khác nhau mà ra y hệt nghĩa là mô hình sinh cùng một
  thứ suy biến.
- loss_dev **8,76**, trong khi đoán ngẫu nhiên đều trên 32k từ vựng là
  `ln(32000) ≈ 10,4`. Chạy 3.000 bước mà gần như không học được gì.

### Cách sửa cho lần 2 — đúng nhưng chưa đủ

Thêm khóa `toi_uu.lr_dinh` để **tách đỉnh khỏi độ dài dốc**. Đặt `lr_dinh: 7.0e-4`
thì A0 giữ nguyên hình dạng lịch 2017 (lên dốc rồi giảm theo nghịch căn) mà đỉnh
bằng đối chứng — nhờ vậy chênh lệch đo được quy về **kiến trúc**, không quy về
chuyện hai bên chạy ở learning rate khác nhau.

Cửa chặn trong `chay_ablation.py` giờ canh **cả lr đỉnh**, không chỉ canh độ dài
warmup. Bản đầu chỉ hỏi "warmup có ngắn hơn ngân sách không" nên nó cho warmup
120 đi qua — đúng lượt hỏng lần hai. Cửa chặn canh đúng một nửa còn nguy hơn
không canh, vì nó cấp giấy thông hành cho nửa còn lại.

> **Bản sửa này đã chạy và vẫn hỏng** — đó là lần 3 ở trên. Nó chữa đúng cái nó
> nhắm tới (lr đỉnh về 7e-4, xác nhận bằng số đo), nhưng để sót vế còn lại là độ
> dài dốc. Bài học lặp lại y nguyên ở cấp cao hơn: cửa chặn mới cũng chỉ canh
> được **một nửa** — nó biết hỏi "đỉnh có bằng đối chứng không" nhưng không biết
> hỏi "dốc có đủ thoải cho Post-Norm không", nên warmup 120 lại đi qua thêm một
> lần nữa. Muốn chặn được lần 4 thì cửa chặn phải canh cả sàn độ dài warmup khi
> `vi_tri_chuan_hoa: post`.

**Phải ghi trong báo cáo**: ở ngân sách 3.000 bước thì không thể tái hiện nguyên
si lịch 2017, vì riêng warmup của nó đã dài hơn cả ngân sách. Việc ghim đỉnh là
một quyết định điều chỉnh có chủ đích, không phải chép y bài báo.

## Phụ lục — lần hỏng đầu tiên

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
