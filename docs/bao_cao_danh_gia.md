# Báo cáo đánh giá — TASK 16 + 19

Checkpoint được chấm là `iwslt_base_v1_seed42/tot_nhat.pt`, metadata xác nhận
`che_do=that`, seed 42, bước 6.000, epoch 34 và loss dev 2,2364. Hướng dịch là
Anh→Việt; dev dùng toàn bộ `tst2012`, test dùng toàn bộ `tst2013`.

## Điểm chính

| Tập | Số câu | Search | BLEU | chrF++ | Thời gian CPU |
|---|---:|---|---:|---:|---:|
| tst2012 (dev) | 1.553 | Greedy + KV cache | 26,24 | 45,57 | 291,6 giây |
| tst2013 (test) | 1.268 | Greedy + KV cache | 29,74 | 48,73 | 165,4 giây |
| tst2013 (test) | 1.268 | Beam-4, alpha=1,0 + KV cache | **30,69** | **49,60** | 913,3 giây |

Beam tăng **+0,95 BLEU** và **+0,87 chrF++** so với Greedy, vượt tiêu chí
TASK 19 là ít nhất +0,5 BLEU. Đổi lại thời gian đánh giá theo batch tăng 5,52 lần.

![So sánh Greedy và Beam](../results/search_comparison.png)

## Protocol metric

Corpus IWSLT trong repo đã có khoảng trắng tách token/dấu câu. Vì vậy điểm chính
dùng SacreBLEU `tokenize=none`; dùng `13a` thêm lần nữa sẽ khiến SacreBLEU cảnh
báo và signature mô tả sai protocol. Hai dòng `tok:13a` cũ vẫn được giữ trong
CSV để truy vết nhưng được đánh dấu không dùng làm kết luận.

- BLEU: `nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.6.0`
- chrF++: `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`

Mốc Luong & Manning 26,4 là tokenized BLEU trên tst2013 được Jones et al. (2020)
tổng hợp. Điểm trong dự án vượt mốc số học đó, nhưng không tuyên bố hơn trực tiếp
vì pipeline tokenization/implementation không được chứng minh giống hoàn toàn.

## Mười câu ví dụ

Các câu được chọn có chủ ý gồm cả tốt lẫn tệ. Beam tốt hơn về corpus BLEU nhưng
không bảo đảm sửa mọi lỗi của Greedy.

| # | EN (rút gọn) | Tham chiếu (rút gọn) | Greedy | Beam |
|---:|---|---|---|---|
| 1 | When I was little... Nothing To Envy | ...bài “Chúng ta chẳng có gì phải ghen tị” | ...“Không có gì với Envy” | ...“Không có gì với Envy” |
| 2 | And I was very proud. | Tôi đã rất tự hào... | Và tôi rất tự hào. | Và tôi rất tự hào. |
| 4 | Although I often wondered... | Mặc dù... tôi vẫn nghĩ... | Truyền đạt đủ ý chính | Trùng Greedy |
| 5 | my first public execution | xử bắn công khai lần đầu | cuộc hành trình công cộng | cuộc hành trình công cộng |
| 7 | a coworker's sister | một người chị em cùng chỗ làm | chị dâu của thợ săn | chị dâu |
| 10 | I was so shocked. | Tôi đã bị sốc. | Tôi đã rất sốc. | Tôi đã rất sốc. |
| 13 | an emaciated child | một đứa bé hốc hác | một đứa trẻ 20 tuổi | một đứa trẻ 20 tuổi |
| 17 | Power outages... sea of lights | cúp điện... ánh sáng đèn | Quyền lực... biển lửa | Trùng Greedy |
| 22 | But many die. | Nhưng rất nhiều người đã chết. | Nhưng nhiều người chết. | Trùng Greedy |
| 23 | bodies floating down the river | xác... nổi trên sông | xác chết đang trôi dạt | xác chết đang trỗi dậy |

Toàn bộ câu dịch nằm ở `results/du_doan_test_greedy.csv` và
`results/du_doan_test_beam.csv`; không chỉ lưu các ví dụ thuận lợi.
