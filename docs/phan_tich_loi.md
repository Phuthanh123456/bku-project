# Phân tích lỗi định tính — TASK 21

> Kết quả này dùng checkpoint thật `iwslt_base_v1_seed42/tot_nhat.pt` (bước 6.000), greedy + KV cache, trên 30 câu đầu của `tst2013`.

## Kết quả định lượng của mẫu

- BLEU: **25.19** — `nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.6.0`
- chrF++: **45.64** — `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`
- Câu không có lỗi nghiêm trọng theo đọc thủ công: **5/30**.
- Đây là số trên mẫu phân tích, không thay cho điểm full test 1.268 câu.

## Bảng phân loại

| Nhóm lỗi | Số câu | Tỉ lệ | Ví dụ |
|---|---:|---:|---|
| Sai nghĩa | 10 | 33.3% | Câu 1: 'best' thành 'người giỏi nhất'; tên bài hát còn từ Envy. |
| Thiếu ý | 4 | 13.3% | Câu 3: Thiếu quan hệ sở hữu 'kẻ thù của chúng tôi'. |
| Lặp từ | 0 | 0.0% | Không ghi nhận trong mẫu 30 câu. |
| Sai tên riêng | 2 | 6.7% | Câu 1: 'best' thành 'người giỏi nhất'; tên bài hát còn từ Envy. |
| Sai con số | 1 | 3.3% | Câu 13: Bịa '20 tuổi', sai 'emaciated/lifeless' và quan hệ đại từ. |
| Sai đại từ | 4 | 13.3% | Câu 8: Ngữ cảnh lá thư cần chị/em, model dùng bạn/chúng ta. |
| Dịch sát chữ | 8 | 26.7% | Câu 12: 'erase from my memory' bị dịch máy móc thành 'xoá bỏ bộ nhớ'. |
| Ngữ pháp/độ trôi chảy | 7 | 23.3% | Câu 6: 'trải nghiệm sự đói kém' cứng và không tự nhiên. |
| Từ chưa dịch | 1 | 3.3% | Câu 1: 'best' thành 'người giỏi nhất'; tên bài hát còn từ Envy. |

![Biểu đồ phân loại lỗi](../results/phan_tich_loi.png)

## Nhận xét

Lỗi chính là chọn sai nghĩa của từ/cụm đa nghĩa và dịch sát cấu trúc tiếng Anh. Các ví dụ rõ nhất gồm `public execution`, `power outages`, `tree bark`, `repatriated` và `interrogation`. Con số nhìn chung được bảo toàn; lỗi số duy nhất trong mẫu là model tự sinh `20 tuổi` ở câu 13. Không ghi nhận lặp từ.

## Giới hạn của phân tích

1. Mẫu 30 câu liên tiếp thuộc cùng một bài TED nên không đại diện toàn bộ chủ đề tst2013.
2. Nhãn do một người đọc thủ công, chưa đo độ đồng thuận giữa nhiều người gán nhãn.
3. Một câu có thể mang nhiều nhãn, vì vậy tổng tỉ lệ các nhóm không cộng thành 100%.
4. Tham chiếu đơn cũng có cách viết/tên riêng chưa thống nhất; khác tham chiếu không tự động là sai.
5. BLEU/chrF++ trên 30 câu có phương sai cao, chỉ dùng mô tả mẫu chứ không kết luận mô hình.

Bảng chi tiết từng câu: `results/phan_tich_loi_30_cau.csv`.
