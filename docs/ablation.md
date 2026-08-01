# Ablation

> **Phụ trách: TASK 17 + 18 — Phú**
>
> File này còn trống. Điền vào trong lúc làm task, đừng để tới tuần 4 mới viết.

## Nội dung cần có

- Bảng ablation kiến trúc (TASK 17), 4 thí nghiệm:
-   A1 RMSNorm vs LayerNorm   <- ưu tiên cao nhất, mentor hỏi trực tiếp
-   A4 RoPE vs sin-cos
-   A5 SwiGLU vs ReLU
-   A6 Pre-Norm vs Post-Norm
-   cột: loss dev | BLEU | chrF++ | ĐỘ LỆCH GIỮA CÁC SEED

- Bảng ablation kỹ thuật huấn luyện (TASK 18), 2 thí nghiệm:
-   A2 Warmup bật vs tắt
-   A3 Label Smoothing 0,1 vs 0
-   kèm 2-3 câu diễn giải con số đó nói lên điều gì
-   kèm biểu đồ hai đường loss vẽ chồng lên nhau để thấy khác biệt tốc độ hội tụ
-   -> kết quả QUYẾT ĐỊNH giữ hay bỏ hai kỹ thuật này khỏi cấu hình cuối cùng

- LƯU Ý VIẾT SẴN VÀO BÁO CÁO:
- A2 nhiều khả năng chênh lệch rất nhỏ, vì Pre-Norm vốn đã làm giảm nhu cầu warmup
- (On Layer Normalization in the Transformer Architecture, arxiv 2002.04745).
- Đó KHÔNG phải bug và cũng KHÔNG phải ablation thất bại, mà chính là điều bài báo dự đoán.
- Giải thích được một kết quả bằng lý thuyết đã công bố thì thuyết phục hơn nhiều
- so với chỉ đưa ra con số.

- Mỗi thí nghiệm: tối thiểu 2 seed, cùng ngân sách bước, chỉ đổi đúng một yếu tố.
