# Bao Cao Kien Truc

> **Phụ trách: TASK 05-10 — Bảo, Quân, My**
>
> File này còn trống. Điền vào trong lúc làm task, đừng để tới tuần 4 mới viết.

## Nội dung cần có

- Sơ đồ kiến trúc do nhóm tự vẽ, có kích thước tensor ở mỗi chặng
- Đánh dấu rõ chỗ nào CÓ áp RoPE, chỗ nào KHÔNG, và vị trí của RMSNorm cuối
- Bảng kết quả 12 bài kiểm tra: tên bài | kiểm cái gì | đạt/không đạt (yêu cầu 12/12)
- Bảng đếm tham số theo thành phần: embedding | encoder | decoder | lớp xuất | tổng
-   -> đối chiếu với con số tính tay trên giấy, sai lệch phải dưới 1%
- Bảng đối chiếu với cài đặt tham chiếu PyTorch (yêu cầu sai lệch < 1e-5):
-   - RMSNorm tự viết  vs  torch.nn.RMSNorm
-   - lõi attention    vs  F.scaled_dot_product_attention
- Biểu đồ loss của bài test học thuộc 50 câu (yêu cầu < 0,05 trong <= 500 bước)
-   -> HÌNH QUAN TRỌNG NHẤT của Phase 2, là bằng chứng trực quan kiến trúc đúng
- Bảng 5 câu ví dụ sau khi học thuộc: câu Anh | bản dịch chuẩn | mô hình dịch ra
- Bản đồ nhiệt trọng số attention cho một câu mẫu
- Hình minh họa tính chất RoPE: hai đường (cặp vị trí từ 0 và từ 20) phải TRÙNG NHAU
