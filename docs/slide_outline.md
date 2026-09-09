# Dàn slide bảo vệ — 15 trang

1. **Tên đề tài & thành viên** — ngày nộp, chữ ký/trách nhiệm từng người.
2. **Bài toán** — dịch Anh→Việt ít tài nguyên; benchmark IWSLT 2015.
3. **Câu hỏi nghiên cứu & target** — ≥22 BLEU; cải tiến vs vanilla cùng budget.
4. **Dữ liệu** — 133.317→131.339 train; dev/test 1.553/1.268; chống rò rỉ.
5. **Tokenizer** — BPE 32k, fertility, UNK 0%; giải thích tokenized BLEU.
6. **Kiến trúc tổng thể** — encoder 6 + decoder 6, d_model 512, 8×64/head.
7. **Bốn thay đổi** — RoPE, Pre-Norm, RMSNorm, SwiGLU; vị trí trong sơ đồ.
8. **Tính đúng đắn** — test đối chiếu, overfit 50 câu, tham số 47.956M.
9. **Train chịu lỗi** — fp16, batch token, checkpoint/HF, phục hồi 0% lệch.
10. **Đường huấn luyện** — `training_curves.png`, checkpoint tốt nhất bước 6.000.
11. **Kết quả chính** — dev 26,24; test Greedy 29,74; chữ ký metric.
12. **Beam & KV cache** — test tương đương; Beam 30,69 (+0,95), trade-off tốc độ.
13. **Phân tích lỗi** — `phan_tich_loi.png`, 3 ví dụ tốt + 3 ví dụ tệ.
14. **Ablation & trung thực thực nghiệm** — 16 lượt đã khóa; 15 lượt GPU còn thiếu,
    không kết luận đóng góp khi chưa có đối chứng.
15. **Demo, hạn chế & hướng tiếp** — mở localhost:8501; Space/ablation/ONNX.

Mỗi slide chỉ giữ một thông điệp, tối đa một bảng/hình chính; số đầy đủ và nguồn
nằm trong `docs/bao_cao_cuoi.md` để trả lời phản biện.
