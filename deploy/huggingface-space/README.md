---
title: ENVI Translate
emoji: 🌏
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
fullWidth: true
header: mini
short_description: Dịch máy Anh–Việt bằng Transformer do nhóm tự xây dựng.
models:
  - mgbao/envi-nmt-scratch-transformer
preload_from_hub:
  - mgbao/envi-nmt-scratch-transformer checkpoints/abl3000_base_seed42/tot_nhat.pt,artifacts/tokenizer/tokenizer.json
startup_duration_timeout: 30m
---

# ENVI Translate

Demo dịch máy Anh–Việt sử dụng Transformer được xây dựng bằng PyTorch.

Ứng dụng chạy bằng Streamlit trong Docker. Checkpoint và tokenizer được tải từ
[`mgbao/envi-nmt-scratch-transformer`](https://huggingface.co/mgbao/envi-nmt-scratch-transformer)
khi Space được dựng, thay vì sao chép model lớn vào repository của Space.
