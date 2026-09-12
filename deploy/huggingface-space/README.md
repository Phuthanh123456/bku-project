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
  - phudeeptry/envi-translate-model
preload_from_hub:
  - phudeeptry/envi-translate-model inference_fp16.pt,tokenizer.json
startup_duration_timeout: 30m
---

# ENVI Translate

Demo dịch máy Anh–Việt sử dụng Transformer được xây dựng bằng PyTorch.

Ứng dụng chạy bằng Streamlit trong Docker. Checkpoint và tokenizer được tải từ
[`phudeeptry/envi-translate-model`](https://huggingface.co/phudeeptry/envi-translate-model)
khi Space được dựng, thay vì sao chép model lớn vào repository của Space.
