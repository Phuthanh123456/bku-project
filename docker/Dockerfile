# TASK 20 — Phú. Đóng gói demo Streamlit.
#
# Kiểu nhiều tầng: tầng đầu cài thư viện, tầng sau chỉ chép kết quả sang.
# Cách này giúp image nhẹ đi nhiều.
#
# Cài PyTorch bản CHẠY CPU thay vì bản có CUDA — bản CUDA nặng hơn khoảng năm lần
# mà demo thì không cần GPU.

# ---------- tầng 1: cài thư viện ----------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Bản CPU của PyTorch, nhẹ hơn nhiều so với bản mặc định có CUDA
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
 && pip install --no-cache-dir --user \
        tokenizers==0.21.0 \
        sacrebleu==2.4.3 \
        huggingface_hub==0.27.1 \
        PyYAML==6.0.2 \
        streamlit==1.41.1 \
        matplotlib==3.10.0

# ---------- tầng 2: image chạy thật ----------
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src

COPY src/     ./src/
COPY configs/ ./configs/
# Checkpoint và tokenizer gắn vào lúc chạy qua volume, KHÔNG nướng vào image
# (image sẽ phình to và mỗi lần đổi mô hình phải build lại từ đầu).

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/nmt/serve/ui.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
