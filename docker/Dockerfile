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

# Bản CPU của PyTorch, nhẹ hơn nhiều so với bản mặc định có CUDA
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
 && pip install --no-cache-dir --user \
        tokenizers==0.21.0 \
        PyYAML==6.0.2 \
        streamlit==1.41.1

# ---------- tầng 2: image chạy thật ----------
FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

COPY --chown=appuser:appuser src/     ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/
# Checkpoint và tokenizer gắn vào lúc chạy qua volume, KHÔNG nướng vào image
# (image sẽ phình to và mỗi lần đổi mô hình phải build lại từ đầu).

EXPOSE 8501

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/nmt/serve/ui.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.fileWatcherType=none"]
