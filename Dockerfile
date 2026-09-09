# Dockerfile ở root để Hugging Face Space loại Docker nhận ra tự động.
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements-demo.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0+cpu \
 && pip install --no-cache-dir --user -r requirements-demo.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    HF_MODEL_REPO=mgbao/envi-nmt-scratch-transformer \
    DUONG_DAN_CONFIG=/app/configs/iwslt_base_v1_seed42.yaml
COPY src/ ./src/
COPY configs/ ./configs/
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/_stcore/health')"
CMD ["streamlit", "run", "src/nmt/serve/ui.py", "--server.port=7860", "--server.address=0.0.0.0"]
