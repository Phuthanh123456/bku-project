FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1+cpu \
 && pip install --no-cache-dir --user \
        tokenizers==0.21.0 \
        PyYAML==6.0.2 \
        streamlit==1.41.1 \
        huggingface_hub==0.36.2

FROM python:3.11-slim

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/home/appuser/.cache/huggingface \
    HF_MODEL_REPO=phudeeptry/envi-translate-model \
    PORT=8501

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/
COPY --chown=appuser:appuser scripts/start_demo.py ./scripts/start_demo.py

EXPOSE 8501
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["python", "scripts/start_demo.py"]
