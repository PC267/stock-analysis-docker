FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_STOCK_PICKER_PORT=9119 \
    REPORT_OUTPUT_DIR=/work/app/results/reports

WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt /tmp/requirements-full.txt
RUN awk 'BEGIN{IGNORECASE=1} !/tensorflow|catboost|lightgbm|xgboost|scikit-learn|matplotlib|seaborn|statsmodels|scipy/ {print}' /tmp/requirements-full.txt > /tmp/requirements.txt \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements.txt requests baostock joblib openpyxl || true

COPY app ./app
COPY web_stock_picker ./web_stock_picker

RUN mkdir -p /work/app/stock_data /work/app/results/reports /work/app/backtest_results \
    && python -m py_compile /work/web_stock_picker/app.py

EXPOSE 9119

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9119/health', timeout=5).read()"

CMD ["python", "/work/web_stock_picker/app.py"]
