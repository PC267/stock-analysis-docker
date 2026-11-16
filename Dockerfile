FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="ZWJ_stock-analysis"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_STOCK_PICKER_PORT=9129 \
    REPORT_OUTPUT_DIR=/work/app/results/reports \
    MPLBACKEND=Agg

# 🌟 修改点：默认使用清华镜像源加速
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /work

RUN echo "* soft nofile 1024" >> /etc/security/limits.conf && \
    echo "* hard nofile 1024" >> /etc/security/limits.conf

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 libgfortran5 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 拦截黑名单
RUN echo "nvidia-nccl-cu12==0.0.0" > /tmp/constraints.txt \
    && echo "nvidia-cublas-cu12==0.0.0" >> /tmp/constraints.txt \
    && echo "nvidia-cudnn-cu12==0.0.0" >> /tmp/constraints.txt \
    && echo "nvidia-cuda-runtime-cu12==0.0.0" >> /tmp/constraints.txt

COPY app/requirements.txt /tmp/requirements-full.txt
RUN awk 'BEGIN{IGNORECASE=1} !/tensorflow/ && !/xgboost/ {print}' /tmp/requirements-full.txt > /tmp/requirements.txt

# 安装基础库
RUN python -m pip install -i ${PIP_INDEX_URL} --upgrade pip setuptools wheel \
    && python -m pip install -c /tmp/constraints.txt -i ${PIP_INDEX_URL} -r /tmp/requirements.txt requests baostock joblib openpyxl waitress \
    # 🌟 安装 TensorFlow-CPU (走清华源)
    && echo ">>> 开始安装 TensorFlow-CPU..." \
    && python -m pip install -c /tmp/constraints.txt -i ${PIP_INDEX_URL} tensorflow-cpu \
    # 🌟 安装 XGBoost (无依赖模式，走清华源)
    && echo ">>> 开始安装 XGBoost (无依赖模式)..." \
    && python -m pip install -c /tmp/constraints.txt -i ${PIP_INDEX_URL} xgboost --no-deps \
    # 验证
    && python -c "import importlib.util, sys; mods = 'pandas numpy sklearn matplotlib seaborn lightgbm catboost xgboost tensorflow scipy statsmodels arch plotly pykalman baostock joblib openpyxl waitress requests yaml pytz psutil'.split(); missing = [m for m in mods if importlib.util.find_spec(m) is None]; print('dependency check ok' if not missing else 'missing: %s' % missing); sys.exit(1 if missing else 0)"

COPY app ./app
COPY web_stock_picker ./web_stock_picker

RUN mkdir -p /work/app/stock_data /work/app/results/reports /work/app/backtest_results /work/runtime /work/web_stock_picker/runtime \
    && python -m py_compile /work/web_stock_picker/app.py

EXPOSE 9129

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9129/health', timeout=5).read()"

CMD ["python", "/work/web_stock_picker/app.py"]