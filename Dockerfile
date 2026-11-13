FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="ZWJ_stock-analysis"

# 加入 MPLBACKEND=Agg，防止 matplotlib 在无头容器中报错
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_STOCK_PICKER_PORT=9129 \
    REPORT_OUTPUT_DIR=/work/app/results/reports \
    MPLBACKEND=Agg

# 依赖下载源：默认官方源；国内网络可改用清华镜像以加快构建
#   docker compose build --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /work

# 降低文件描述符限制
RUN echo "* soft nofile 1024" >> /etc/security/limits.conf && \
    echo "* hard nofile 1024" >> /etc/security/limits.conf

# 基础依赖：
#   libgomp1 / libgfortran5 —— LightGBM、XGBoost、SciPy 的运行时依赖
#   fonts-noto-cjk         —— matplotlib 生成中文图表所需（否则中文显示为方块）
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 libgfortran5 fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖：只过滤掉 tensorflow。
# 容器内已用 stub 跳过 LSTM（TensorFlow 在本 CPU 环境会 SIGILL），装它只会白占镜像体积。
# 这里不再使用 `|| true`：依赖装不上就让构建直接失败，避免容器启动后才暴露缺包。
COPY app/requirements.txt /tmp/requirements-full.txt
RUN awk 'BEGIN{IGNORECASE=1} !/tensorflow/ {print}' /tmp/requirements-full.txt > /tmp/requirements.txt \
    && python -m pip install -i ${PIP_INDEX_URL} --upgrade pip setuptools wheel \
    && python -m pip install -i ${PIP_INDEX_URL} -r /tmp/requirements.txt requests baostock joblib openpyxl waitress \
    && python -c "import importlib.util, sys; mods = 'pandas numpy sklearn matplotlib seaborn lightgbm catboost xgboost scipy statsmodels arch plotly pykalman baostock joblib openpyxl waitress requests yaml pytz psutil'.split(); missing = [m for m in mods if importlib.util.find_spec(m) is None]; print('dependency check ok' if not missing else 'missing: %s' % missing); sys.exit(1 if missing else 0)"

# 复制代码
COPY app ./app
COPY web_stock_picker ./web_stock_picker

# 创建必要目录，并预编译校验 app.py 语法
RUN mkdir -p /work/app/stock_data /work/app/results/reports /work/app/backtest_results /work/runtime /work/web_stock_picker/runtime \
    && python -m py_compile /work/web_stock_picker/app.py

EXPOSE 9129

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9129/health', timeout=5).read()"

CMD ["python", "/work/web_stock_picker/app.py"]
