# stock-analysis-docker

A股 Web 选股、AI 训练、量化回测与报告管理系统。

当前 Web 访问地址：

http://localhost:9119/
```

## 项目简介

本项目基于 Python + Docker，面向 A 股数据分析场景，提供以下能力：

- Web 选股评分：读取本地历史行情 CSV，并合并实时行情。
- 技术指标评分：MA、RSI、量比、20 日动量、最大回撤、波动率等。
- K 线模拟评分：对股票短期走势进行轻量模拟评分。
- 个股删除：在 Web 表格中删除指定股票历史数据。
- 历史数据更新：通过 baostock 更新服务端 CSV 历史行情。
- 重型 AI 训练：调用原项目 `app/main.py` 的多模型分析流程。
- 行业/概念资金批量训练：使用东方财富板块资金流接口筛选概念与成分股。
- 回测模块：Web 端直接调用原项目 `app/backtest.py`。
- 报告管理：展示、打开、删除 HTML/PNG/CSV/TXT 报告。
- Docker 部署：通过 `docker compose` 启动 `web-stock-picker` 服务。

## 当前主要功能

### 1. 选股评分

页面入口：

```text
http://localhost:9119/
```

### 2. 实时行情合并

Web 选股评分不会只依赖历史 CSV。服务端会尝试调用实时行情源补齐最新交易日数据：

### 3. 个股删除

选股评分页面表格有“删除”按钮。

删除成功后，页面会自动刷新股票列表。

### 4. 历史数据更新

页面按钮：“更新历史数据”。

### 5. 重型 AI 训练

Web 页面“重型AI训练”可启动完整多模型分析任务。

### 6. 东方财富板块资金流批量训练

支持按东方财富概念/板块资金流筛选股票并批量训练。
### 7. 回测模块

Web 回测模块已改为直接调用原项目：

### 8. 报告管理

报告页面支持：

- HTML 报告打开
- PNG 图表访问
- 风险指标表格展示
- 删除报告
- 按列排序

```

## 项目结构

```text
stock-analysis-docker/
├── Dockerfile                         # 轻量 Web 镜像 Dockerfile
├── docker-compose.yml                 # 当前服务编排文件
├── .dockerignore                      # Docker 构建忽略规则
├── .gitignore                         # Git 忽略规则
├── README.md                          # 项目说明文档
├── app/
│   ├── main.py                        # 原项目主分析入口，多模型 AI 分析
│   ├── backtest.py                    # 原项目回测模块
│   ├── stock_analysis.py              # 报告分析/Excel 风险汇总
│   ├── config.yaml                    # 股票系统配置
│   ├── requirements.txt               # Python 依赖
│   ├── stock_data/                    # 历史行情 CSV，运行时目录
│   ├── results/                       # 报告、图表、回测结果，运行时目录
│   ├── backtest_results/              # 回测输出目录，运行时目录
│   ├── data/                          # 数据处理相关模块
│   ├── features/                      # 特征工程模块
│   ├── model/                         # 模型/行业概念资金相关脚本
│   └── report_chart/                  # HTML/PNG/报告输出配置
└── web_stock_picker/
    ├── app.py                         # Web 服务主入口，wsgiref 实现
    ├── Dockerfile.full                # 满血 AI 镜像 Dockerfile
    └── static/                        # 静态资源目录
```

## Docker 部署

当前 compose 服务：

```yaml
services:
  web-stock-picker:
    image: web-stock-picker-full:latest
    container_name: web-stock-picker
    ports:
      - "9119:9119"
```
## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `WEB_STOCK_PICKER_PORT` | `9119` | Web 服务监听端口 |
| `REPORT_OUTPUT_DIR` | `/work/app/results/reports` | HTML/PNG/回测报告输出目录 |

## Python 依赖

依赖文件：

```text
app/requirements.txt
```

主要依赖：

- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow-cpu
- lightgbm
- catboost
- xgboost
- baostock
- statsmodels
- scipy
- openpyxl
- arch
- pykalman
- plotly

说明：

- 当前 `web-stock-picker-full:latest` 镜像用于支持完整 AI/回测依赖。
- 根目录 `Dockerfile` 会过滤部分重型依赖，适合轻量 Web 模式。
- 如需完整 AI 能力，优先使用 `web_stock_picker/Dockerfile.full` 构建。

## 免责声明

本项目输出仅用于量化研究、数据分析和策略验证，不构成任何投资建议。股票市场存在风险，历史数据、模型预测和回测结果均不能保证未来收益。
