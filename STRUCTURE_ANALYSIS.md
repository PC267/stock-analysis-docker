# stock-analysis-docker 项目结构分析

路径：`/vol1/1000/docker/stock-analysis-docker`

容器路径：`/work`

Web 地址：`http://192.168.3.17:9119/`

运行容器：`web-stock-picker`

更新时间：2026-05-16

---

## 一、整体定位

这是一个“原始重型 AI 股票分析项目 + 轻量 Web 选股系统”的混合项目。

- `app/`：原始股票 AI 分析主项目，包含数据获取、特征工程、多模型预测、回测、HTML 报告生成、历史行情 CSV、报告结果。
- `web_stock_picker/`：新增 Web 选股服务，端口 9119，负责页面、选股评分、报告列表、重型 AI 任务入口、回测入口、历史数据更新。

---

## 二、顶层结构

```text
stock-analysis-docker/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── .dockerignore
├── .gitignore
├── app/
└── web_stock_picker/
```

### 说明

- `Dockerfile`：轻量 Web 镜像构建文件。
- `docker-compose.yml`：启动 `web-stock-picker` 容器，挂载 app 数据和 web_stock_picker 服务。
- `README.md`：项目说明文档。
- `app/`：原始股票 AI 分析核心。
- `web_stock_picker/`：9119 Web 服务入口。

---

## 三、当前目录体量

容器 `/work` 内统计：

| 目录 | 大小 | 文件数 |
|---|---:|---:|
| `/work` 总计 | 约 39 MB | 约 328 |
| `app/` | 约 38.47 MB | 约 229 |
| `web_stock_picker/` | 约 0.52 MB | 约 94 |

注意：该统计来自容器内 `/work`，不包含宿主机可能存在但未挂载进容器的 `app/venv` 等大目录。

---

## 四、核心文件

| 文件 | 大小/行数 | 作用 |
|---|---:|---|
| `web_stock_picker/app.py` | 约 58 KB / 844 行 | 当前 Web 系统主入口，包含 API 路由、页面、评分、报告、AI 任务、回测、历史数据更新等逻辑 |
| `app/main.py` | 约 47 KB / 1084 行 | 原始重型 AI 分析入口，核心函数 `analyze_single_stock()` |
| `app/backtest.py` | 约 51 KB / 1337 行 | 原始回测系统，含参数优化、回测评估、推荐聚合 |
| `app/stock_analysis.py` | 约 28 KB / 714 行 | 分析已有股票报告结果 |
| `app/config_loader.py` | 约 95 KB | 配置系统核心，包含大量配置类 |
| `app/config.yaml` | 约 14 KB / 525 行 | 原始分析项目配置文件 |
| `app/report_chart/report_generator.py` | 约 69 KB | HTML 报告、图表、批量分析报告生成 |
| `app/report_chart/report_config.py` | 约 4 KB | 报告输出目录控制，路径问题关键文件 |
| `app/model/factory.py` | 约 49 KB | 模型工厂，统一创建预测模型 |
| `app/model/simulator.py` | 约 71 KB | 股票模拟预测主逻辑 |

---

## 五、`app/` 目录结构

```text
app/
├── main.py
├── run.py
├── stock_analysis.py
├── backtest.py
├── config.py
├── config.yaml
├── config_loader.py
├── requirements.txt
├── README.md
├── data/
├── features/
├── model/
├── report_chart/
├── stock_data/
├── results/
└── backtest_results/
```

### 1. `app/data/`

数据层：

```text
app/data/
├── data_fetcher.py
├── data_fetcher_multi.py
└── data_processor.py
```

作用：拉取股票历史数据、多源数据、清洗处理。

### 2. `app/features/`

特征工程：

```text
app/features/
├── technical_features.py
└── market_features.py
```

作用：计算技术指标、市场特征、模型输入特征。

### 3. `app/model/`

模型层：

```text
app/model/
├── factory.py
├── simulator.py
├── evaluator.py
├── GaiNian stock.py
├── Capital inflow analysis.py
└── Financial ranking.py
```

作用：

- 多模型预测
- 模型评估
- 概念板块资金流
- 财务排名
- 资金流分析

重点文件：

- `GaiNian stock.py`：东方财富概念板块资金流相关逻辑关键文件。
- `factory.py`：模型工厂。
- `simulator.py`：股票模拟预测核心。

### 4. `app/report_chart/`

报告生成：

```text
app/report_chart/
├── chart_generator.py
├── report_generator.py
└── report_config.py
```

作用：生成图表、HTML 报告、控制报告目录。

### 5. `app/stock_data/`

历史行情 CSV 数据目录。

当前约 43 个文件。

Web 选股评分主要读取：

```text
*_baostock.csv
```

### 6. `app/results/reports/`

HTML 报告输出目录。

Web「报告结果」页读取这里。

### 7. `app/backtest_results/`

回测结果目录，包含参数推荐结果：

```text
app/backtest_results/parameter_recommendations/
```

---

## 六、`web_stock_picker/` 目录结构

```text
web_stock_picker/
├── app.py
├── Dockerfile.full
├── requirements.txt
├── static/
├── templates/
├── ai_job_*.log
└── ai_runner_*.py
```

### 1. `web_stock_picker/app.py`

当前 Web 服务核心，主要 API 都在这里。

核心功能：

- 股票列表评分
- 实时行情合并
- K 线模拟评分
- 行业批量 AI 分析
- 东方财富 / 新浪备用数据源
- 历史数据更新
- 报告列表读取
- HTML 报告指标解析
- 报告删除
- Web API 路由

主要 API：

```text
/health
/api/stocks
/api/stocks/delete
/api/ai/train
/api/ai/status
/api/models
/api/features
/api/analyze
/api/history/update
/api/backtest
/api/reports
/api/reports/delete
/api/config
/reports/<filename>
```

### 2. `web_stock_picker/Dockerfile.full`

重型 AI 满血镜像方案，用于安装完整 AI 依赖，例如：

- sklearn
- xgboost
- lightgbm
- catboost
- matplotlib
- scipy
- statsmodels
- arch

### 3. `web_stock_picker/requirements.txt`

当前为空。

### 4. `web_stock_picker/static/`

旧式静态资源：

```text
web_stock_picker/static/
├── app.js
└── style.css
```

注意：当前实际页面很多内容可能直接内嵌在 `web_stock_picker/app.py` 中，而不是完全依赖 `static/app.js`。

### 5. `web_stock_picker/templates/`

旧式模板：

```text
web_stock_picker/templates/index.html
```

### 6. `ai_job_*.log` 与 `ai_runner_*.py`

重型 AI 任务运行时产物：

- `ai_job_*.log`：重型 AI 任务日志。
- `ai_runner_*.py`：Web 发起重型 AI 任务时生成的临时 runner 脚本。

建议：这些不应提交 GitHub，可定期清理，或迁移到 `runtime/jobs/`。

---

## 七、当前架构关系

### 页面访问链路

```text
浏览器
→ http://192.168.3.17:9119/
→ web-stock-picker 容器
→ /work/web_stock_picker/app.py
```

### 轻量选股链路

```text
web_stock_picker/app.py
→ 读取 /work/app/stock_data/*.csv
→ 合并腾讯/新浪实时行情
→ 计算评分、波动率、回撤、夏普、胜率
→ 返回 /api/stocks
```

### 报告结果链路

```text
web_stock_picker/app.py
→ 读取 /work/app/results/reports/*.html
→ 解析 HTML 指标
→ 返回 /api/reports
→ 打开 /reports/<报告名>.html
```

### 重型 AI 链路

```text
web_stock_picker/app.py
→ 生成 ai_runner_*.py
→ 子进程调用 /work/app/main.py 的 analyze_single_stock()
→ 输出 HTML 报告到 /work/app/results/reports/
```

### 回测链路

```text
web_stock_picker/app.py
→ 可调用轻量回测或 app/backtest.py 原始回测能力
```

---

## 八、关键函数 / 类概览

### `web_stock_picker/app.py`

主要函数：

```text
to_float
avg
ma
rsi
max_drawdown
volatility
normalize_code
parse_name
read_stock
fetch_sina_quote
fetch_tencent_quote
fetch_realtime_quote
merge_realtime_row
kline_simulation
score_stock
find_item
load_items
apply_strategy
stock_payload
delete_stock_payload
ai_python_cmd
em_json
datacenter_json
local_industry_batch
sina_market_rows
alt_top_industries
alt_industry_stocks
eastmoney_top_industries
eastmoney_industry_stocks
fallback_industry_batch
load_gainian_module
resolve_industry_batch
run_ai_batch_job
run_ai_job
ai_train_payload
ai_status_payload
models_payload
features_payload
analyze_payload
update_history_for_file
history_update_payload
backtest_payload
report_roots
safe_report_path
extract_html_metrics
extract_html_total_return
report_display_name
report_metrics_from_name
reports_payload
delete_report_payload
config_payload
resp
read_body
application
```

### `app/main.py`

主要函数：

```text
exit_handler
get_best_model_fallback
cleanup_resources
analyze_single_stock
main
```

### `app/backtest.py`

主要类：

```text
StockDataLoader
ParameterOptimizer
BacktestConfig
BacktestProcessor
BacktestEvaluator
BacktestSimulator
ParameterRecommendationAggregator
```

### `app/config_loader.py`

主要类：

```text
StockConfig
DataConfig
ModelConfig
SARIMAConfig
GARCHConfig
XGBoostConfig
MultiFactorConfig
FeaturesConfig
EnsembleConfig
CorrectionConfig
ProbabilisticConfig
MonitoringConfig
SimulatorConfig
ReportingConfig
TechnicalIndicatorsConfig
TechnicalFeaturesConfig
FullConfig
ModelRecommendation
ParameterConfigLoader
ConfigManager
Config
```

### `app/report_chart/report_generator.py`

主要类：

```text
ReportGenerator
ReportConfig
BatchAnalysisManager
```

### `app/model/factory.py`

主要类：

```text
ModelFactory
```

### `app/model/simulator.py`

主要类：

```text
DynamicModelSelector
ModelConfigManager
StockSimulator
```

---

## 九、结构优点

1. 原始 AI 能力保留完整

`app/` 目录仍包含完整数据、模型、回测、报告系统。

2. Web 层轻量

`web_stock_picker/app.py` 使用标准库 WSGI 实现，不强依赖 Flask。

3. 数据目录挂载清晰

CSV、报告、回测结果都在 `app/` 下，便于持久化。

4. 支持轻量和重型两种模式

- 快速评分：读取 CSV + 实时行情。
- 深度分析：调用 `main.py` 重型模型流程。

5. Docker 化程度较好

根目录已有：

```text
Dockerfile
docker-compose.yml
.dockerignore
.gitignore
```

---

## 十、结构问题

1. `web_stock_picker/app.py` 过于集中

844 行，承担了：

- 前端 HTML
- API 路由
- 数据源
- 评分逻辑
- 报告解析
- AI 任务调度
- 回测
- 历史数据更新

建议后续拆分。

2. `app/` 原始项目和 Web 服务耦合偏强

Web 直接读取：

```text
app/stock_data
app/results/reports
app/main.py
```

优点是简单，缺点是路径和环境变量容易混乱。

3. 运行时文件混在源码目录

`web_stock_picker/` 下有大量：

```text
ai_job_*.log
ai_runner_*.py
```

建议迁移到：

```text
runtime/jobs/
```

或：

```text
app/results/jobs/
```

4. 报告指标来源有重复逻辑

报告结果页既可以自己计算风险指标，也可以解析 HTML。

这会导致：

- 风险评分不一致
- 风险等级不一致

建议以 HTML 报告解析值为准，或统一使用同一个 risk 计算函数。

5. `app/model/` 文件名有空格

例如：

```text
GaiNian stock.py
Capital inflow analysis.py
Financial ranking.py
```

这会增加 import 和维护难度。

建议长期改名：

```text
gainian_stock.py
capital_inflow_analysis.py
financial_ranking.py
```

6. `app/config_loader.py` 过大

约 95 KB，配置类过于集中。

后续可以拆成：

```text
config/schema.py
config/loader.py
config/defaults.py
config/exporter.py
```

---

## 十一、建议目标结构

```text
stock-analysis-docker/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── app/
│   ├── main.py
│   ├── backtest.py
│   ├── stock_analysis.py
│   ├── config.yaml
│   ├── data/
│   ├── features/
│   ├── model/
│   ├── report_chart/
│   ├── stock_data/
│   ├── results/
│   └── backtest_results/
├── web_stock_picker/
│   ├── app.py
│   ├── routes/
│   │   ├── stocks.py
│   │   ├── reports.py
│   │   ├── ai.py
│   │   ├── backtest.py
│   │   └── history.py
│   ├── services/
│   │   ├── scoring.py
│   │   ├── realtime.py
│   │   ├── report_parser.py
│   │   ├── ai_runner.py
│   │   └── eastmoney.py
│   ├── static/
│   └── templates/
└── runtime/
    ├── jobs/
    └── logs/
```

---

## 十二、优先整理建议

低风险优化顺序：

1. 保留现有功能，不立即大规模拆结构。
2. 已把 `ai_job_*.log`、`ai_runner_*.py` 移到 `runtime/jobs/`。
3. 已让报告结果页指标优先从 HTML 报告解析，避免和 HTML 报告不一致。
4. 已把 `web_stock_picker/app.py` 中的报告解析函数拆到 `web_stock_picker/services/report_parser.py`。
5. 已把东方财富 / 新浪数据源函数拆到 `web_stock_picker/services/eastmoney.py`。
6. 已把股票评分函数拆到 `web_stock_picker/services/scoring.py`。
7. 最后再整理 `app/model/` 中带空格的文件名。

本轮整理后，`web_stock_picker/app.py` 已从约 843 行降到约 353 行，保留原有 WSGI 路由和前端内联结构，降低一次性大改风险。

---

## 十三、结论

当前项目结构可运行，但已经进入：

```text
单文件 Web 层偏重
运行时文件混入源码
指标逻辑重复
原始 AI 项目与 Web 层强耦合
```

的阶段。

最关键维护点：

```text
web_stock_picker/app.py                       # Web 页面和 API
app/main.py                                   # 原始 AI 分析入口
app/backtest.py                               # 回测核心
app/report_chart/report_generator.py          # 报告生成
app/report_chart/report_config.py             # 报告输出目录
app/stock_data/                               # 历史行情数据
app/results/reports/                          # HTML 报告
app/model/GaiNian stock.py                    # 东方财富概念资金流
```

建议下一步优先处理：

1. 统一报告结果页和 HTML 报告指标。
2. 清理 / 迁移 `ai_job_*.log`、`ai_runner_*.py` 运行时文件。
3. 拆分 `web_stock_picker/app.py`，降低后续维护成本。
