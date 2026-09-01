# 更新日志 (CHANGELOG)

## 2026-08-18 — 报告结果页默认排序调整

### 修改内容

- **默认排序**：报告结果页面的默认排序规则改为「修改时间 降序」（最新生成的报告排在最前）为第一优先级，其余指标依次作为并列时的次级排序：
  1. 修改时间 降序（原为最后一项，现提升到第一位）
  2. 风险等级 升序
  3. 夏普比率 降序
  4. 胜率 降序
  5. Total Return 降序
  6. 波动率 升序
  7. 风险评分 升序
- 页面状态栏文案同步更新为：「默认排序：修改时间↓ 风险等级↑ 夏普比率↓ 胜率↓ Total Return↓ 波动率↑ 风险评分↑」。
- 手动点击列头排序的行为不变（点击后切换为手动排序，刷新页面恢复默认）。

### 涉及文件

| 文件 | 变更 |
|------|------|
| `web_stock_picker/app.py` | `defaultReportCompare` 排序优先级调整；状态栏文案更新 |

### 验证结果

- JS 片段 `node --check` 语法通过
- `python -m py_compile web_stock_picker/app.py` 通过

## 2026-05-28 ~ 2026-06-01 — 全项目性能优化与代码质量提升

### 性能优化（P0 — Web 页面速度）

#### 1. 并行行情请求
- `scoring.py`: 新增 `batch_fetch_realtime_quotes()`，使用 `ThreadPoolExecutor(max_workers=10)` 并行获取 84 只股票实时行情。
- **效果**: 84 次串行 HTTP 25~126 秒 → 并行 1~3 秒。

#### 2. 三级缓存体系
- **CSV 解析缓存** (`scoring.py`): `_csv_cache` 按文件 mtime+size 失效，避免重复读取 170,000 行。
- **目录扫描缓存** (`scoring.py`): `_paths_cache` 按 DATA_DIR mtime 失效，跳过每次全量 glob。
- **30 秒 TTL 结果缓存** (`scoring.py`): `_items_cache` 避免热刷新全量重算，返回深拷贝副本。
- **线程安全** (`scoring.py`): 添加 `_csv_cache_lock` 保护多线程缓存读写。

#### 3. 多线程 WSGI 服务器
- `app.py`: `make_server` → `ThreadedWSGIServer(ThreadingMixIn + WSGIServer)`。
- **效果**: 静态文件请求不再被 API 长请求阻塞，即时响应。

#### 4. 报告列表跳过无用重算
- `report_parser.py`: `report_metrics_from_name()` 新增 `html_only=True` 参数，跳过 `find_item()`（读 CSV + HTTP 取行情 + 全量指标计算）。
- **效果**: 报告页刷新 17~86 秒 → 0.3~1 秒。（HTML 文件已包含全部指标数据）

#### 5. 静态资源从文件读取
- `app.py`: 删除 3 个内联大字符串（INDEX_HTML 6KB + APP_JS 9KB + STYLE_CSS 3KB），改为从 `templates/index.html` 和 `static/` 目录直接读取。
- **效果**: 修改前端无需修改 Python 代码，文件更清晰。

### 性能优化（P1 — AI 管道启动）

#### 6. ML 库惰性导入
- `factory.py`: 将 5 个重库（TensorFlow/Keras、LightGBM、CatBoost、SARIMAX、ARIMA）从模块顶层 try/except 改为各训练方法内按需 `import`。删除未使用的 `pykalman` import。`import xgboost` 移至 `_train_xgboost()` 内。
- **效果**: `main.py` 启动时间 5~20 秒 → 近 0 秒。

### Bug 修复

| 问题 | 位置 | 修复 |
|------|------|------|
| `market_regise` 拼写错误 → NameError | `main.py:307` | 修正为 `market_regime` |
| 模型训练两次，第一次结果被覆盖 | `main.py:276-283` | 删除重复代码 |
| 缩进错位导致语法错误 | `main.py` | 修正 try 缩进 |
| 重复的判断和打印语句 | `main.py` | 去重 |

### 新增功能 — 报告页行业显示

#### 7. 报告列表行业列
- `eastmoney.py`: 新增 `_warm_industry_cache()` 在模块导入时从东方财富 API 同步预热约 1500 只 A 股行业数据（约 4 秒），新增 `fetch_stock_industries()` 纯缓存查询。
- `report_parser.py`: 新增 `get_stock_industry()` 两级查询：EM API 行业数据 → 关键词匹配备用，去掉板块代码后缀（主板(深)/主板(沪)/创业板）。
- `app.py`: `reports_payload()` 注入 `industry` 字段；HTML 表头插入「行业」列（报告名称 ↔ 风险等级之间）；JS 渲染 + 排序适配。
- **效果**: 90 个报告中 67 个显示真实 A 股行业分类（光学光电子、白色家电、半导体、化学制品...）。

### 清理

| 清理项 | 数量 |
|--------|:----:|
| AI runner 生成脚本 `runtime/jobs/ai_runner_*.py` | 44 个 |
| AI job 日志 `runtime/jobs/ai_job_*.log` | 44 个 |
| `app.py.bak` 备份文件 | 2 个 |
| `__pycache__` 目录 | 全部清空 |
| `*.pyc` 缓存文件 | 全部删除 |

### 配置精简
- `config.yaml`: 删除未使用的三个配置节（`correction`、`probabilistic`、`monitoring`），共 53 行。
- 这些配置仅被 `config_loader.py` 解析为内存对象，从未被业务逻辑读取。

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `web_stock_picker/services/scoring.py` | 并行 HTTP、三级缓存、线程锁 |
| `web_stock_picker/app.py` | 多线程 WSGI、静态文件化、报告行业列、路由内联字符串删除 |
| `web_stock_picker/services/report_parser.py` | `html_only` 参数、`get_stock_industry()` 行业查询、`_INDUSTRY_KEYWORDS` 映射 |
| `web_stock_picker/services/eastmoney.py` | `_warm_industry_cache()`、`fetch_stock_industries()` 行业数据预热与缓存 |
| `app/model/factory.py` | 5 个 ML 库惰性导入、pykalman 死代码删除 |
| `app/main.py` | `market_regime` typo 修复、模型训练去重、缩进修复 |
| `app/config.yaml` | 删除 3 个死配置节（-53 行） |

## 2026-05-28 — 量化因子模型

### 新增功能

#### 1. 量化因子模型 (`quant_factor`)
- 在 `app/model/factory.py` 中新增 `_train_quant_factor()` — 从 OHLCV 数据计算 16+ 个标准化量化因子（动量、波动、趋势强度、偏度峰度、均线交叉、价格位置、Amihud 非流动性、量价背离、MFI 资金流量），使用 RandomForestRegressor 训练预测未来收益率。
- 在 `app/model/simulator.py` 中新增 `_simulate_quant_factor()` — 实时滚动构建因子向量，经 StandardScaler 标准化后调用模型预测，生成未来价格路径。
- 已注册至全链路：5 种市场状态的选择配置、权重配置、集成融合、模型评估、组合置信度计算。
- `app/config.yaml` 新增 `quant_factor` 独立配置节（`n_estimators`, `max_depth`, `n_factors`, `window_size` 等）。

#### 2. 因子可解释性
- 训练日志输出 Top-5 因子重要性排名，便于分析哪些因子驱动当前预测。
- 示例输出：`Top-5 因子: momentum_60d(0.147), volatility_10d(0.124), price_volume_divergence(0.108), kurtosis_20d(0.087), trend_strength(0.082)`

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `app/model/factory.py` | 新增 `_train_quant_factor()` (~150 行)；注册到训练管道、选择配置、权重配置、评估方法 |
| `app/model/simulator.py` | 新增 `_simulate_quant_factor()` (~100 行)；注册到 BACKTEST_RESULTS、market_state_mapping、ModelConfigManager、model_simulation_methods |
| `app/config.yaml` | 新增 `quant_factor` 配置节；ensemble 权重引用 |

### 验证结果
- 测试标的：003021 兆威机电（1322 条日线数据）
- 训练集 R²: 0.6354
- Top-3 主导因子：动量(60日) > 波动率(10日) > 量价背离
- 全链路模拟通过，`quant_factor` 正常参与 ensemble 集成

## 2026-05-22 — 微信通知集成

### 新增功能

#### 1. 定时训练结果微信通知
- 在 `app.py` 中新增 `rank_finished_ai_results(job)` — 从已完成的任务结果中提取股票代码、名称、综合评分、信号、价格、涨跌幅、K线评分、模拟年化收益等字段，去重后按评分降序排序，返回前 5 名。
- 在 `app.py` 中新增 `format_ai_top5_message(job)` — 将排名前 5 的股票格式化为竖列文本消息。
- 在 `app.py` 中新增 `save_schedule_notification(job)` — 将定时任务完成后的前 5 结果写入 `runtime/schedule_notify/` 目录下的 JSON 通知文件。
- 在 `app.py` 中新增 `maybe_send_scheduled_weixin(job)` — 在 `run_ai_job` 和 `run_ai_batch_job` 完成阶段调用，如果任务标记为 `scheduled` 且尚未处理，则写入通知文件。

#### 2. 通知队列 API
- `GET /api/ai/schedule_notify` — 查看所有通知记录（含字段：`job_id`, `mode`, `total`, `completed`, `top5_rank`, `weixin_message`, `notified` 状态）。
- `POST /api/ai/schedule_notify_done` — 标记指定通知文件为"已发送"。

#### 3. 外部轮询 + 微信发送（宿主机侧）
- 新增脚本 `/opt/data/scripts/stock_schedule_notify.py` — 每 5 分钟轮询 `GET /api/ai/schedule_notify`，发现未发送的通知则调用微信直连 API 发送，成功后标记为已发送。
- Hermes 定时任务 `stock-weixin-notify`（`*/5 * * * *`，无代理模式）驱动上述脚本。
- 微信接收目标：`o9cq80zU7znZ3Bh_2RN5C0w3YIjo@im.wechat`

### 配置文件
- 定时任务配置：`runtime/ai_schedule.json`（当前：`enabled: true, time: "18:30", mode: "industry_10d"`）

### 文件变更清单

| 文件 | 变更 |
|------|------|
| `/vol1/1000/docker/stock-analysis-docker/web_stock_picker/app.py` | 新增通知队列函数、2个API路由 |
| `/vol1/1000/docker/stock-analysis-docker/web_stock_picker/runtime/schedule_notify/` | 新建目录，存放通知JSON文件（自动创建） |
| 宿主机 `/opt/data/scripts/stock_schedule_notify.py` | 新增外部轮询+微信发送脚本 |
| Hermes cron job `stock-weixin-notify` | 新增定时任务，每5分钟轮询 |

### 注意事项
- 容器内无 Hermes 运行环境，微信发送由宿主机侧脚本完成
- 容器侧只需写入 JSON 通知文件，无需安装额外依赖
- 如需更换微信接收者，修改 `gateway.platforms.weixin` 调用中的 `chat_id` 参数
