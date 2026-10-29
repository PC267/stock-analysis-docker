# 更新日志 - 2026-05-21

## 项目

/vol1/1000/docker/stock-analysis-docker

## 本次优化范围

Web 选股系统：http://192.168.3.17:9119/

## 1. 报告结果页优化

### 修改目标

报告结果根据股票代码去重，只保留同一股票代码中修改时间最新的 HTML 报告；分析建议全部显示中文。

### 已完成

- /api/reports 自动扫描 HTML 报告。
- 按 6 位股票代码分组。
- 每个股票代码只保留修改时间最新的 1 个 HTML 报告。
- 旧的重复 HTML 报告会被物理删除。
- 无法识别股票代码的 HTML 报告不会删除。
- 修复 report_roots 重叠导致同一路径被重复扫描的问题，避免误删最新文件。
- 英文风险等级映射为中文：
  - Extreme Risk -> 极高
  - High Risk -> 高
  - Medium Risk -> 中
  - Low Risk -> 低
  - Very Low Risk -> 极低
- 英文分析建议自动翻译为中文，例如：
  - Extremely high risk, recommend avoiding operations -> 风险极高：建议暂时回避，等待风险释放后再观察。
  - Recommend strict position control, set stop loss -> 风险较高：建议严格控制仓位，并设置止损位。

### 涉及文件

- web_stock_picker/app.py
- web_stock_picker/services/report_parser.py

## 2. 选股评分页优化

### 修改目标

选股评分页面也按股票代码去重，只保留同一股票代码中修改时间最新的数据文件。

### 已完成

- /api/stocks 自动扫描 app/stock_data 下的 *_baostock.csv。
- 按 6 位股票代码分组。
- 每个股票代码只保留修改时间最新的 1 个 CSV 数据文件。
- 旧的重复 CSV 文件会被物理删除。
- 选股评分只基于最新 CSV 数据计算。
- /api/stocks 返回增加：
  - deleted_duplicates
  - deleted_duplicate_items

### 涉及文件

- web_stock_picker/services/scoring.py

## 3. 验证结果

### 服务状态

- /health 正常
- Docker 容器 web-stock-picker 已重启
- Python 编译检查通过

### 报告结果页验证

- /api/reports 正常
- 重复股票代码报告：0
- 英文分析建议：0

### 选股评分页验证

- /api/stocks 正常
- 当前选股数量：42
- 重复股票代码：0
- 本次删除重复 CSV：0

## 4. 当前行为说明

### 报告结果页

刷新“报告结果”或调用 /api/reports 时，会自动执行 HTML 报告去重和旧文件清理。

### 选股评分页

刷新“选股评分”或调用 /api/stocks 时，会自动执行 CSV 数据文件去重和旧文件清理。

## 5. 注意事项

- 物理清理是自动执行的，不只是前端隐藏。
- 删除逻辑只针对能识别出 6 位股票代码的文件。
- 保留标准是文件修改时间最新。
- 报告结果页清理 HTML 文件。
- 选股评分页清理 *_baostock.csv 文件。
