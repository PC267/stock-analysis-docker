# Web 选股系统

轻量 Web 服务，直接读取 `app/stock_data/*_baostock.csv`，计算 MA、RSI、量比、20日动量、最大回撤、波动率和综合评分。

不依赖损坏的 `app/venv`，不需要 Flask 等第三方库，仅使用 Python 标准库。

## 运行

```bash
cd /vol1/1000/docker/stock-analysis-docker
python3 web_stock_picker/app.py
```

访问：`http://192.168.3.17:9119/`

API：
- `GET /health`
- `GET /api/stocks?strategy=balanced&limit=50&keyword=`

策略：`balanced`、`momentum`、`volume_breakout`、`oversold`
