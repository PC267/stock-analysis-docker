# stock-analysis-docker

A股 Web 选股、AI 训练、量化回测与报告管理系统。Python 3.11, pure WSGI (wsgiref), Docker 部署.

## Project

- **Stack:** Python 3.11, pure WSGI (no Flask), pandas/numpy/scikit-learn/tensorflow-cpu/lightgbm/catboost/xgboost/baostock
- **Web entry:** `web_stock_picker/app.py` — WSGI server on `0.0.0.0:9119`
- **AI analysis:** `app/main.py` → `analyze_single_stock()`
- **Backtest:** `app/backtest.py`
- **Runner:** `app/run.py` (wraps main.py with path setup)
- **Config:** `app/config.yaml` + `app/config_loader.py`

## Commands

| Action | Command |
|---|---|
| Build (full image) | `docker compose build` |
| Run service | `docker compose up -d` |
| Local web dev | `python web_stock_picker/app.py` |
| AI analysis | `python app/main.py` |
| Backtest | `python app/backtest.py` |
| Python deps | `app/requirements.txt` |

## Architecture

```
web_stock_picker/app.py  (WSGI → API routes)
  ├── services/scoring.py       — 选股评分 (MA/RSI/动量/回撤/波动率/K线模拟)
  ├── services/eastmoney.py     — 东方财富 / 新浪实时行情数据源
  ├── services/report_parser.py — HTML 报告解析
  ├── app/main.py               — (subprocess) 重型 AI 多模型分析
  └── app/backtest.py           — (subprocess) 回测
```

Three data sources: local baostock CSV (`app/stock_data/*_baostock.csv`), Tencent/Sina real-time quotes, East Money industry fund flow.

## Key modules

- `app/data/` — data fetching & processing (baostock, multi-source)
- `app/features/` — technical indicators & market features
- `app/model/` — factory.py (model factory), simulator.py (prediction), evaluator.py, GaiNian stock.py (concept flow)
- `app/report_chart/` — report_generator.py, chart_generator.py, report_config.py
- `app/stock_data/` — historical CSV data dir
- `app/results/reports/` — generated reports (HTML/PNG/XLSX)

## Conventions

- **Encoding:** `# -*- coding: utf-8 -*-` everywhere
- **Web framework:** pure stdlib WSGI (`wsgiref.simple_server`), inline HTML/JS/CSS in `app.py`
- **Entry guard:** `if __name__ == '__main__':` for all entry points
- **CSV naming:** `{code}_{name}_baostock.csv`
- **Report naming:** `{code}_{name}_HTML报告_{timestamp}.html`
- **Config:** YAML + dataclass config loader in `config_loader.py`
- **GPU/TF:** `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'` to disable GPU
- **Heavy AI:** launched via `subprocess.Popen` generating temporary runner scripts
- **Model filenames** contain spaces (e.g. `GaiNian stock.py`) — import via `importlib`
- **Docker:** `Dockerfile` (lightweight, filters heavy deps) / `web_stock_picker/Dockerfile.full` (full AI)
- **Env vars:** `WEB_STOCK_PICKER_PORT` (default 9119), `REPORT_OUTPUT_DIR` (default `/work/app/results/reports`)
- **Runtime files** (ai_job_*.log, ai_runner_*.py) → `web_stock_picker/runtime/jobs/`

## Notes

(Add quick notes here as needed — e.g. common gotchas, pending TODOs, deployment tips.)
