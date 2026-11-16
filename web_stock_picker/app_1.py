#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, math, os, re, urllib.request, urllib.parse, subprocess, uuid, threading, time, sys, mimetypes, io, zipfile, importlib.util, contextlib
from datetime import datetime, date, timedelta
from pathlib import Path
from statistics import pstdev
from urllib.parse import parse_qs, urlparse, quote, unquote
from wsgiref.simple_server import make_server

from services.scoring import to_float, avg, ma, rsi, max_drawdown, volatility, normalize_code, parse_name, read_stock, fetch_sina_quote, fetch_tencent_quote, fetch_realtime_quote, merge_realtime_row, kline_simulation, score_stock, find_item, load_items, apply_strategy, stock_payload, delete_stock_payload
from services.eastmoney import em_json, datacenter_json, local_industry_batch, sina_market_rows, alt_top_industries, alt_industry_stocks, eastmoney_top_industries, eastmoney_industry_stocks, fallback_industry_batch, load_gainian_module, resolve_industry_batch
from services.report_parser import report_roots, safe_report_path, extract_html_metrics, extract_html_total_return, report_display_name, report_metrics_from_name, report_stock_code, get_stock_industry
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'app' / 'stock_data'
PORT = int(os.environ.get('WEB_STOCK_PICKER_PORT', '9119'))

# ---- Basic Auth 配置 ----
# 凭据通过环境变量注入；缺省 admin/admin（仅供本机/内网测试，生产务必修改）
AUTH_USER = os.environ.get('WEB_STOCK_PICKER_USER', 'admin')
AUTH_PASS = os.environ.get('WEB_STOCK_PICKER_PASS', 'admin')
AUTH_REALM = 'Web Stock Picker'
# 需要认证的路径前缀：API 与报告（含删除/训练/回测等破坏性操作）
AUTH_PREFIXES = ('/api', '/reports')

INDEX_HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Web选股系统</title><link rel="stylesheet" href="/static/style.css"></head><body><div class="wrap"><header><h1>Web选股系统</h1><p>选股 / 回测 / 重型AI训练 / 报告</p></header><nav class="tabs"><button data-tab="picker" class="active">选股评分</button><button data-tab="backtest">回测</button><button data-tab="heavyai">重型AI训练</button><button data-tab="reports">报告结果</button></nav><section id="picker" class="panel active"><div class="toolbar"><select id="strategy"><option value="balanced">综合评分</option><option value="momentum">趋势动量</option><option value="volume_breakout">放量突破</option><option value="oversold">超跌修复</option></select><input id="keyword" placeholder="股票代码/名称"><input id="limit" type="number" value="50"><button id="refresh">刷新</button><button id="downloadHistory">更新历史数据</button><span id="status"></span></div><div class="tablebox"><table><thead><tr><th data-sort="rank">排名 <span class="sort-arrows"><button onclick="sortStocks('rank','asc')">&#9650;</button><button onclick="sortStocks('rank','desc')">&#9660;</button></span></th><th data-sort="code">代码 <span class="sort-arrows"><button onclick="sortStocks('code','asc')">&#9650;</button><button onclick="sortStocks('code','desc')">&#9660;</button></span></th><th data-sort="name">名称 <span class="sort-arrows"><button onclick="sortStocks('name','asc')">&#9650;</button><button onclick="sortStocks('name','desc')">&#9660;</button></span></th><th data-sort="signal">信号 <span class="sort-arrows"><button onclick="sortStocks('signal','asc')">&#9650;</button><button onclick="sortStocks('signal','desc')">&#9660;</button></span></th><th data-sort="score">评分 <span class="sort-arrows"><button onclick="sortStocks('score','asc')">&#9650;</button><button onclick="sortStocks('score','desc')">&#9660;</button></span></th><th data-sort="price">最新价 <span class="sort-arrows"><button onclick="sortStocks('price','asc')">&#9650;</button><button onclick="sortStocks('price','desc')">&#9660;</button></span></th><th data-sort="change_pct">涨跌幅 <span class="sort-arrows"><button onclick="sortStocks('change_pct','asc')">&#9650;</button><button onclick="sortStocks('change_pct','desc')">&#9660;</button></span></th><th data-sort="ma">MA5/20/60 <span class="sort-arrows"><button onclick="sortStocks('ma','asc')">&#9650;</button><button onclick="sortStocks('ma','desc')">&#9660;</button></span></th><th data-sort="rsi14">RSI <span class="sort-arrows"><button onclick="sortStocks('rsi14','asc')">&#9650;</button><button onclick="sortStocks('rsi14','desc')">&#9660;</button></span></th><th data-sort="volume_ratio">量比 <span class="sort-arrows"><button onclick="sortStocks('volume_ratio','asc')">&#9650;</button><button onclick="sortStocks('volume_ratio','desc')">&#9660;</button></span></th><th data-sort="momentum20">20日动量 <span class="sort-arrows"><button onclick="sortStocks('momentum20','asc')">&#9650;</button><button onclick="sortStocks('momentum20','desc')">&#9660;</button></span></th><th data-sort="drawdown">最大回撤 <span class="sort-arrows"><button onclick="sortStocks('drawdown','asc')">&#9650;</button><button onclick="sortStocks('drawdown','desc')">&#9660;</button></span></th><th data-sort="kline_score">K线模拟 <span class="sort-arrows"><button onclick="sortStocks('kline_score','asc')">&#9650;</button><button onclick="sortStocks('kline_score','desc')">&#9660;</button></span></th><th>操作</th><th data-sort="data_source">数据源 <span class="sort-arrows"><button onclick="sortStocks('data_source','asc')">&#9650;</button><button onclick="sortStocks('data_source','desc')">&#9660;</button></span></th><th data-sort="tags">标签 <span class="sort-arrows"><button onclick="sortStocks('tags','asc')">&#9650;</button><button onclick="sortStocks('tags','desc')">&#9660;</button></span></th></tr></thead><tbody id="tbody"></tbody></table></div></section><section id="backtest" class="panel"><h2>回测模块</h2><div class="toolbar"><input id="backtestCode" placeholder="股票代码，可留空"><input id="backtestDays" type="number" value="20"><button id="backtestRun">运行 backtest.py 回测</button></div><pre id="backtestResult" class="result"></pre></section><section id="heavyai" class="panel"><h2>重型AI训练</h2><div class="toolbar heavyai-toolbar"><input id="aiCode" class="ai-code" placeholder="股票代码"><input id="aiName" class="ai-name" placeholder="股票名称"><input id="aiDays" class="ai-days" type="number" value="20" title="模拟天数"><select id="aiMode" class="ai-mode"><option value="single">单股训练</option><option value="custom_batch">自定义批量</option><option value="industry_5d">行业资金5日前三</option><option value="industry_10d">行业资金10日前三</option></select><input id="aiLimit" class="ai-limit" type="number" value="3" title="每个行业训练股票数"><button id="aiRun">启动训练</button><button id="aiStatusBtn">刷新状态</button></div><textarea id="aiProjects" class="ai-projects" placeholder="自定义训练项目：每行一个，如：300088 长信科技&#10;600519 贵州茅台"></textarea><div class="toolbar ai-schedule"><label class="schedule-check"><input id="aiScheduleEnabled" type="checkbox"> 定时开启</label><input id="aiScheduleTime" class="ai-time" type="time" value="09:30"><button id="aiScheduleSave">保存定时</button><span id="aiScheduleStatus"></span></div><pre id="aiResult" class="result"></pre></section><section id="reports" class="panel"><h2>报告结果</h2><div class="toolbar"><button id="reportsLoad">刷新报告列表</button><span id="reportsStatus"></span></div><div class="tablebox report-tablebox"><table class="report-table"><thead><tr><th data-sort="idx">序号 <span class="sort-arrows"><button onclick="sortReports('idx','asc')">&#9650;</button><button onclick="sortReports('idx','desc')">&#9660;</button></span></th><th data-sort="name">报告名称 <span class="sort-arrows"><button onclick="sortReports('name','asc')">&#9650;</button><button onclick="sortReports('name','desc')">&#9660;</button></span></th><th data-sort="industry">行业</th><th data-sort="risk_level">风险等级 <span class="sort-arrows"><button onclick="sortReports('risk_level','asc')">&#9650;</button><button onclick="sortReports('risk_level','desc')">&#9660;</button></span></th><th data-sort="risk_score">风险评分 <span class="sort-arrows"><button onclick="sortReports('risk_score','asc')">&#9650;</button><button onclick="sortReports('risk_score','desc')">&#9660;</button></span></th><th data-sort="volatility">波动率 <span class="sort-arrows"><button onclick="sortReports('volatility','asc')">&#9650;</button><button onclick="sortReports('volatility','desc')">&#9660;</button></span></th><th data-sort="max_drawdown">Total Return <span class="sort-arrows"><button onclick="sortReports('max_drawdown','asc')">&#9650;</button><button onclick="sortReports('max_drawdown','desc')">&#9660;</button></span></th><th data-sort="sharpe">夏普比率 <span class="sort-arrows"><button onclick="sortReports('sharpe','asc')">&#9650;</button><button onclick="sortReports('sharpe','desc')">&#9660;</button></span></th><th data-sort="win_rate">胜率 <span class="sort-arrows"><button onclick="sortReports('win_rate','asc')">&#9650;</button><button onclick="sortReports('win_rate','desc')">&#9660;</button></span></th><th>操作</th><th data-sort="mtime">修改时间 <span class="sort-arrows"><button onclick="sortReports('mtime','asc')">&#9650;</button><button onclick="sortReports('mtime','desc')">&#9660;</button></span></th><th data-sort="advice">分析建议 <span class="sort-arrows"><button onclick="sortReports('advice','asc')">&#9650;</button><button onclick="sortReports('advice','desc')">&#9660;</button></span></th></tr></thead><tbody id="reportsBox"></tbody></table></div></section></div><script src="/static/app.js"></script></body></html>'''
APP_JS = '''function q(id){return document.getElementById(id)}document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');q(b.dataset.tab).classList.add('active')});async function gj(u){return await (await fetch(u)).json()}async function pj(u,o){return await (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})).json()}
let stockItems=[];let stockSort={key:'rank',dir:'asc'};function stockSortValue(x,k){if(k==='ma')return Number(x.ma20??-999999);if(k==='tags')return (x.tags||[]).join(',');if(['rank','score','price','change_pct','rsi14','volume_ratio','momentum20','drawdown','kline_score'].includes(k))return Number(x[k]??-999999);if(k==='data_source')return x.data_fresh?'实时':'CSV';return String(x[k]??'')}function sortStocks(k,d){stockSort={key:k,dir:d};renderStocks()}function renderStocks(){let arr=[...stockItems];arr.sort((a,b)=>{let av=stockSortValue(a,stockSort.key),bv=stockSortValue(b,stockSort.key);let r=(typeof av==='number'&&typeof bv==='number')?av-bv:String(av).localeCompare(String(bv),'zh-CN');return stockSort.dir==='asc'?r:-r});let tb=q('tbody');tb.innerHTML='';for(let i=0;i<arr.length;i++){let x=arr[i],tr=document.createElement('tr');tr.innerHTML=`<td>${i+1}</td><td class="code">${x.code}</td><td>${x.name}</td><td><span class="sig">${x.signal}</span></td><td class="score">${x.score}</td><td>${x.price}</td><td class="${x.change_pct>=0?'red':'green'}">${x.change_pct}%</td><td>${x.ma5??'-'} / ${x.ma20??'-'} / ${x.ma60??'-'}</td><td>${x.rsi14??'-'}</td><td>${x.volume_ratio??'-'}</td><td>${x.momentum20??'-'}%</td><td>${x.drawdown??'-'}%</td><td>${x.kline_score??'-'} / ${x.sim_return??'-'}%</td><td><button class="danger" onclick="deleteStock('${x.code}','${(x.name||'').replace(/'/g,\"\\'\")}')">删除</button></td><td>${x.data_fresh?'实时':'CSV'}</td><td>${(x.tags||[]).map(t=>`<span class="tag">${t}</span>`).join('')}</td>`;tb.appendChild(tr)}}async function deleteStock(code,name){if(!confirm('确认删除个股：'+code+' '+name+' ?'))return;let r=await fetch('/api/stocks/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});let d=await r.json();if(!d.ok){alert(d.error||'删除失败');return}await loadStocks()}async function loadStocks(){let s=q('strategy').value,k=q('keyword').value,l=q('limit').value||50;q('status').textContent='加载中...';let d=await gj(`/api/stocks?strategy=${s}&limit=${l}&keyword=${encodeURIComponent(k)}`);stockItems=d.items||[];renderStocks();q('status').textContent=`${d.updated_at} · ${d.count}只 · 排序 ${stockSort.key} ${stockSort.dir==='asc'?'↑':'↓'}`}q('refresh').onclick=loadStocks;q('downloadHistory').onclick=async()=>{let k=q('keyword').value.trim();if(!k&&!confirm('未输入股票代码/名称，将更新全部CSV，可能耗时较长，继续？'))return;q('status').textContent='历史数据更新中...';try{let d=await pj('/api/history/update',{keyword:k});q('status').textContent=`更新完成：${d.updated} 个，失败 ${d.failed}`;await loadStocks()}catch(e){q('status').textContent='更新失败：'+e}};q('strategy').onchange=loadStocks;q('keyword').onkeydown=e=>{if(e.key==='Enter')loadStocks()};
q('backtestRun').onclick=async()=>q('backtestResult').textContent=JSON.stringify(await pj('/api/backtest',{code:q('backtestCode').value.trim(),days:q('backtestDays').value}),null,2);let lastJob='';function aiPayload(){return {code:q('aiCode').value.trim(),name:q('aiName').value.trim(),days:q('aiDays').value,mode:q('aiMode').value,limit:q('aiLimit').value,projects:q('aiProjects').value}}function syncAiProjectBox(){q('aiProjects').style.display=q('aiMode').value==='custom_batch'?'block':'none'}q('aiMode').onchange=syncAiProjectBox;syncAiProjectBox();async function loadAiSchedule(){try{let d=await gj('/api/ai/schedule');q('aiScheduleEnabled').checked=!!d.enabled;q('aiScheduleTime').value=d.time||'09:30';if(d.projects)q('aiProjects').value=d.projects;if(d.mode)q('aiMode').value=d.mode;if(d.days)q('aiDays').value=d.days;if(d.limit)q('aiLimit').value=d.limit;if(d.code)q('aiCode').value=d.code;if(d.name)q('aiName').value=d.name;syncAiProjectBox();q('aiScheduleStatus').textContent=d.enabled?`已开启：${d.time}`:'未开启'}catch(e){q('aiScheduleStatus').textContent='定时配置读取失败'}}q('aiRun').onclick=async()=>{q('aiResult').textContent='启动中...';let d=await pj('/api/ai/train',aiPayload());lastJob=d.job_id||'';q('aiResult').textContent=JSON.stringify(d,null,2)};q('aiStatusBtn').onclick=async()=>{q('aiResult').textContent=JSON.stringify(await gj('/api/ai/status?job_id='+encodeURIComponent(lastJob)),null,2)};q('aiScheduleSave').onclick=async()=>{let p=aiPayload();p.enabled=q('aiScheduleEnabled').checked;p.time=q('aiScheduleTime').value||'09:30';let d=await pj('/api/ai/schedule',p);q('aiScheduleStatus').textContent=d.enabled?`已开启：${d.time}`:'未开启';q('aiResult').textContent=JSON.stringify(d,null,2)};loadAiSchedule();
async function deleteReport(id,name){if(!confirm('确认删除报告：'+name+' ?'))return;let r=await fetch('/api/reports/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});let d=await r.json();if(!d.ok)alert(d.error||'删除失败');q('reportsLoad').click()}let reportItems=[];let reportSort={key:'__default',dir:'default'};function riskRank(v){return {'极低':0,'低':1,'中':2,'高':3,'极高':4,'Very Low Risk':0,'Low Risk':1,'Medium Risk':2,'High Risk':3,'Extreme Risk':4}[v]??0}function reportSortValue(x,k){if(k==='idx')return x._idx||0;if(k==='industry')return (x.industry||[]).join(',');if(k==='risk_level')return riskRank(x.risk_level);if(['risk_score','volatility','max_drawdown','sharpe','win_rate'].includes(k))return Number(x[k]??-999999);return String(x[k]||'')}function compareReport(a,b,k,d){let av=reportSortValue(a,k),bv=reportSortValue(b,k);let r=(typeof av==='number'&&typeof bv==='number')?av-bv:String(av).localeCompare(String(bv),'zh-CN');return d==='asc'?r:-r}function defaultReportCompare(a,b){return compareReport(a,b,'mtime','desc')||compareReport(a,b,'risk_level','asc')||compareReport(a,b,'sharpe','desc')||compareReport(a,b,'win_rate','desc')||compareReport(a,b,'max_drawdown','desc')||compareReport(a,b,'volatility','asc')||compareReport(a,b,'risk_score','asc')}function sortReports(k,d){reportSort={key:k,dir:d};renderReports()}function renderReports(){let arr=[...reportItems];arr.sort((a,b)=>reportSort.key==='__default'?defaultReportCompare(a,b):compareReport(a,b,reportSort.key,reportSort.dir));let tb=q('reportsBox');tb.innerHTML=arr.map((x,i)=>{let safeName=(x.name||'').replace(/'/g,"\'");let pct=v=>v==null?'-':(Number(v).toFixed(2)+'%');let num=v=>v==null?'-':Number(v).toFixed(2);return `<tr><td>${i+1}</td><td><a class="report-name" href="${x.url}" target="_blank">${x.display_name||x.name}</a></td><td>${(x.industry||[]).map(c=>'<span class="tag">'+c+'</span>').join('')}</td><td><span class="risk risk-${x.risk_level}">${x.risk_level||'-'}</span></td><td>${num(x.risk_score)}</td><td>${pct(x.volatility)}</td><td>${pct(x.max_drawdown)}</td><td>${num(x.sharpe)}</td><td>${pct(x.win_rate)}</td><td><div class="report-actions"><a class="open-report" href="${x.url}" target="_blank">打开</a><button class="danger" onclick="deleteReport('${x.id}','${safeName}')">删除</button></div></td><td>${x.mtime||''}</td><td class="advice-cell" title="${x.advice||''}">${x.advice||'-'}</td></tr>`}).join('');q('reportsStatus').textContent=reportSort.key==='__default'?`共 ${arr.length} 个报告 · 默认排序：修改时间↓ 风险等级↑ 夏普比率↓ 胜率↓ Total Return↓ 波动率↑ 风险评分↑`:`共 ${arr.length} 个报告 · 排序 ${reportSort.key} ${reportSort.dir==='asc'?'↑':'↓'}`}q('reportsLoad').onclick=async()=>{let d=await gj('/api/reports');reportItems=(d.items||[]).map((x,i)=>({...x,_idx:i+1}));reportSort={key:'__default',dir:'default'};renderReports()};loadStocks();'''
STYLE_CSS = '''body{margin:0;background:#0f172a;color:#e5e7eb;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Microsoft YaHei",sans-serif}.wrap{max-width:1480px;margin:0 auto;padding:24px}h1{margin:0 0 8px}.tabs,.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.tabs button,select,input,button{background:#111827;color:#e5e7eb;border:1px solid #334155;border-radius:8px;padding:9px 12px}button,.tabs button.active{cursor:pointer;background:#2563eb;border-color:#2563eb}.panel{display:none}.panel.active{display:block}.tablebox{overflow:auto;border:1px solid #1e293b;border-radius:12px}table{width:100%;border-collapse:collapse;background:#111827}th,td{padding:10px 12px;border-bottom:1px solid #1e293b;white-space:nowrap;text-align:left}th{background:#1e293b}.score{font-weight:700;color:#fbbf24}.sig,.tag{display:inline-block;background:#334155;border-radius:999px;padding:3px 8px}.tag{background:#1d4ed8;margin:2px}.red{color:#f87171}.green{color:#4ade80}.result{background:#020617;border:1px solid #1e293b;border-radius:12px;padding:14px;min-height:180px;white-space:pre-wrap;overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:14px}.card p{color:#94a3b8}.report-tablebox{max-height:70vh}.report-table th{position:sticky;top:0;z-index:1;vertical-align:middle}.report-name,.report-card a{color:#93c5fd;text-decoration:none}.report-name:hover,.report-card a:hover{text-decoration:underline}.advice-cell{max-width:360px;overflow:hidden;text-overflow:ellipsis}.report-actions{display:flex;gap:8px;align-items:center;margin-top:0}.open-report{background:#2563eb;color:#fff;border-radius:8px;padding:7px 10px;text-decoration:none}.danger{background:#dc2626;border-color:#dc2626;padding:7px 10px}.sort-arrows{display:inline-flex;flex-direction:column;gap:1px;margin-left:4px;vertical-align:middle}.sort-arrows button{font-size:9px;line-height:9px;padding:0 3px;border-radius:2px;background:#334155;border-color:#475569;color:#cbd5e1}.risk{display:inline-block;border-radius:999px;padding:2px 8px}.risk-低{background:#14532d;color:#86efac}.risk-中{background:#713f12;color:#fde68a}.risk-高,.risk-极高{background:#7f1d1d;color:#fecaca}.risk-极低{background:#064e3b;color:#a7f3d0}.heavyai-toolbar{align-items:center;gap:8px}.heavyai-toolbar input,.heavyai-toolbar select{padding:7px 9px}.heavyai-toolbar .ai-code{width:95px}.heavyai-toolbar .ai-name{width:110px}.heavyai-toolbar .ai-days{width:70px}.heavyai-toolbar .ai-mode{width:150px}.heavyai-toolbar .ai-limit{width:70px}.ai-projects{display:none;width:520px;max-width:100%;height:82px;background:#111827;color:#e5e7eb;border:1px solid #334155;border-radius:8px;padding:9px 12px;box-sizing:border-box}.ai-schedule{align-items:center}.schedule-check{display:inline-flex;align-items:center;gap:6px;background:#111827;border:1px solid #334155;border-radius:8px;padding:8px 10px}.schedule-check input{padding:0}.ai-time{width:105px}'''



AI_JOBS = {}

# 报告列表缓存：60 秒 TTL，GET 请求只读缓存，避免每次全盘扫描 + 删除副作用
_REPORTS_CACHE = {'ts': 0.0, 'data': None}
_REPORTS_CACHE_TTL = 60
_REPORTS_CACHE_LOCK = threading.Lock()

# 报告清理线程：把 cleanup_duplicate_reports_by_code 从 GET 路径中移出，
# 改为周期性后台清理（清理会 unlink 文件，不应在只读请求里执行）
_REPORT_CLEANUP_INTERVAL = 600  # 10 分钟
_report_cleanup_started = False


def _reports_cleanup_loop():
    while True:
        try:
            cleanup_duplicate_reports_by_code()
        except Exception:
            pass
        time.sleep(_REPORT_CLEANUP_INTERVAL)


def start_report_cleanup_once():
    global _report_cleanup_started
    if not _report_cleanup_started:
        _report_cleanup_started = True
        t = threading.Thread(target=_reports_cleanup_loop, daemon=True)
        t.start()

# 回测并发信号量：限制同时运行的 backtest 子进程数，防止高并发下线程/内存堆积
_BACKTEST_SEM = threading.Semaphore(2)
# 历史数据更新并发信号量：限制同时运行的 baostock 更新任务
_HISTORY_UPDATE_SEM = threading.Semaphore(1)

AI_SCHEDULE_PATH = ROOT / 'web_stock_picker' / 'runtime' / 'ai_schedule.json'
AI_SCHEDULE_LOCK = threading.Lock()
AI_SCHEDULE_LAST_RUN = ''

def parse_custom_projects(projects):
    if isinstance(projects, list):
        lines = [str(x) for x in projects]
    else:
        lines = str(projects or '').splitlines()
    stocks = []
    seen = set()
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        m = re.search(r'\d{6}', raw)
        if not m:
            continue
        code = normalize_code(m.group(0))
        if not code or code in seen:
            continue
        name = (raw[:m.start()] + raw[m.end():]).strip(' ,，;；|\t-—:：')
        stocks.append({'code': code, 'name': name})
        seen.add(code)
    return stocks

def get_ai_schedule():
    cfg = {'enabled': False, 'time': '09:30', 'mode': 'single', 'code': '', 'name': '', 'days': 20, 'limit': 3, 'projects': ''}
    try:
        if AI_SCHEDULE_PATH.exists():
            data = json.loads(AI_SCHEDULE_PATH.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                cfg.update(data)
    except Exception:
        pass
    return cfg

def save_ai_schedule(cfg):
    AI_SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AI_SCHEDULE_LOCK:
        AI_SCHEDULE_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return cfg

def ai_schedule_payload(body, method='GET'):
    if method == 'GET':
        return get_ai_schedule()
    cfg = get_ai_schedule()
    cfg.update({
        'enabled': bool(body.get('enabled')),
        'time': body.get('time') or '09:30',
        'mode': body.get('mode') or 'single',
        'code': normalize_code(body.get('code','')),
        'name': body.get('name',''),
        'days': int(body.get('days') or 20),
        'limit': int(body.get('limit') or 3),
        'projects': body.get('projects',''),
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })
    if not re.match(r'^\d{2}:\d{2}$', cfg['time']):
        cfg['time'] = '09:30'
    return save_ai_schedule(cfg)

def trigger_schedule_job(cfg):
    mode = cfg.get('mode','single')
    body = dict(cfg)
    body['days'] = int(cfg.get('days') or 20)
    body['limit'] = int(cfg.get('limit') or 3)
    body['_scheduled'] = True
    body['_schedule_time'] = cfg.get('time') or ''
    body['_schedule_source'] = 'daily_timer'
    return ai_train_payload(body)

def _bj_now():
    """Return current Beijing time (UTC+8)."""
    return datetime.utcnow() + timedelta(hours=8)

def ai_scheduler_loop():
    global AI_SCHEDULE_LAST_RUN
    while True:
        try:
            cfg = get_ai_schedule()
            if cfg.get('enabled'):
                bj = _bj_now()
                now = bj.strftime('%H:%M')
                when = cfg.get('time') or '09:30'
                key = bj.strftime('%Y-%m-%d') + ' ' + when
                if now == when and AI_SCHEDULE_LAST_RUN != key:
                    AI_SCHEDULE_LAST_RUN = key
                    trigger_schedule_job(cfg)
        except Exception:
            pass
        time.sleep(30)

def start_ai_scheduler_once():
    t = threading.Thread(target=ai_scheduler_loop, daemon=True)
    t.start()


def ai_python_cmd():
    # 原 app/venv 损坏时，优先使用当前容器 Python；重型依赖缺失会在状态中返回错误
    venv = ROOT / 'app' / 'venv' / 'bin' / 'python'
    try:
        if venv.exists() and venv.stat().st_size > 0:
            return str(venv)
    except Exception:
        pass
    return sys.executable



def run_ai_batch_job(job_id, stocks, days):
    job = AI_JOBS[job_id]
    job['status']='running'; job['started_at']=datetime.now().strftime('%Y-%m-%d %H:%M:%S'); job['total']=len(stocks); job['completed']=0; job['results']=[]
    for st in stocks:
        sub_id = job_id + '_' + st['code']
        job['current'] = st
        try:
            AI_JOBS[sub_id] = {'job_id':sub_id,'code':st['code'],'name':st.get('name',''),'days':days,'status':'queued','created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mode':'行业批量子任务'}
            run_ai_job(sub_id, st['code'], st.get('name',''), days)
            sub = AI_JOBS.get(sub_id, {})
            job['results'].append({'code':st['code'],'name':st.get('name',''),'industry':st.get('industry',''),'status':sub.get('status'),'log_path':sub.get('log_path'),'returncode':sub.get('returncode')})
        except Exception as e:
            job['results'].append({'code':st['code'],'name':st.get('name',''),'industry':st.get('industry',''),'status':'failed','error':str(e)})
        job['completed'] += 1
    job['status']='completed'; job['finished_at']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    maybe_send_scheduled_weixin(job)

def run_ai_job(job_id, code, name, days):
    job = AI_JOBS[job_id]
    job['status'] = 'running'
    job['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    safe_code = normalize_code(code)
    safe_name = name or (find_item(safe_code)[0] or {}).get('name') or ('股票' + safe_code)
    jobs_dir = ROOT / 'runtime' / 'jobs'
    jobs_dir.mkdir(parents=True, exist_ok=True)
    script = jobs_dir / ('ai_runner_' + job_id + '.py')
    script.write_text("""
import os, sys, json, traceback, types
os.environ.setdefault('REPORT_OUTPUT_DIR', '/work/app/results/reports')
os.chdir(%r)
sys.path.insert(0, os.getcwd())
# TensorFlow 在当前 CPU/镜像仍会 SIGILL，保留 LSTM 自动跳过；LightGBM/CatBoost/GARCH 已启用
keras = types.ModuleType('tensorflow.keras')
models = types.ModuleType('tensorflow.keras.models')
layers = types.ModuleType('tensorflow.keras.layers')
class _DummySequential:
    def __init__(self, *a, **k): raise RuntimeError('TensorFlow unavailable in this CPU container')
models.Sequential = _DummySequential
layers.LSTM = layers.Dense = layers.Dropout = object
tf = types.ModuleType('tensorflow')
tf.keras = keras
keras.models = models
keras.layers = layers
sys.modules['tensorflow'] = tf
sys.modules['tensorflow.keras'] = keras
sys.modules['tensorflow.keras.models'] = models
sys.modules['tensorflow.keras.layers'] = layers
try:
    import main
    result = main.analyze_single_stock(%r, %r, %d, output_format='html', auto_open_html=False)
    print('AI_RESULT_JSON=' + json.dumps({'ok': True, 'result': str(result)}, ensure_ascii=False))
except Exception as e:
    traceback.print_exc()
    print('AI_RESULT_JSON=' + json.dumps({'ok': False, 'error': str(e)}, ensure_ascii=False))
""" % (str(ROOT / 'app'), safe_code, safe_name, int(days)))
    log_path = jobs_dir / ('ai_job_' + job_id + '.log')
    job['log_path'] = str(log_path)
    try:
        with open(log_path, 'w', encoding='utf-8') as log:
            p = subprocess.Popen([ai_python_cmd(), str(script)], cwd=str(ROOT / 'app'), stdout=log, stderr=subprocess.STDOUT, text=True)
            job['pid'] = p.pid
            rc = p.wait()
        job['returncode'] = rc
        job['status'] = 'completed' if rc == 0 else 'failed'
        job['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        maybe_send_scheduled_weixin(job)
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        job['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def ai_train_payload(body):
    days = int(body.get('days') or 20)
    mode = body.get('mode','single')
    job_id = uuid.uuid4().hex[:12]
    schedule_meta = {}
    if body.get('_scheduled') or body.get('scheduled'):
        schedule_meta = {'scheduled': True, 'schedule_time': body.get('_schedule_time') or body.get('time',''), 'schedule_source': body.get('_schedule_source','daily_timer')}
    if mode == 'custom_batch':
        stocks = parse_custom_projects(body.get('projects',''))
        if not stocks:
            return {'error':'请在自定义训练项目中每行填写一个6位股票代码'}
        AI_JOBS[job_id] = {'job_id':job_id,'days':days,'status':'queued','created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mode':'自定义批量训练','stocks':stocks,'total':len(stocks),'completed':0}
        if schedule_meta: AI_JOBS[job_id].update(schedule_meta)
        t = threading.Thread(target=run_ai_batch_job, args=(job_id, stocks, days), daemon=True)
        t.start()
        return AI_JOBS[job_id]
    if mode in ('industry_5d','industry_10d'):
        period = '5d' if mode == 'industry_5d' else '10d'
        limit = max(1, min(int(body.get('limit') or 3), 10))
        industries, stocks = resolve_industry_batch(period, limit)
        if not stocks:
            return {'error':'替代行情接口当前不可达，且本地候选为空'}
        AI_JOBS[job_id] = {'job_id':job_id,'days':days,'status':'queued','created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mode':'东方财富板块资金流%s批量训练'%('5日' if period=='5d' else '10日'),'source':'eastmoney_push2delay','industries':industries,'stocks':stocks,'total':len(stocks),'completed':0}
        if schedule_meta: AI_JOBS[job_id].update(schedule_meta)
        t = threading.Thread(target=run_ai_batch_job, args=(job_id, stocks, days), daemon=True)
        t.start()
        return AI_JOBS[job_id]
    code = normalize_code(body.get('code',''))
    if not code:
        return {'error':'请输入股票代码'}
    AI_JOBS[job_id] = {'job_id':job_id,'code':code,'name':body.get('name',''),'days':days,'status':'queued','created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mode':'main.py完整多模型训练'}
    if schedule_meta: AI_JOBS[job_id].update(schedule_meta)
    t = threading.Thread(target=run_ai_job, args=(job_id, code, body.get('name',''), days), daemon=True)
    t.start()
    return AI_JOBS[job_id]

def ai_status_payload(job_id):
    if not job_id:
        return {'jobs':list(AI_JOBS.values())[-10:]}
    job = AI_JOBS.get(job_id)
    if not job:
        return {'error':'任务不存在或服务已重启'}
    log_path = job.get('log_path')
    if log_path and os.path.exists(log_path):
        try:
            data = Path(log_path).read_text(encoding='utf-8', errors='ignore')
            job = dict(job)
            job['log_tail'] = data[-4000:]
        except Exception:
            pass
    return job


def rank_finished_ai_results(job):
    rows = []
    seen = set()
    # Get latest report files for each trained stock
    reports_dir = None
    for rp in report_roots():
        if rp.exists() and rp.joinpath().is_dir():
            reports_dir = rp
            break
    for r in job.get('results') or []:
        code = normalize_code(r.get('code',''))
        if not code or code in seen:
            continue
        seen.add(code)
        name = r.get('name','')
        # Find the latest HTML report file for this stock
        report_path = None
        if reports_dir:
            candidates = sorted(reports_dir.glob(f'{code}_*_HTML报告_*.html'), reverse=True)
            if candidates:
                report_path = candidates[0]
        # Parse report metrics
        risk_score = None
        sharpe = None
        win_rate = None
        total_return = None
        volatility = None
        risk_level = ''
        advice = ''
        if report_path:
            try:
                metrics = extract_html_metrics(report_path)
                if metrics:
                    risk_score = metrics.get('risk_score')
                    sharpe = metrics.get('sharpe')
                    win_rate = metrics.get('win_rate')
                    total_return = metrics.get('total_return')
                    volatility = metrics.get('volatility')
                    risk_level = metrics.get('risk_level', '')
                    advice = metrics.get('advice', '')
            except Exception:
                pass
        row = {'code': code, 'name': name, 'risk_score': risk_score, 'sharpe': sharpe,
               'win_rate': win_rate, 'total_return': total_return, 'volatility': volatility,
               'risk_level': risk_level, 'advice': advice,
               'status': r.get('status'), 'returncode': r.get('returncode')}
        rows.append(row)
    if not rows and job.get('code'):
        code = normalize_code(job.get('code',''))
        report_path = None
        if reports_dir:
            candidates = sorted(reports_dir.glob(f'{code}_*_HTML报告_*.html'), reverse=True)
            if candidates:
                report_path = candidates[0]
        risk_score = None; sharpe = None; win_rate = None; total_return = None
        risk_level = ''; advice = ''
        if report_path:
            try:
                metrics = extract_html_metrics(report_path)
                risk_score = metrics.get('risk_score')
                sharpe = metrics.get('sharpe')
                win_rate = metrics.get('win_rate')
                total_return = metrics.get('total_return')
                risk_level = metrics.get('risk_level', '')
                advice = metrics.get('advice', '')
            except Exception:
                pass
        rows.append({'code': code, 'name': job.get('name',''), 'risk_score': risk_score, 'sharpe': sharpe,
                     'win_rate': win_rate, 'total_return': total_return, 'risk_level': risk_level, 'advice': advice,
                     'status': job.get('status'), 'returncode': job.get('returncode')})
    # Sort by report metrics: sharpe descending (higher is better), then risk_score ascending (lower is better)
    def key(x):
        s = x.get('sharpe') if isinstance(x.get('sharpe'), (int, float)) else -999
        rs = -(x.get('risk_score') if isinstance(x.get('risk_score'), (int, float)) else 999)
        return (s, rs)
    return sorted(rows, key=key, reverse=True)[:5]

def format_ai_top5_message(job):
    rows = rank_finished_ai_results(job)
    title = '每日重型AI训练完成'
    lines = [title, '时间：' + datetime.now().strftime('%Y-%m-%d %H:%M'), '任务：' + str(job.get('mode',''))]
    if job.get('total') is not None:
        lines.append('数量：' + str(job.get('completed', 0)) + '/' + str(job.get('total')))
    lines.append('')
    lines.append('排名前5：')
    if not rows:
        lines.append('暂无可排名结果')
        return '\n'.join(lines)
    for i, x in enumerate(rows, 1):
        lines.append(str(i) + '. ' + (x.get('name') or '') + ' ' + x.get('code',''))
        lines.append('   评分：' + (str(x.get('score')) if x.get('score') is not None else '-'))
        lines.append('   信号：' + (x.get('signal') or '-'))
        lines.append('   最新价：' + (str(x.get('price')) if x.get('price') is not None else '-'))
        lines.append('   涨跌幅：' + ((str(x.get('change_pct')) + '%') if x.get('change_pct') is not None else '-'))
        lines.append('   K线模拟：' + (str(x.get('kline_score')) if x.get('kline_score') is not None else '-') + ' / ' + ((str(x.get('sim_return')) + '%') if x.get('sim_return') is not None else '-'))
        if i != len(rows):
            lines.append('')
    return '\n'.join(lines)[:1800]

SCHEDULE_NOTIFY_DIR = ROOT / 'web_stock_picker' / 'runtime' / 'schedule_notify'
SCHEDULE_NOTIFY_DIR.mkdir(parents=True, exist_ok=True)

def save_schedule_notification(job):
    msg = format_ai_top5_message(job)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    nf = SCHEDULE_NOTIFY_DIR / f'notify_{ts}_{job["job_id"]}.json'
    payload = {
        'job_id': job['job_id'],
        'mode': job.get('mode', ''),
        'schedule_time': job.get('schedule_time', ''),
        'total': job.get('total', 0),
        'completed': job.get('completed', 0),
        'top5_rank': rank_finished_ai_results(job),
        'weixin_message': msg,
        'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'notified': False
    }
    nf.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    job['weixin_message'] = msg
    job['weixin_status'] = 'pending'

def maybe_send_scheduled_weixin(job):
    if not job.get('scheduled') or job.get('weixin_status'):
        return
    try:
        save_schedule_notification(job)
    except Exception as e:
        job['weixin_status'] = 'failed'
        job['weixin_error'] = str(e)

def schedule_notify_payload():
    items = []
    seen_sent = False
    for nf in sorted(SCHEDULE_NOTIFY_DIR.glob('notify_*.json'), reverse=True):
        try:
            data = json.loads(nf.read_text(encoding='utf-8'))
            data['file'] = nf.name
            items.append(data)
            if data.get('notified'):
                seen_sent = True
        except Exception:
            pass
    return {'count': len(items), 'items': items, 'dir': str(SCHEDULE_NOTIFY_DIR)}

def schedule_notify_done_payload(body):
    filename = body.get('file', '')
    fpath = SCHEDULE_NOTIFY_DIR / filename
    # 安全检查：确保路径在 SCHEDULE_NOTIFY_DIR 内
    try:
        fpath = fpath.resolve()
        notify_dir = SCHEDULE_NOTIFY_DIR.resolve()
        if notify_dir not in fpath.parents and fpath != notify_dir:
            return {'ok': False, 'error': '路径非法'}
    except Exception:
        return {'ok': False, 'error': '路径解析失败'}
    if not fpath.exists():
        return {'ok': False, 'error': '文件不存在'}
    try:
        data = json.loads(fpath.read_text(encoding='utf-8'))
        data['notified'] = True
        data['notified_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fpath.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        job = AI_JOBS.get(data.get('job_id', ''))
        if job:
            job['weixin_status'] = 'sent'
            job['weixin_sent_at'] = data['notified_at']
        return {'ok': True}
    except Exception as e:
        return {'ok': False, 'error': str(e)}

def models_payload(): return {'items':[{'name':n,'status':s,'description':d} for n,s,d in [('SARIMA','可选','趋势时间序列'),('GARCH','可选','波动率建模'),('XGBoost','可选','非线性树模型'),('Multi Factor','可选','多因子模型'),('ARIMA','可选','传统时间序列'),('LSTM','重型','深度序列模型'),('LightGBM','可选','轻量梯度提升'),('CatBoost','可选','梯度提升模型'),('Ensemble','推荐','多模型集成')]]}

def features_payload(code):
    item,rows=find_item(code)
    return {'error':'未找到股票'} if not item else {'code':item['code'],'name':item['name'],'date':item['date'],'technical':{k:item.get(k) for k in ['ma5','ma10','ma20','ma60','rsi14','volume_ratio','momentum20','drawdown','volatility']},'market':{'price':item['price'],'change_pct':item['change_pct'],'data_source':item['data_source']},'kline_simulation':{k:item.get(k) for k in ['kline_score','kline_signal','sim_return']}}

def analyze_payload(b):
    item,rows=find_item(normalize_code(b.get('code',''))); days=int(b.get('days') or 20)
    if not item: return {'error':'未找到股票'}
    sim=kline_simulation(rows,days); return {'module':'main.py 单股智能分析-轻量版','code':item['code'],'name':item['name'],'date':item['date'],'score':item['score'],'signal':item['signal'],'price':item['price'],'change_pct':item['change_pct'],'technical_summary':item,'kline_simulation':sim,'data_source':item['data_source']}


def update_history_for_file(path, code, name):
    import baostock as bs
    start_date = '2010-01-01'
    try:
        rows = read_stock(path)
        if rows:
            start_date = rows[-1]['date']
    except Exception:
        pass
    end_date = date.today().isoformat()
    bs_code = ('sh.' if code.startswith(('6','9')) else 'sz.') + code
    rs = bs.query_history_k_data_plus(
        bs_code,
        'date,open,high,low,close,volume,amount,turn,tradestatus,pctChg',
        start_date=start_date,
        end_date=end_date,
        frequency='d',
        adjustflag='3'
    )
    if rs.error_code != '0':
        raise Exception(rs.error_msg)
    data=[]
    while (rs.error_code == '0') and rs.next():
        data.append(rs.get_row_data())
    if not data:
        return {'code':code,'name':name,'updated_rows':0,'latest_date':start_date,'path':str(path)}
    existing=[]
    if path.exists():
        with path.open('r',encoding='utf-8-sig',newline='') as f:
            existing=list(csv.DictReader(f))
    by_date={r.get('date',''):r for r in existing if r.get('date')}
    for r in data:
        by_date[r[0]]=dict(zip(rs.fields,r))
    rows=[by_date[k] for k in sorted(by_date)]
    fields=['date','open','high','low','close','volume','amount','turn','tradestatus','pctChg']
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    return {'code':code,'name':name,'updated_rows':len(data),'latest_date':rows[-1].get('date') if rows else None,'path':str(path)}

def history_update_payload(body):
    kw=(body.get('keyword','') or '').strip().lower()
    files=[]
    for path in DATA_DIR.glob('*_baostock.csv'):
        code,name=parse_name(path)
        if kw and kw not in code.lower() and kw not in name.lower():
            continue
        files.append((path,code,name))
    if not files:
        return {'ok':False,'updated':0,'failed':0,'items':[],'errors':['未找到匹配CSV']}
    if not _HISTORY_UPDATE_SEM.acquire(timeout=30):
        return {'ok':False,'updated':0,'failed':len(files),'items':[],'errors':['已有历史更新任务在运行，请稍后重试']}
    try:
        import baostock as bs
        lg=bs.login()
        if lg.error_code!='0':
            return {'ok':False,'updated':0,'failed':len(files),'items':[],'errors':['baostock登录失败: '+lg.error_msg]}
        items=[]; errors=[]
        try:
            for path,code,name in files:
                try:
                    items.append(update_history_for_file(path,code,name))
                except Exception as e:
                    errors.append({'code':code,'name':name,'error':str(e)})
        finally:
            try: bs.logout()
            except Exception: pass
        return {'ok':not errors,'updated':len(items),'failed':len(errors),'items':items,'errors':errors,'data_dir':str(DATA_DIR),'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    finally:
        _HISTORY_UPDATE_SEM.release()

def backtest_payload(b):
    code = normalize_code(b.get('code',''))
    days = int(b.get('days') or 30)
    if days < 1:
        days = 30
    if days > 252:
        days = 252

    stock_files = []
    if code:
        stock_files = sorted(DATA_DIR.glob(f'{code}_*_baostock.csv'))
        if not stock_files:
            return {'ok': False, 'module': 'app/backtest.py', 'error': '未找到股票CSV数据', 'code': code}
        stock_files = [stock_files[0].name]
    else:
        ranked = apply_strategy(load_items(), 'balanced')[:10]
        for item in ranked:
            matches = sorted(DATA_DIR.glob(f"{item['code']}_*_baostock.csv"))
            if matches:
                stock_files.append(matches[0].name)
        if not stock_files:
            return {'ok': False, 'module': 'app/backtest.py', 'error': '未找到可回测CSV数据'}

    script = f'''
import os, sys, json, contextlib, io, traceback
os.chdir(r"/work/app")
sys.path.insert(0, r"/work/app")
import backtest
out = []
errors = []
for stock_file in {stock_files!r}:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            bt = backtest.BacktestSimulator(data_dir="stock_data")
            bt.config.parameter_optimization.enabled = False
            result = bt.run_backtest(stock_file, {days})
            backtest.cleanup_resources()
        if result:
            out.append({{
                "stock_file": stock_file,
                "report_path": result.get("report_path"),
                "parameter_report_path": result.get("parameter_report_path"),
                "models": [m.get("模型名称") or m.get("model") for m in result.get("metrics", [])],
                "metrics_count": len(result.get("metrics", [])),
                "log_tail": buf.getvalue()[-5000:],
            }})
        else:
            errors.append({{"stock_file": stock_file, "error": "backtest.py 未返回结果", "log_tail": buf.getvalue()[-5000:]}})
    except Exception as e:
        errors.append({{"stock_file": stock_file, "error": str(e), "traceback": traceback.format_exc()[-3000:], "log_tail": buf.getvalue()[-5000:]}})
print(json.dumps({{"ok": len(errors)==0, "module": "app/backtest.py", "days": {days}, "count": len(out), "items": out, "errors": errors}}, ensure_ascii=False))
'''
    try:
        # 限制并发回测数，获取不到许可 60s 后放弃（避免线程堆积）
        if not _BACKTEST_SEM.acquire(timeout=60):
            return {'ok': False, 'module': 'app/backtest.py', 'error': '回测并发已满，请稍后重试'}
        try:
            cp = subprocess.run([ai_python_cmd(), '-c', script], cwd=str(ROOT/'app'), text=True, capture_output=True, timeout=900)
        finally:
            _BACKTEST_SEM.release()
        raw = (cp.stdout or '').strip()
        err = (cp.stderr or '').strip()
        if cp.returncode != 0:
            return {'ok': False, 'module': 'app/backtest.py', 'error': 'backtest.py 执行失败', 'returncode': cp.returncode, 'stdout_tail': raw[-5000:], 'stderr_tail': err[-5000:]}
        try:
            return json.loads(raw.splitlines()[-1])
        except Exception:
            return {'ok': False, 'module': 'app/backtest.py', 'error': 'backtest.py 输出解析失败', 'stdout_tail': raw[-5000:], 'stderr_tail': err[-5000:]}
    except subprocess.TimeoutExpired:
        return {'ok': False, 'module': 'app/backtest.py', 'error': 'backtest.py 回测超时'}
    except Exception as e:
        return {'ok': False, 'module': 'app/backtest.py', 'error': str(e)}


def cleanup_duplicate_reports_by_code():
    grouped = {}
    deleted = []
    seen_paths = set()
    for root in report_roots():
        if not root.exists():
            continue
        for p in root.rglob('*.html'):
            try:
                rp = p.resolve()
                rp_key = str(rp)
                if rp_key in seen_paths:
                    continue
                seen_paths.add(rp_key)
                code = report_stock_code(p.name)
                if not code:
                    continue
                st = p.stat()
                grouped.setdefault(code, []).append((st.st_mtime, rp_key, rp))
            except Exception:
                pass
    for code, rows in grouped.items():
        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for _, _, old_path in rows[1:]:
            try:
                old_path.unlink()
                deleted.append({'stock_code': code, 'path': str(old_path)})
            except FileNotFoundError:
                pass
            except Exception as e:
                deleted.append({'stock_code': code, 'path': str(old_path), 'error': str(e)})
    return deleted


def reports_payload():
    # 60 秒 TTL 缓存：GET 请求不再每次全盘扫描 + 执行删除副作用
    now = time.time()
    with _REPORTS_CACHE_LOCK:
        if _REPORTS_CACHE['data'] is not None and (now - _REPORTS_CACHE['ts']) < _REPORTS_CACHE_TTL:
            return _REPORTS_CACHE['data']
    latest_by_code = {}
    latest_without_code = {}
    for root in report_roots():
        if root.exists():
            for p in root.rglob('*.html'):
                try:
                    rp = p.resolve()
                    st = p.stat()
                    code = report_stock_code(p.name)
                    item = {
                        'name': p.name,
                        'display_name': report_display_name(p.name),
                        'path': str(rp),
                        'id': quote(p.name),
                        'url': '/reports/' + quote(p.name),
                        'size': st.st_size,
                        'mtime_ts': st.st_mtime,
                        'mtime': datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'format': 'html',
                        'stock_code': code,
                    }
                    item.update(report_metrics_from_name(p.name, st, rp, html_only=True))
                    item['industry'] = get_stock_industry(code, item.get('display_name', ''))
                    bucket = latest_by_code if code else latest_without_code
                    key = code or str(rp)
                    old_item = bucket.get(key)
                    if old_item is None or item['mtime_ts'] > old_item.get('mtime_ts', 0):
                        bucket[key] = item
                except Exception:
                    pass
    items = list(latest_by_code.values()) + list(latest_without_code.values())
    items = sorted(items, key=lambda x: x.get('mtime_ts', 0), reverse=True)
    for item in items:
        item.pop('mtime_ts', None)
    result = {'count': len(items), 'items': items, 'deleted_duplicates': 0, 'deleted_duplicate_items': []}
    with _REPORTS_CACHE_LOCK:
        _REPORTS_CACHE['ts'] = time.time()
        _REPORTS_CACHE['data'] = result
    return result

def delete_report_payload(body):
    p=safe_report_path(body.get('id',''))
    if not p or p.suffix.lower()!='.html':
        return {'ok':False,'error':'报告不存在'}
    try:
        p.unlink()
        return {'ok':True}
    except Exception as e:
        return {'ok':False,'error':str(e)}

def config_payload():
    p=ROOT/'app'/'config.yaml'; sections=[]
    if p.exists():
        for line in p.read_text(encoding='utf-8',errors='ignore').splitlines():
            if line and not line.startswith(' ') and line.endswith(':'): sections.append(line.strip(':'))
    return {'path':str(p),'exists':p.exists(),'sections':sections,'enabled_options':['选股评分','单股分析','回测','AI模型','特征工程','报告结果','系统配置']}

def resp(start,status,body,ctype):
    data=(body if isinstance(body,bytes) else body.encode('utf-8')); start(status,[('Content-Type',ctype+'; charset=utf-8'),('Content-Length',str(len(data)))]); return [data]

def read_body(env):
    try: return json.loads(env['wsgi.input'].read(int(env.get('CONTENT_LENGTH') or 0)).decode('utf-8') or '{}')
    except Exception: return {}

# ---- Basic Auth 辅助 ----
import base64 as _b64
import hmac as _hmac
from auth import AuthMiddleware
SECRET_KEY = os.environ.get('SECRET_KEY', '')
if not SECRET_KEY:
    raise SystemExit('SECRET_KEY 未设置')


def _check_basic_auth(env):
    """校验 Authorization: Basic <user:pass>，恒定时间比较。"""
    header = env.get('HTTP_AUTHORIZATION', '')
    if not header.startswith('Basic '):
        return False
    try:
        decoded = _b64.b64decode(header[6:]).decode('utf-8')
    except Exception:
        return False
    if ':' not in decoded:
        return False
    user, _, pw = decoded.partition(':')
    return _hmac.compare_digest(user, AUTH_USER) and _hmac.compare_digest(pw, AUTH_PASS)


def _requires_auth(path):
    return any(path == p or path.startswith(p + '/') for p in AUTH_PREFIXES)


def _unauthorized(start):
    body = json.dumps({'error': '未认证：需要 Basic Auth'}, ensure_ascii=False).encode('utf-8')
    start('401 Unauthorized', [
        ('Content-Type', 'application/json; charset=utf-8'),
        ('Content-Length', str(len(body))),
        ('WWW-Authenticate', 'Basic realm="%s"' % AUTH_REALM),
    ])
    return [body]


def application(env,start):
    path=urlparse(env.get('PATH_INFO','/')).path; q=parse_qs(env.get('QUERY_STRING',''))
    # ---- Basic Auth 门禁：受保护路径未认证返回 401 ----
    if _requires_auth(path) and not _check_basic_auth(env):
        return _unauthorized(start)
    if path=='/': return resp(start,'200 OK',INDEX_HTML,'text/html')
    if path=='/static/app.js': return resp(start,'200 OK',APP_JS,'application/javascript')
    if path=='/static/style.css': return resp(start,'200 OK',STYLE_CSS,'text/css')
    if path=='/reports' or path.startswith('/reports/'):
        target = q.get('file',[''])[0] if path=='/reports' else path[len('/reports/'):]
        p=safe_report_path(target)
        if p:
            data = p.read_bytes()
            ctype = mimetypes.guess_type(str(p))[0] or 'application/octet-stream'
            if p.suffix.lower()=='.html':
                ctype = 'text/html; charset=utf-8'
            start('200 OK',[('Content-Type',ctype),('Content-Length',str(len(data)))])
            return [data]
        return resp(start,'404 Not Found','Not Found','text/plain')
    routes={'/health':lambda:{'status':'ok','data_dir':str(DATA_DIR), 'exists':DATA_DIR.exists()},'/api/stocks':lambda:stock_payload(q),'/api/stocks/delete':lambda:delete_stock_payload(read_body(env)),'/api/history/update':lambda:history_update_payload(read_body(env)),'/api/models':models_payload,'/api/features':lambda:features_payload(q.get('code',[''])[0]),'/api/reports':reports_payload,'/api/reports/delete':lambda:delete_report_payload(read_body(env)),'/api/config':config_payload,'/api/ai/train':lambda:ai_train_payload(read_body(env)),'/api/ai/status':lambda:ai_status_payload(q.get('job_id',[''])[0]),'/api/ai/schedule':lambda:ai_schedule_payload(read_body(env), env.get('REQUEST_METHOD','GET')),'/api/ai/schedule_notify':schedule_notify_payload,'/api/ai/schedule_notify_done':lambda:schedule_notify_done_payload(read_body(env)),'/api/analyze':lambda:analyze_payload(read_body(env)),'/api/backtest':lambda:backtest_payload(read_body(env))}
    if path in routes: return resp(start,'200 OK',json.dumps(routes[path](),ensure_ascii=False),'application/json')
    return resp(start,'404 Not Found','Not Found','text/plain')


# ---- 旧 Basic Auth 由 AuthMiddleware 接管（强制放行）----
def _requires_auth(*a, **k): return False
def _check_basic_auth(*a, **k): return True
def _unauthorized(*a, **k): return None

application = AuthMiddleware(application, SECRET_KEY)

if __name__=='__main__':
    print(f'Web Stock Picker listening on 0.0.0.0:{PORT}')
    start_ai_scheduler_once()
    start_report_cleanup_once()
    # 优先使用生产级 WSGI 服务器 waitress（线程池 + keep-alive + HTTP/1.1），
    # 未安装时回退到 wsgiref + ThreadingMixIn。
    try:
        from waitress import serve
        serve(application, host='0.0.0.0', port=PORT, threads=32, channel_timeout=120)
    except ImportError:
        from socketserver import ThreadingMixIn
        from wsgiref.simple_server import WSGIServer
        class ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
            daemon_threads = True
        make_server('0.0.0.0', PORT, application, server_class=ThreadedWSGIServer).serve_forever()
