#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import csv, json, math, os, re, urllib.request, urllib.parse, subprocess, uuid, threading, time, sys, mimetypes, io, zipfile, importlib.util, contextlib
from datetime import datetime, date, timedelta
from pathlib import Path
from statistics import pstdev
from urllib.parse import parse_qs, urlparse, quote, unquote
from wsgiref.simple_server import make_server

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'app' / 'stock_data'
PORT = int(os.environ.get('WEB_STOCK_PICKER_PORT', '9119'))

INDEX_HTML = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Web选股系统</title><link rel="stylesheet" href="/static/style.css"></head><body><div class="wrap"><header><h1>Web选股系统</h1><p>选股 / 回测 / 重型AI训练 / 报告</p></header><nav class="tabs"><button data-tab="picker" class="active">选股评分</button><button data-tab="backtest">回测</button><button data-tab="heavyai">重型AI训练</button><button data-tab="reports">报告结果</button></nav><section id="picker" class="panel active"><div class="toolbar"><select id="strategy"><option value="balanced">综合评分</option><option value="momentum">趋势动量</option><option value="volume_breakout">放量突破</option><option value="oversold">超跌修复</option></select><input id="keyword" placeholder="股票代码/名称"><input id="limit" type="number" value="50"><button id="refresh">刷新</button><button id="downloadHistory">更新历史数据</button><span id="status"></span></div><div class="tablebox"><table><thead><tr><th data-sort="rank">排名 <span class="sort-arrows"><button onclick="sortStocks('rank','asc')">&#9650;</button><button onclick="sortStocks('rank','desc')">&#9660;</button></span></th><th data-sort="code">代码 <span class="sort-arrows"><button onclick="sortStocks('code','asc')">&#9650;</button><button onclick="sortStocks('code','desc')">&#9660;</button></span></th><th data-sort="name">名称 <span class="sort-arrows"><button onclick="sortStocks('name','asc')">&#9650;</button><button onclick="sortStocks('name','desc')">&#9660;</button></span></th><th data-sort="signal">信号 <span class="sort-arrows"><button onclick="sortStocks('signal','asc')">&#9650;</button><button onclick="sortStocks('signal','desc')">&#9660;</button></span></th><th data-sort="score">评分 <span class="sort-arrows"><button onclick="sortStocks('score','asc')">&#9650;</button><button onclick="sortStocks('score','desc')">&#9660;</button></span></th><th data-sort="price">最新价 <span class="sort-arrows"><button onclick="sortStocks('price','asc')">&#9650;</button><button onclick="sortStocks('price','desc')">&#9660;</button></span></th><th data-sort="change_pct">涨跌幅 <span class="sort-arrows"><button onclick="sortStocks('change_pct','asc')">&#9650;</button><button onclick="sortStocks('change_pct','desc')">&#9660;</button></span></th><th data-sort="ma">MA5/20/60 <span class="sort-arrows"><button onclick="sortStocks('ma','asc')">&#9650;</button><button onclick="sortStocks('ma','desc')">&#9660;</button></span></th><th data-sort="rsi14">RSI <span class="sort-arrows"><button onclick="sortStocks('rsi14','asc')">&#9650;</button><button onclick="sortStocks('rsi14','desc')">&#9660;</button></span></th><th data-sort="volume_ratio">量比 <span class="sort-arrows"><button onclick="sortStocks('volume_ratio','asc')">&#9650;</button><button onclick="sortStocks('volume_ratio','desc')">&#9660;</button></span></th><th data-sort="momentum20">20日动量 <span class="sort-arrows"><button onclick="sortStocks('momentum20','asc')">&#9650;</button><button onclick="sortStocks('momentum20','desc')">&#9660;</button></span></th><th data-sort="drawdown">最大回撤 <span class="sort-arrows"><button onclick="sortStocks('drawdown','asc')">&#9650;</button><button onclick="sortStocks('drawdown','desc')">&#9660;</button></span></th><th data-sort="kline_score">K线模拟 <span class="sort-arrows"><button onclick="sortStocks('kline_score','asc')">&#9650;</button><button onclick="sortStocks('kline_score','desc')">&#9660;</button></span></th><th>操作</th><th data-sort="data_source">数据源 <span class="sort-arrows"><button onclick="sortStocks('data_source','asc')">&#9650;</button><button onclick="sortStocks('data_source','desc')">&#9660;</button></span></th><th data-sort="tags">标签 <span class="sort-arrows"><button onclick="sortStocks('tags','asc')">&#9650;</button><button onclick="sortStocks('tags','desc')">&#9660;</button></span></th></tr></thead><tbody id="tbody"></tbody></table></div></section><section id="backtest" class="panel"><h2>回测模块</h2><div class="toolbar"><input id="backtestCode" placeholder="股票代码，可留空"><input id="backtestDays" type="number" value="20"><button id="backtestRun">运行 backtest.py 回测</button></div><pre id="backtestResult" class="result"></pre></section><section id="heavyai" class="panel"><h2>重型AI训练</h2><div class="toolbar"><input id="aiCode" placeholder="股票代码"><input id="aiName" placeholder="股票名称，可留空"><input id="aiDays" type="number" value="20"><select id="aiMode"><option value="single">单股训练</option><option value="industry_5d">行业资金5日流入前三</option><option value="industry_10d">行业资金10日流入前三</option></select><input id="aiLimit" type="number" value="3" title="每个行业训练股票数"><button id="aiRun">启动完整多模型训练</button><button id="aiStatusBtn">刷新状态</button></div><pre id="aiResult" class="result"></pre></section><section id="reports" class="panel"><h2>报告结果</h2><div class="toolbar"><button id="reportsLoad">刷新报告列表</button><span id="reportsStatus"></span></div><div class="tablebox report-tablebox"><table class="report-table"><thead><tr><th data-sort="idx">序号 <span class="sort-arrows"><button onclick="sortReports('idx','asc')">&#9650;</button><button onclick="sortReports('idx','desc')">&#9660;</button></span></th><th data-sort="name">报告名称 <span class="sort-arrows"><button onclick="sortReports('name','asc')">&#9650;</button><button onclick="sortReports('name','desc')">&#9660;</button></span></th><th data-sort="risk_level">风险等级 <span class="sort-arrows"><button onclick="sortReports('risk_level','asc')">&#9650;</button><button onclick="sortReports('risk_level','desc')">&#9660;</button></span></th><th data-sort="risk_score">风险评分 <span class="sort-arrows"><button onclick="sortReports('risk_score','asc')">&#9650;</button><button onclick="sortReports('risk_score','desc')">&#9660;</button></span></th><th data-sort="volatility">波动率 <span class="sort-arrows"><button onclick="sortReports('volatility','asc')">&#9650;</button><button onclick="sortReports('volatility','desc')">&#9660;</button></span></th><th data-sort="max_drawdown">Total Return <span class="sort-arrows"><button onclick="sortReports('max_drawdown','asc')">&#9650;</button><button onclick="sortReports('max_drawdown','desc')">&#9660;</button></span></th><th data-sort="sharpe">夏普比率 <span class="sort-arrows"><button onclick="sortReports('sharpe','asc')">&#9650;</button><button onclick="sortReports('sharpe','desc')">&#9660;</button></span></th><th data-sort="win_rate">胜率 <span class="sort-arrows"><button onclick="sortReports('win_rate','asc')">&#9650;</button><button onclick="sortReports('win_rate','desc')">&#9660;</button></span></th><th>操作</th><th data-sort="mtime">修改时间 <span class="sort-arrows"><button onclick="sortReports('mtime','asc')">&#9650;</button><button onclick="sortReports('mtime','desc')">&#9660;</button></span></th><th data-sort="advice">分析建议 <span class="sort-arrows"><button onclick="sortReports('advice','asc')">&#9650;</button><button onclick="sortReports('advice','desc')">&#9660;</button></span></th></tr></thead><tbody id="reportsBox"></tbody></table></div></section></div><script src="/static/app.js"></script></body></html>'''
APP_JS = '''function q(id){return document.getElementById(id)}document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button,.panel').forEach(x=>x.classList.remove('active'));b.classList.add('active');q(b.dataset.tab).classList.add('active')});async function gj(u){return await (await fetch(u)).json()}async function pj(u,o){return await (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})).json()}
let stockItems=[];let stockSort={key:'rank',dir:'asc'};function stockSortValue(x,k){if(k==='ma')return Number(x.ma20??-999999);if(k==='tags')return (x.tags||[]).join(',');if(['rank','score','price','change_pct','rsi14','volume_ratio','momentum20','drawdown','kline_score'].includes(k))return Number(x[k]??-999999);if(k==='data_source')return x.data_fresh?'实时':'CSV';return String(x[k]??'')}function sortStocks(k,d){stockSort={key:k,dir:d};renderStocks()}function renderStocks(){let arr=[...stockItems];arr.sort((a,b)=>{let av=stockSortValue(a,stockSort.key),bv=stockSortValue(b,stockSort.key);let r=(typeof av==='number'&&typeof bv==='number')?av-bv:String(av).localeCompare(String(bv),'zh-CN');return stockSort.dir==='asc'?r:-r});let tb=q('tbody');tb.innerHTML='';for(let i=0;i<arr.length;i++){let x=arr[i],tr=document.createElement('tr');tr.innerHTML=`<td>${i+1}</td><td class="code">${x.code}</td><td>${x.name}</td><td><span class="sig">${x.signal}</span></td><td class="score">${x.score}</td><td>${x.price}</td><td class="${x.change_pct>=0?'red':'green'}">${x.change_pct}%</td><td>${x.ma5??'-'} / ${x.ma20??'-'} / ${x.ma60??'-'}</td><td>${x.rsi14??'-'}</td><td>${x.volume_ratio??'-'}</td><td>${x.momentum20??'-'}%</td><td>${x.drawdown??'-'}%</td><td>${x.kline_score??'-'} / ${x.sim_return??'-'}%</td><td><button class="danger" onclick="deleteStock('${x.code}','${(x.name||'').replace(/'/g,\"\\'\")}')">删除</button></td><td>${x.data_fresh?'实时':'CSV'}</td><td>${(x.tags||[]).map(t=>`<span class="tag">${t}</span>`).join('')}</td>`;tb.appendChild(tr)}}async function deleteStock(code,name){if(!confirm('确认删除个股：'+code+' '+name+' ?'))return;let r=await fetch('/api/stocks/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});let d=await r.json();if(!d.ok){alert(d.error||'删除失败');return}await loadStocks()}async function loadStocks(){let s=q('strategy').value,k=q('keyword').value,l=q('limit').value||50;q('status').textContent='加载中...';let d=await gj(`/api/stocks?strategy=${s}&limit=${l}&keyword=${encodeURIComponent(k)}`);stockItems=d.items||[];renderStocks();q('status').textContent=`${d.updated_at} · ${d.count}只 · 排序 ${stockSort.key} ${stockSort.dir==='asc'?'↑':'↓'}`}q('refresh').onclick=loadStocks;q('downloadHistory').onclick=async()=>{let k=q('keyword').value.trim();if(!k&&!confirm('未输入股票代码/名称，将更新全部CSV，可能耗时较长，继续？'))return;q('status').textContent='历史数据更新中...';try{let d=await pj('/api/history/update',{keyword:k});q('status').textContent=`更新完成：${d.updated} 个，失败 ${d.failed}`;await loadStocks()}catch(e){q('status').textContent='更新失败：'+e}};q('strategy').onchange=loadStocks;q('keyword').onkeydown=e=>{if(e.key==='Enter')loadStocks()};
q('backtestRun').onclick=async()=>q('backtestResult').textContent=JSON.stringify(await pj('/api/backtest',{code:q('backtestCode').value.trim(),days:q('backtestDays').value}),null,2);let lastJob='';q('aiRun').onclick=async()=>{q('aiResult').textContent='启动中...';let d=await pj('/api/ai/train',{code:q('aiCode').value.trim(),name:q('aiName').value,days:q('aiDays').value,mode:q('aiMode').value,limit:q('aiLimit').value});lastJob=d.job_id||'';q('aiResult').textContent=JSON.stringify(d,null,2)};q('aiStatusBtn').onclick=async()=>{q('aiResult').textContent=JSON.stringify(await gj('/api/ai/status?job_id='+encodeURIComponent(lastJob)),null,2)};
async function deleteReport(id,name){if(!confirm('确认删除报告：'+name+' ?'))return;let r=await fetch('/api/reports/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});let d=await r.json();if(!d.ok)alert(d.error||'删除失败');q('reportsLoad').click()}let reportItems=[];let reportSort={key:'__default',dir:'default'};function riskRank(v){return {'低':1,'中':2,'高':3}[v]||0}function reportSortValue(x,k){if(k==='idx')return x._idx||0;if(k==='risk_level')return riskRank(x.risk_level);if(['risk_score','volatility','max_drawdown','sharpe','win_rate'].includes(k))return Number(x[k]??-999999);return String(x[k]||'')}function compareReport(a,b,k,d){let av=reportSortValue(a,k),bv=reportSortValue(b,k);let r=(typeof av==='number'&&typeof bv==='number')?av-bv:String(av).localeCompare(String(bv),'zh-CN');return d==='asc'?r:-r}function defaultReportCompare(a,b){return compareReport(a,b,'risk_level','asc')||compareReport(a,b,'sharpe','desc')||compareReport(a,b,'win_rate','desc')||compareReport(a,b,'max_drawdown','desc')||compareReport(a,b,'volatility','asc')||compareReport(a,b,'risk_score','asc')||compareReport(a,b,'mtime','desc')}function sortReports(k,d){reportSort={key:k,dir:d};renderReports()}function renderReports(){let arr=[...reportItems];arr.sort((a,b)=>reportSort.key==='__default'?defaultReportCompare(a,b):compareReport(a,b,reportSort.key,reportSort.dir));let tb=q('reportsBox');tb.innerHTML=arr.map((x,i)=>{let safeName=(x.name||'').replace(/'/g,"\'");let pct=v=>v==null?'-':(Number(v).toFixed(2)+'%');let num=v=>v==null?'-':Number(v).toFixed(2);return `<tr><td>${i+1}</td><td><a class="report-name" href="${x.url}" target="_blank">${x.display_name||x.name}</a></td><td><span class="risk risk-${x.risk_level}">${x.risk_level||'-'}</span></td><td>${num(x.risk_score)}</td><td>${pct(x.volatility)}</td><td>${pct(x.max_drawdown)}</td><td>${num(x.sharpe)}</td><td>${pct(x.win_rate)}</td><td><div class="report-actions"><a class="open-report" href="${x.url}" target="_blank">打开</a><button class="danger" onclick="deleteReport('${x.id}','${safeName}')">删除</button></div></td><td>${x.mtime||''}</td><td class="advice-cell" title="${x.advice||''}">${x.advice||'-'}</td></tr>`}).join('');q('reportsStatus').textContent=reportSort.key==='__default'?`共 ${arr.length} 个报告 · 默认排序：风险等级↑ 夏普比率↓ 胜率↓ Total Return↓ 波动率↑ 风险评分↑`:`共 ${arr.length} 个报告 · 排序 ${reportSort.key} ${reportSort.dir==='asc'?'↑':'↓'}`}q('reportsLoad').onclick=async()=>{let d=await gj('/api/reports');reportItems=(d.items||[]).map((x,i)=>({...x,_idx:i+1}));renderReports()};loadStocks();'''
STYLE_CSS = '''body{margin:0;background:#0f172a;color:#e5e7eb;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,"Microsoft YaHei",sans-serif}.wrap{max-width:1480px;margin:0 auto;padding:24px}h1{margin:0 0 8px}.tabs,.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0}.tabs button,select,input,button{background:#111827;color:#e5e7eb;border:1px solid #334155;border-radius:8px;padding:9px 12px}button,.tabs button.active{cursor:pointer;background:#2563eb;border-color:#2563eb}.panel{display:none}.panel.active{display:block}.tablebox{overflow:auto;border:1px solid #1e293b;border-radius:12px}table{width:100%;border-collapse:collapse;background:#111827}th,td{padding:10px 12px;border-bottom:1px solid #1e293b;white-space:nowrap;text-align:left}th{background:#1e293b}.score{font-weight:700;color:#fbbf24}.sig,.tag{display:inline-block;background:#334155;border-radius:999px;padding:3px 8px}.tag{background:#1d4ed8;margin:2px}.red{color:#f87171}.green{color:#4ade80}.result{background:#020617;border:1px solid #1e293b;border-radius:12px;padding:14px;min-height:180px;white-space:pre-wrap;overflow:auto}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}.card{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:14px}.card p{color:#94a3b8}.report-tablebox{max-height:70vh}.report-table th{position:sticky;top:0;z-index:1;vertical-align:middle}.report-name,.report-card a{color:#93c5fd;text-decoration:none}.report-name:hover,.report-card a:hover{text-decoration:underline}.advice-cell{max-width:360px;overflow:hidden;text-overflow:ellipsis}.report-actions{display:flex;gap:8px;align-items:center;margin-top:0}.open-report{background:#2563eb;color:#fff;border-radius:8px;padding:7px 10px;text-decoration:none}.danger{background:#dc2626;border-color:#dc2626;padding:7px 10px}.sort-arrows{display:inline-flex;flex-direction:column;gap:1px;margin-left:4px;vertical-align:middle}.sort-arrows button{font-size:9px;line-height:9px;padding:0 3px;border-radius:2px;background:#334155;border-color:#475569;color:#cbd5e1}.risk{display:inline-block;border-radius:999px;padding:2px 8px}.risk-低{background:#14532d;color:#86efac}.risk-中{background:#713f12;color:#fde68a}.risk-高{background:#7f1d1d;color:#fecaca}'''

def to_float(v, default=None):
    try:
        if v is None or str(v).strip() == '': return default
        x=float(v); return default if math.isnan(x) or math.isinf(x) else x
    except Exception: return default

def avg(v):
    v=[x for x in v if x is not None]; return sum(v)/len(v) if v else None

def ma(v,n): return avg(v[-n:]) if len(v)>=n else None

def rsi(c,n=14):
    if len(c)<=n: return None
    gains=[]; losses=[]
    for i in range(-n,0):
        d=c[i]-c[i-1]; gains.append(max(d,0)); losses.append(abs(min(d,0)))
    ag,al=avg(gains),avg(losses); return 100.0 if al==0 else 100-(100/(1+ag/al))

def max_drawdown(c,n=60):
    vals=c[-n:] if len(c)>=n else c
    if not vals: return None
    peak=vals[0]; worst=0
    for p in vals:
        peak=max(peak,p); worst=min(worst,(p-peak)/peak*100 if peak else 0)
    return worst

def volatility(p,n=20):
    vals=[x for x in p[-n:] if x is not None]; return pstdev(vals) if len(vals)>=2 else None


def normalize_code(value):
    text = str(value or '').strip()
    m = re.search(r'(?:sh|sz)?(\d{6})', text, re.I)
    return m.group(1) if m else text

def parse_name(path):
    m=re.match(r'^(\d{6})_(.+?)_baostock$', path.stem)
    return (m.group(1),m.group(2)) if m else (path.stem[:6],path.stem)

def read_stock(path):
    rows=[]
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            close=to_float(r.get('close'))
            if close and close>0: rows.append({'date':r.get('date',''),'open':to_float(r.get('open'),close),'high':to_float(r.get('high'),close),'low':to_float(r.get('low'),close),'close':close,'volume':to_float(r.get('volume'),0) or 0,'pctChg':to_float(r.get('pctChg'),0) or 0})
    rows.sort(key=lambda x:x['date']); return rows

def fetch_sina_quote(code):
    try:
        sec=('sh' if code.startswith(('6','9')) else 'sz')+code
        url='https://hq.sinajs.cn/list='+sec
        headers={'User-Agent':'Mozilla/5.0','Referer':'https://finance.sina.com.cn/'}
        raw=urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=5).read(2048).decode('gb18030','ignore')
        body=raw.split('="',1)[1].split('"',1)[0]
        parts=body.split(',')
        if len(parts)<32 or not parts[0]: return None
        prev=to_float(parts[2]); price=to_float(parts[3]) or prev
        if not price: return None
        return {'date':parts[30] or date.today().isoformat(),'open':to_float(parts[1],price),'high':to_float(parts[4],price),'low':to_float(parts[5],price),'close':price,'volume':to_float(parts[8],0) or 0,'pctChg':((price/prev-1)*100 if prev else 0),'source':'sina_realtime'}
    except Exception: return None

def fetch_tencent_quote(code):
    try:
        sec=('sh' if code.startswith(('6','9')) else 'sz')+code
        raw=urllib.request.urlopen(urllib.request.Request('http://qt.gtimg.cn/q='+sec,headers={'User-Agent':'Mozilla/5.0'}),timeout=5).read(1024).decode('gbk','ignore')
        parts=raw.split('="',1)[1].split('"',1)[0].split('~')
        price=to_float(parts[3]); prev=to_float(parts[4]); ts=parts[30]
        if not price: return None
        return {'date':f'{ts[:4]}-{ts[4:6]}-{ts[6:8]}' if len(ts)>=8 else date.today().isoformat(),'open':to_float(parts[5],price),'high':to_float(parts[33],price) if len(parts)>33 else price,'low':to_float(parts[34],price) if len(parts)>34 else price,'close':price,'volume':to_float(parts[6],0) or 0,'pctChg':((price/prev-1)*100 if prev else 0),'source':'tencent_realtime'}
    except Exception: return None

def fetch_realtime_quote(code):
    # push2 当前在该环境会被远端断开；改用可用的新浪实时接口，腾讯作为备用。
    return fetch_sina_quote(code) or fetch_tencent_quote(code)

def merge_realtime_row(code, rows):
    q=fetch_realtime_quote(code)
    if not q: return rows,'csv'
    rows=list(rows)
    if rows and rows[-1]['date']==q['date']: rows[-1].update({k:q[k] for k in ('open','high','low','close','volume','pctChg')})
    else: rows.append({k:q[k] for k in ('date','open','high','low','close','volume','pctChg')})
    rows.sort(key=lambda x:x['date']); return rows,q['source']

def kline_simulation(rows,days=10):
    c=[r['close'] for r in rows if r.get('close')]
    if len(c)<25: return {'kline_score':0,'kline_signal':'样本不足','sim_return':0}
    rets=[c[i]/c[i-1]-1 for i in range(1,len(c)) if c[i-1]]; recent=rets[-20:]
    drift=avg(recent[-10:]) or 0; vol=pstdev(recent) if len(recent)>=2 else 0; trend=((ma(c,5)/ma(c,20)-1) if ma(c,5) and ma(c,20) else 0); mom=((c[-1]/c[-11]-1) if len(c)>=11 and c[-11] else 0)
    sim=(drift*.45+trend*.08+mom*.05)*days*100-min(vol*4,.08)*25; score=max(0,min(100,50+sim*4-vol*120)); sig='模拟偏强' if score>=70 else '模拟向上' if score>=55 else '模拟震荡' if score>=40 else '模拟偏弱'
    return {'kline_score':round(score,2),'kline_signal':sig,'sim_return':round(sim,2)}

def score_stock(code,name,rows):
    if len(rows)<30: return None
    c=[r['close'] for r in rows]; vols=[r['volume'] for r in rows]; p=[r['pctChg'] for r in rows]; last=rows[-1]; price=c[-1]
    ma5,ma10,ma20,ma60=ma(c,5),ma(c,10),ma(c,20),ma(c,60); rsi14=rsi(c); vr=(ma(vols,5)/ma(vols,20)) if ma(vols,5) and ma(vols,20) else None; mom=((price/c[-21]-1)*100) if len(c)>=21 and c[-21] else None; dd=max_drawdown(c); vol=volatility(p)
    score=50; tags=[]
    if ma5 and ma20 and ma60:
        if price>ma5>ma20>ma60: score+=18; tags.append('多头排列')
        elif price>ma20>ma60: score+=10; tags.append('中期向上')
        elif price<ma20: score-=8
    if mom is not None: score+=max(min(mom,20),-20)*.8
    if vr is not None:
        if vr>=1.5: score+=8; tags.append('放量')
        elif vr<.7: score-=3
    if rsi14 is not None:
        if 45<=rsi14<=70: score+=8
        elif 25<=rsi14<35: score+=6; tags.append('超卖修复')
        elif rsi14>82: score-=10
    if dd is not None: score+=5 if dd>-10 else -8 if dd<-30 else 0
    recent=c[-20:]
    if price>=max(recent)*.98: tags.append('接近20日新高')
    if price<=min(recent)*1.05: tags.append('接近20日低位')
    score=round(max(0,min(100,score)),2); signal='强关注' if score>=78 else '关注' if score>=65 else '观察' if score>=50 else '谨慎'
    return {'code':code,'name':name,'date':last['date'],'price':round(price,3),'change_pct':round(last['pctChg'],2),'score':score,'signal':signal,'ma5':round(ma5,3) if ma5 else None,'ma10':round(ma10,3) if ma10 else None,'ma20':round(ma20,3) if ma20 else None,'ma60':round(ma60,3) if ma60 else None,'rsi14':round(rsi14,2) if rsi14 is not None else None,'volume_ratio':round(vr,2) if vr is not None else None,'momentum20':round(mom,2) if mom is not None else None,'drawdown':round(dd,2) if dd is not None else None,'volatility':round(vol,2) if vol is not None else None,'tags':tags}

def find_item(code):
    code = normalize_code(code)
    for path in DATA_DIR.glob('*_baostock.csv'):
        c,n=parse_name(path)
        if not code or c==code:
            rows,src=merge_realtime_row(c,read_stock(path)); item=score_stock(c,n,rows)
            if item:
                sim=kline_simulation(rows); item.update(sim); item['score']=round(max(0,min(100,item['score']*.82+item['kline_score']*.18)),2); item['data_source']=src; item['data_fresh']=src!='csv'; item['tags'].append(item['kline_signal']); return item,rows
    return None,[]

def load_items():
    out=[]
    for path in DATA_DIR.glob('*_baostock.csv'):
        item,_=find_item(parse_name(path)[0])
        if item: out.append(item)
    return out

def apply_strategy(items,strategy):
    if strategy=='momentum': items=[x for x in items if (x.get('momentum20') or -999)>0 and x.get('ma20') and x['price']>=x['ma20']]; return sorted(items,key=lambda x:((x.get('momentum20') or -999),x['score']),reverse=True)
    if strategy=='volume_breakout': items=[x for x in items if (x.get('volume_ratio') or 0)>=1.3 and (x.get('momentum20') or -999)>=-3]; return sorted(items,key=lambda x:((x.get('volume_ratio') or 0),x['score']),reverse=True)
    if strategy=='oversold': items=[x for x in items if (x.get('rsi14') or 999)<=45 or '接近20日低位' in x.get('tags',[])]; return sorted(items,key=lambda x:(x['score'],-(x.get('rsi14') or 100)),reverse=True)
    return sorted(items,key=lambda x:x['score'],reverse=True)

def stock_payload(q):
    items=load_items(); kw=q.get('keyword',[''])[0].lower(); limit=max(1,min(int(q.get('limit',['50'])[0] or 50),500)); strategy=q.get('strategy',['balanced'])[0]
    if kw: items=[x for x in items if kw in x['code'].lower() or kw in x['name'].lower()]
    items=apply_strategy(items,strategy)[:limit]
    for i,x in enumerate(items,1): x['rank']=i
    return {'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'strategy':strategy,'count':len(items),'items':items}


def delete_stock_payload(body):
    code = normalize_code(str(body.get('code','')))
    if not code or len(code) != 6:
        return {'ok': False, 'error': '股票代码无效'}
    deleted = []
    for path in DATA_DIR.glob(f'{code}_*_baostock.csv'):
        try:
            path.unlink()
            deleted.append(str(path))
        except Exception as e:
            return {'ok': False, 'error': str(e), 'code': code, 'deleted': deleted}
    if not deleted:
        return {'ok': False, 'error': '未找到该股票历史数据文件', 'code': code}
    return {'ok': True, 'code': code, 'deleted': deleted, 'count': len(deleted)}


AI_JOBS = {}

def ai_python_cmd():
    # 原 app/venv 损坏时，优先使用当前容器 Python；重型依赖缺失会在状态中返回错误
    venv = ROOT / 'app' / 'venv' / 'bin' / 'python'
    try:
        if venv.exists() and venv.stat().st_size > 0:
            return str(venv)
    except Exception:
        pass
    return sys.executable


def em_json(path, params):
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36',
        'Referer':'https://data.eastmoney.com/bkzj/hy.html',
        'Accept':'application/json,text/plain,*/*',
        'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection':'close',
    }
    query = urllib.parse.urlencode(params)
    urls = [
        'https://push2delay.eastmoney.com' + path + '?' + query,
        'https://push2.eastmoney.com' + path + '?' + query,
        'http://push2delay.eastmoney.com' + path + '?' + query,
        'http://push2.eastmoney.com' + path + '?' + query,
    ]
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            return json.loads(urllib.request.urlopen(req, timeout=12).read().decode('utf-8','ignore'))
        except Exception:
            continue
    return None

def datacenter_json(report_name, params=None):
    p = {'reportName': report_name, 'columns':'ALL', 'pageNumber':'1', 'pageSize':'50', 'sortTypes':'-1'}
    if params:
        p.update(params)
    url = 'https://datacenter-web.eastmoney.com/api/data/v1/get?' + urllib.parse.urlencode(p)
    req = urllib.request.Request(url, headers={
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36',
        'Referer':'https://data.eastmoney.com/',
        'Accept':'application/json,text/plain,*/*',
    })
    try:
        return json.loads(urllib.request.urlopen(req, timeout=12).read().decode('utf-8','ignore'))
    except Exception:
        return None

def local_industry_batch(period='5d', limit=3):
    items = apply_strategy(load_items(), 'balanced')[:max(9, limit*3)]
    buckets = [
        ('LOCAL_AI', 'AI算力/光模块候选', lambda x: any(k in x['name'] for k in ('新易盛','中际旭创','天孚','源杰','光迅','仕佳'))),
        ('LOCAL_SEMI', '半导体/电子候选', lambda x: any(k in x['name'] for k in ('芯','半导体','电子','微','晶','光电'))),
        ('LOCAL_MOM', '本地强势动量候选', lambda x: True),
    ]
    industries=[]; stocks=[]; used=set()
    for code,name,pred in buckets:
        group=[x for x in items if pred(x) and x['code'] not in used][:limit]
        if not group:
            continue
        industries.append({'code':code+'_'+period,'name':name,'flow':0,'period':period,'source':'local_realtime_fallback'})
        for x in group:
            used.add(x['code'])
            stocks.append({'code':x['code'],'name':x['name'],'industry':name,'industry_code':code+'_'+period,'flow':0,'source':'local_realtime_fallback'})
        if len(industries) >= 3:
            break
    if not stocks:
        industries=[{'code':'LOCAL_TOP_'+period,'name':'本地评分候选','flow':0,'period':period,'source':'local_realtime_fallback'}]
        stocks=[{'code':x['code'],'name':x['name'],'industry':'本地评分候选','industry_code':'LOCAL_TOP_'+period,'flow':0,'source':'local_realtime_fallback'} for x in items[:limit]]
    return industries, stocks

def sina_market_rows(nodes=('hs_a','cyb','sh_a','sz_a'), limit=80):
    rows=[]; seen=set()
    for node in nodes:
        try:
            url='https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?'+urllib.parse.urlencode({'page':'1','num':str(limit),'sort':'changepercent','asc':'0','node':node,'symbol':'','_s_r_a':'page'})
            req=urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0','Referer':'https://finance.sina.com.cn/'})
            txt=urllib.request.urlopen(req, timeout=8).read().decode('gb18030','ignore')
            data=json.loads(txt) if txt.strip().startswith('[') else []
            for x in data:
                code=normalize_code(x.get('code') or x.get('symbol'))
                if len(code)==6 and code[0] in '0368' and code not in seen:
                    seen.add(code); rows.append(x)
        except Exception:
            continue
    return rows

def alt_top_industries(period='5d'):
    rows=sina_market_rows(limit=100)
    groups=[
        ('SINA_AI_'+period,'新浪A股强势AI/光模块候选',lambda x:any(k in (x.get('name') or '') for k in ('新易盛','中际','天孚','源杰','光迅','光库','仕佳','剑桥','太辰'))),
        ('SINA_SEMI_'+period,'新浪A股强势半导体/电子候选',lambda x:any(k in (x.get('name') or '') for k in ('芯','半导体','电子','微','晶','光电','集成'))),
        ('SINA_MOM_'+period,'新浪A股强势动量候选',lambda x:True),
    ]
    out=[]
    for code,name,pred in groups:
        vals=[to_float(x.get('changepercent'),0) or 0 for x in rows if pred(x)]
        if vals:
            out.append({'code':code,'name':name,'flow':round(sum(vals),2),'period':period,'source':'sina_market_center'})
    return out[:3]

def alt_industry_stocks(industry_code, limit=3, period='5d'):
    rows=sina_market_rows(limit=120)
    if 'AI' in industry_code:
        pred=lambda x:any(k in (x.get('name') or '') for k in ('新易盛','中际','天孚','源杰','光迅','光库','仕佳','剑桥','太辰'))
    elif 'SEMI' in industry_code:
        pred=lambda x:any(k in (x.get('name') or '') for k in ('芯','半导体','电子','微','晶','光电','集成'))
    else:
        pred=lambda x:True
    out=[]
    for x in rows:
        if not pred(x): continue
        code=normalize_code(x.get('code') or x.get('symbol'))
        name=x.get('name') or ('股票'+code)
        pct=to_float(x.get('changepercent'),0) or 0
        if len(code)==6:
            out.append({'code':code,'name':name,'flow':pct,'source':'sina_market_center'})
    out.sort(key=lambda x:x['flow'], reverse=True)
    return out[:limit]

def eastmoney_top_industries(period='5d'):
    fid = 'f164' if period == '5d' else 'f174'
    fields = 'f12,f14,f2,f160,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f260,f261,f124,f1,f13'
    data = em_json('/api/qt/clist/get', {
        'fid': fid,
        'po': '1',
        'pz': '50',
        'pn': '1',
        'np': '1',
        'fltt': '2',
        'invt': '2',
        'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
        'fs': 'm:90+t:3',
        'fields': fields,
    })
    rows = (((data or {}).get('data') or {}).get('diff') or [])
    items=[]
    for x in rows:
        flow = to_float(x.get(fid), 0) or 0
        code = x.get('f12','')
        name = x.get('f14','')
        if code and name:
            items.append({'code':code,'name':name,'flow':flow,'period':period,'source':'eastmoney_push2delay'})
    items.sort(key=lambda x:x['flow'], reverse=True)
    return items[:3]

def eastmoney_industry_stocks(industry_code, limit=3, period='5d'):
    fid = 'f164' if period == '5d' else 'f174'
    fields = 'f12,f14,f2,f160,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f260,f261,f124,f1,f13'
    data = em_json('/api/qt/clist/get', {
        'fid': fid,
        'po': '1',
        'pz': str(max(20, limit*5)),
        'pn': '1',
        'np': '1',
        'fltt': '2',
        'invt': '2',
        'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
        'fs': 'b:' + industry_code,
        'fields': fields,
    })
    rows = (((data or {}).get('data') or {}).get('diff') or [])
    out=[]
    for x in rows:
        code = normalize_code(x.get('f12',''))
        name = x.get('f14','') or ('股票'+code)
        if code and len(code)==6:
            out.append({'code':code,'name':name,'flow':to_float(x.get(fid),0) or 0,'source':'eastmoney_push2delay'})
    out.sort(key=lambda x:x['flow'], reverse=True)
    return out[:limit]

def fallback_industry_batch(period='5d', limit=3):
    return local_industry_batch(period, limit)

def load_gainian_module():
    path = ROOT / 'app' / 'model' / 'GaiNian stock.py'
    spec = importlib.util.spec_from_file_location('gainian_stock_pool', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError('无法加载 GaiNian stock.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def resolve_industry_batch(period='5d', limit=3):
    source = 'eastmoney_push2delay'
    industries = eastmoney_top_industries(period)
    if not industries:
        raise RuntimeError('东方财富板块资金流未返回行业数据，已禁用本地算法候选池')

    out_industries = []
    stocks = []
    for ind0 in industries[:3]:
        ind = {
            'code': str(ind0.get('code', '')),
            'name': str(ind0.get('name', '')),
            'flow': ind0.get('flow', 0),
            'period': period,
            'source': source,
        }
        if not ind['code'] or not ind['name']:
            continue
        out_industries.append(ind)
        for st0 in eastmoney_industry_stocks(ind['code'], limit, period):
            code = normalize_code(st0.get('code', ''))
            if len(code) != 6:
                continue
            stocks.append({
                'code': code,
                'name': str(st0.get('name') or ('股票' + code)),
                'industry': ind['name'],
                'industry_code': ind['code'],
                'flow': st0.get('flow', 0),
                'source': source,
            })

    seen = set()
    uniq = []
    for st in stocks:
        if st['code'] not in seen:
            seen.add(st['code'])
            uniq.append(st)

    if not out_industries or not uniq:
        raise RuntimeError('东方财富板块资金流未返回有效成分股，已禁用本地算法候选池')
    return out_industries, uniq

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

def run_ai_job(job_id, code, name, days):
    job = AI_JOBS[job_id]
    job['status'] = 'running'
    job['started_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    safe_code = normalize_code(code)
    safe_name = name or (find_item(safe_code)[0] or {}).get('name') or ('股票' + safe_code)
    script = ROOT / 'web_stock_picker' / ('ai_runner_' + job_id + '.py')
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
    log_path = ROOT / 'web_stock_picker' / ('ai_job_' + job_id + '.log')
    job['log_path'] = str(log_path)
    try:
        with open(log_path, 'w', encoding='utf-8') as log:
            p = subprocess.Popen([ai_python_cmd(), str(script)], cwd=str(ROOT / 'app'), stdout=log, stderr=subprocess.STDOUT, text=True)
            job['pid'] = p.pid
            rc = p.wait()
        job['returncode'] = rc
        job['status'] = 'completed' if rc == 0 else 'failed'
        job['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        job['finished_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def ai_train_payload(body):
    days = int(body.get('days') or 20)
    mode = body.get('mode','single')
    job_id = uuid.uuid4().hex[:12]
    if mode in ('industry_5d','industry_10d'):
        period = '5d' if mode == 'industry_5d' else '10d'
        limit = max(1, min(int(body.get('limit') or 3), 10))
        industries, stocks = resolve_industry_batch(period, limit)
        if not stocks:
            return {'error':'替代行情接口当前不可达，且本地候选为空'}
        AI_JOBS[job_id] = {'job_id':job_id,'days':days,'status':'queued','created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mode':'东方财富板块资金流%s批量训练'%('5日' if period=='5d' else '10日'),'source':'eastmoney_push2delay','industries':industries,'stocks':stocks,'total':len(stocks),'completed':0}
        t = threading.Thread(target=run_ai_batch_job, args=(job_id, stocks, days), daemon=True)
        t.start()
        return AI_JOBS[job_id]
    code = normalize_code(body.get('code',''))
    if not code:
        return {'error':'请输入股票代码'}
    AI_JOBS[job_id] = {'job_id':job_id,'code':code,'name':body.get('name',''),'days':days,'status':'queued','created_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'mode':'main.py完整多模型训练'}
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
        cp = subprocess.run([ai_python_cmd(), '-c', script], cwd=str(ROOT/'app'), text=True, capture_output=True, timeout=900)
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

def report_roots():
    return [ROOT/'app'/'results'/'reports', ROOT/'app'/'results', ROOT/'app'/'backtest_results']

def safe_report_path(rel):
    raw = unquote(rel or '').lstrip('/')
    candidates = [raw]
    try:
        fixed = raw.encode('latin-1').decode('utf-8')
        if fixed not in candidates:
            candidates.append(fixed)
    except Exception:
        pass
    for root in report_roots():
        if not root.exists():
            continue
        try:
            rr = root.resolve()
            for name in candidates:
                p = (root / name).resolve()
                if p.is_file() and (p == rr or rr in p.parents):
                    return p
            for x in root.rglob('*'):
                if not x.is_file():
                    continue
                xp = x.resolve()
                if not (xp == rr or rr in xp.parents):
                    continue
                if x.name in candidates or str(xp) in candidates:
                    return xp
        except Exception:
            pass
    return None


def extract_html_total_return(path):
    try:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'<div class="stat-label">\s*Total Return\s*</div>\s*<div class="stat-value"[^>]*>\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*%', text, re.I)
        if m:
            return round(float(m.group(1)), 2)
    except Exception:
        pass
    return None

def report_display_name(filename):
    m = re.match(r'^(\d{6})_([^_]+)', filename or '')
    return f"{m.group(1)}_{m.group(2)}" if m else (filename or '')

def report_metrics_from_name(name, stat, path=None):
    code = normalize_code(name)
    item, rows = find_item(code) if code and len(code)==6 else (None, [])
    vol = item.get('volatility') if item else None
    dd = item.get('drawdown') if item else None
    total_return = extract_html_total_return(path) if path else None
    score = item.get('score') if item else None
    if rows and len(rows) > 2:
        rets=[rows[i]['close']/rows[i-1]['close']-1 for i in range(1,len(rows)) if rows[i-1].get('close')]
        recent=rets[-60:] if len(rets)>=60 else rets
        win=round(sum(1 for r in recent if r>0)/len(recent)*100,2) if recent else None
        mean=avg(recent) or 0
        sd=pstdev(recent) if len(recent)>=2 else 0
        sharpe=round((mean/sd)*(252**0.5),2) if sd else None
        if total_return is None and len(rows) > 1 and rows[0].get('close'):
            total_return = round((rows[-1]['close']/rows[0]['close']-1)*100, 2)
    else:
        win=None; sharpe=None
    risk_score = round(max(0, min(100, (vol or 0)*6 + abs(dd or 0)*1.5 + max(0, 70-(score or 50))*0.6)), 2)
    risk_level = '高' if risk_score >= 70 else '中' if risk_score >= 40 else '低'
    if risk_level == '高': advice='高风险：控制仓位，等待波动和回撤收敛后再观察。'
    elif risk_level == '中': advice='中等风险：可小仓位跟踪，重点观察趋势延续和止损位。'
    else: advice='低风险：走势相对稳定，可结合评分和基本面继续筛选。'
    return {'risk_level':risk_level,'risk_score':risk_score,'volatility':vol,'max_drawdown':total_return,'total_return':total_return,'sharpe':sharpe,'win_rate':win,'advice':advice}

def reports_payload():
    found={}
    for root in report_roots():
        if root.exists():
            for p in root.rglob('*.html'):
                try:
                    rp=p.resolve()
                    item={'name':p.name,'display_name':report_display_name(p.name),'path':str(rp),'id':quote(p.name),'url':'/reports/'+quote(p.name),'size':p.stat().st_size,'mtime':datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),'format':'html'}; item.update(report_metrics_from_name(p.name, p.stat(), rp)); found[str(rp)]=item
                except Exception:
                    pass
    items=sorted(found.values(), key=lambda x:x['mtime'], reverse=True)
    return {'count':len(items),'items':items}

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

def application(env,start):
    path=urlparse(env.get('PATH_INFO','/')).path; q=parse_qs(env.get('QUERY_STRING',''))
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
    routes={'/health':lambda:{'status':'ok','data_dir':str(DATA_DIR), 'exists':DATA_DIR.exists()},'/api/stocks':lambda:stock_payload(q),'/api/stocks/delete':lambda:delete_stock_payload(read_body(env)),'/api/history/update':lambda:history_update_payload(read_body(env)),'/api/models':models_payload,'/api/features':lambda:features_payload(q.get('code',[''])[0]),'/api/reports':reports_payload,'/api/reports/delete':lambda:delete_report_payload(read_body(env)),'/api/config':config_payload,'/api/ai/train':lambda:ai_train_payload(read_body(env)),'/api/ai/status':lambda:ai_status_payload(q.get('job_id',[''])[0]),'/api/analyze':lambda:analyze_payload(read_body(env)),'/api/backtest':lambda:backtest_payload(read_body(env))}
    if path in routes: return resp(start,'200 OK',json.dumps(routes[path](),ensure_ascii=False),'application/json')
    return resp(start,'404 Not Found','Not Found','text/plain')

if __name__=='__main__':
    print(f'Web Stock Picker listening on 0.0.0.0:{PORT}')
    make_server('0.0.0.0',PORT,application).serve_forever()
