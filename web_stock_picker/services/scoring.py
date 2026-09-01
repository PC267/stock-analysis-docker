# -*- coding: utf-8 -*-
import csv, math, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date
from pathlib import Path
from statistics import pstdev

# CSV 解析结果缓存：key=(解析路径, mtime, size), 避免重复读取
_csv_cache = {}
_csv_cache_lock = threading.Lock()
# 目录扫描缓存：按 DATA_DIR mtime 失效，避免每次全量 glob
_paths_cache = {'key': None, 'data': ({}, [])}
# load_items 结果缓存：30 秒 TTL，热刷新不重算
_items_cache = {'ts': 0.0, 'data': None}
_ITEMS_CACHE_TTL = 300

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'app' / 'stock_data'

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


def batch_fetch_realtime_quotes(codes):
    """并行批量获取多只股票实时行情，显著减少串行 HTTP 请求耗时。"""
    quotes = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        fut_map = {pool.submit(fetch_realtime_quote, code): code for code in codes}
        for fut in as_completed(fut_map):
            code = fut_map[fut]
            try:
                result = fut.result()
                if result:
                    quotes[code] = result
            except Exception:
                pass
    return quotes

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

def latest_stock_paths(cleanup=True):
    # 检查路径缓存：DATA_DIR 未变化则直接返回（路径缓存不过期，可进一步按文件修改时间失效）
    try:
        st = DATA_DIR.stat()
        dir_key = (st.st_mtime, st.st_size)
    except Exception:
        dir_key = None
    if dir_key is not None and _paths_cache['key'] == dir_key:
        return _paths_cache['data']

    grouped = {}
    deleted = []
    seen = set()
    for path in DATA_DIR.glob('*_baostock.csv'):
        try:
            rp = path.resolve()
            key_path = str(rp)
            if key_path in seen:
                continue
            seen.add(key_path)
            code, name = parse_name(path)
            code = normalize_code(code)
            if not code or len(code) != 6:
                continue
            st = path.stat()
            grouped.setdefault(code, []).append((st.st_mtime, key_path, path, name))
        except Exception:
            pass
    latest = {}
    for code, rows in grouped.items():
        rows.sort(key=lambda x: (x[0], x[1]), reverse=True)
        latest[code] = rows[0][2]
        if cleanup:
            for _, _, old_path, _ in rows[1:]:
                try:
                    old_path.unlink()
                    deleted.append({'code': code, 'path': str(old_path)})
                except FileNotFoundError:
                    pass
                except Exception as e:
                    deleted.append({'code': code, 'path': str(old_path), 'error': str(e)})
    result = (latest, deleted)
    if dir_key is not None:
        _paths_cache['key'] = dir_key
        _paths_cache['data'] = result
    return result


def enrich_stock_item(c, n, rows, src):
    item = score_stock(c, n, rows)
    if item:
        sim = kline_simulation(rows)
        item.update(sim)
        item['score'] = round(max(0, min(100, item['score']*.82 + item['kline_score']*.18)), 2)
        item['data_source'] = src
        item['data_fresh'] = src != 'csv'
        item['tags'].append(item['kline_signal'])
    return item


def find_item(code):
    code = normalize_code(code)
    paths, _ = latest_stock_paths(cleanup=True)
    for c, path in paths.items():
        if not code or c == code:
            _, n = parse_name(path)
            rows, src = merge_realtime_row(c, read_stock(path))
            item = enrich_stock_item(c, n, rows, src)
            if item:
                return item, rows
    return None, []


def _read_stock_cached(path):
    """带缓存的 read_stock，按文件 mtime + size 失效。"""
    try:
        st = path.stat()
        key = (str(path.resolve()), st.st_mtime, st.st_size)
    except Exception:
        return read_stock(path)
    with _csv_cache_lock:
        if key in _csv_cache:
            return [dict(r) for r in _csv_cache[key]]  # 返回副本防止外部修改
        rows = read_stock(path)
        _csv_cache[key] = rows
    return rows


def load_items(return_deleted=False):
    # 30 秒 TTL 缓存：热刷新跳过全量重算
    now = time.time()
    if _items_cache['data'] is not None and (now - _items_cache['ts']) < _ITEMS_CACHE_TTL:
        cached_out, cached_del = _items_cache['data']
        out = [dict(x) for x in cached_out]   # 深拷贝防止调用方修改缓存
        del_list = [dict(x) for x in cached_del]
        return (out, del_list) if return_deleted else out

    out = []
    paths, deleted = latest_stock_paths(cleanup=True)

    # Phase 1: 读取所有 CSV（带缓存）
    code_info = {}  # code -> {name, path, rows}
    for c, path in paths.items():
        _, n = parse_name(path)
        code_info[c] = {'name': n, 'path': path, 'rows': _read_stock_cached(path)}

    # Phase 2: 并行获取所有股票实时行情
    codes = list(code_info.keys())
    quotes = batch_fetch_realtime_quotes(codes)

    # Phase 3: 合并行情并计算评分
    for c in codes:
        info = code_info[c]
        rows = info['rows']
        q = quotes.get(c)
        if q:
            rows = list(rows)
            if rows and rows[-1]['date'] == q['date']:
                rows[-1].update({k: q[k] for k in ('open','high','low','close','volume','pctChg')})
            else:
                rows.append({k: q[k] for k in ('date','open','high','low','close','volume','pctChg')})
            rows.sort(key=lambda x: x['date'])
            src = q['source']
        else:
            src = 'csv'
        item = enrich_stock_item(c, info['name'], rows, src)
        if item:
            out.append(item)

    # 缓存结果（深拷贝确保后续调用方修改不影响缓存）
    _items_cache['ts'] = time.time()
    _items_cache['data'] = ([dict(x) for x in out], [dict(x) for x in deleted])
    return (out, deleted) if return_deleted else out

def apply_strategy(items,strategy):
    if strategy=='momentum': items=[x for x in items if (x.get('momentum20') or -999)>0 and x.get('ma20') and x['price']>=x['ma20']]; return sorted(items,key=lambda x:((x.get('momentum20') or -999),x['score']),reverse=True)
    if strategy=='volume_breakout': items=[x for x in items if (x.get('volume_ratio') or 0)>=1.3 and (x.get('momentum20') or -999)>=-3]; return sorted(items,key=lambda x:((x.get('volume_ratio') or 0),x['score']),reverse=True)
    if strategy=='oversold': items=[x for x in items if (x.get('rsi14') or 999)<=45 or '接近20日低位' in x.get('tags',[])]; return sorted(items,key=lambda x:(x['score'],-(x.get('rsi14') or 100)),reverse=True)
    return sorted(items,key=lambda x:x['score'],reverse=True)

def stock_payload(q):
    items, deleted_duplicates = load_items(return_deleted=True)
    kw=q.get('keyword',[''])[0].lower(); limit=max(1,min(int(q.get('limit',['50'])[0] or 50),500)); strategy=q.get('strategy',['balanced'])[0]
    if kw: items=[x for x in items if kw in x['code'].lower() or kw in x['name'].lower()]
    items=apply_strategy(items,strategy)[:limit]
    for i,x in enumerate(items,1): x['rank']=i
    return {'updated_at':datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'strategy':strategy,'count':len(items),'items':items,'deleted_duplicates':len(deleted_duplicates),'deleted_duplicate_items':deleted_duplicates}


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