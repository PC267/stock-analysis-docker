# -*- coding: utf-8 -*-
import json, time, urllib.request, urllib.parse, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from .scoring import to_float, normalize_code, apply_strategy, load_items

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

# 行业数据缓存：stock_code -> industry_name
_industry_cache = {}
_industry_cache_ts = 0
_INDUSTRY_CACHE_TTL = 3600  # 1小时


def _fast_em_clist(params):
    """快速直连 EM API，不走 em_json 的 4 次重试 + 12s 超时"""
    url = 'https://push2delay.eastmoney.com/api/qt/clist/get?' + urllib.parse.urlencode(params)
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/',
        })
        return json.loads(urllib.request.urlopen(req, timeout=10).read().decode('utf-8', 'ignore'))
    except Exception:
        return None


def _warm_industry_cache():
    """后台预热行业数据缓存，重试失败页面。"""
    global _industry_cache, _industry_cache_ts
    result = dict(_industry_cache)
    page = 1
    max_pages = 30
    failed_retries = 2
    while page <= max_pages:
        for attempt in range(failed_retries + 1):
            try:
                url = ('https://push2delay.eastmoney.com/api/qt/clist/get?' +
                       urllib.parse.urlencode({
                           'fid': 'f12', 'po': '0', 'pz': '100', 'pn': str(page),
                           'np': '1', 'fltt': '2', 'invt': '2',
                           'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
                           'fs': 'm:0+t:6', 'fields': 'f12,f100',
                       }))
                req = urllib.request.Request(url, headers={
                    'User-Agent': 'Mozilla/5.0', 'Referer': 'https://data.eastmoney.com/',
                })
                d = json.loads(urllib.request.urlopen(req, timeout=5).read().decode('utf-8', 'ignore'))
                items = ((d or {}).get('data') or {}).get('diff', [])
                for x in items:
                    code = normalize_code(str(x.get('f12', '')))
                    industry = str(x.get('f100', '') or '')
                    if code and len(code) == 6 and industry and industry != '-':
                        result[code] = industry
                if len(items) < 100:
                    _industry_cache.update(result)
                    _industry_cache_ts = time.time()
                    return
                break  # success, exit retry loop
            except Exception:
                if attempt >= failed_retries:
                    break  # give up this page
                continue
        page += 1
    _industry_cache.update(result)
    _industry_cache_ts = time.time()


def fetch_stock_industries(codes=None):
    """批量获取A股行业数据（东方财富行业分类），缓存1小时。
    
    纯缓存查询，不发起API请求。
    
    Args:
        codes: 需要查询的股票代码列表，为None时获取全部。
    
    Returns:
        dict: {code: industry_name} 映射
    """
    if codes:
        return {c: _industry_cache.get(c, '') for c in codes}
    return dict(_industry_cache)


# 同步预热（模块导入时执行，约4秒）
# 如果网络不可用则静默失败，不影响启动
try:
    _warm_industry_cache()
except Exception:
    pass


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
