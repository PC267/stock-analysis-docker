# -*- coding: utf-8 -*-
import re
from pathlib import Path
from statistics import pstdev
from urllib.parse import unquote
from html import unescape

ROOT = Path(__file__).resolve().parents[2]
from .scoring import normalize_code, find_item, avg
from .eastmoney import fetch_stock_industries

def report_roots():
    return [ROOT/'app'/'results'/'reports', ROOT/'app'/'results', ROOT/'app'/'backtest_results']

# safe_report_path 结果缓存：name -> Path|None，按根目录 mtime 失效，避免未命中时全盘 rglob
_report_path_cache = {'key': None, 'data': {}}


def _report_roots_key():
    try:
        return tuple((str(r), r.stat().st_mtime_ns, r.stat().st_size) for r in report_roots() if r.exists())
    except Exception:
        return None


def safe_report_path(rel):
    raw = unquote(rel or '').lstrip('/')
    candidates = [raw]
    try:
        fixed = raw.encode('latin-1').decode('utf-8')
        if fixed not in candidates:
            candidates.append(fixed)
    except Exception:
        pass

    # 1) 先按名字在根目录直接拼路径（O(1)），命中即返回
    for root in report_roots():
        if not root.exists():
            continue
        try:
            rr = root.resolve()
            for name in candidates:
                if not name or name.startswith('..') or '/' in name:
                    continue
                p = (root / name).resolve()
                if p.is_file() and (p == rr or rr in p.parents):
                    return p
        except Exception:
            pass

    # 2) 未命中且候选是纯文件名时，查缓存（按根目录状态失效），避免全盘 rglob
    only_names = all(n and '/' not in n and '\\' not in n for n in candidates)
    if only_names:
        key = _report_roots_key()
        if key is None:
            return None
        if _report_path_cache['key'] != key:
            # 重建：为三个根目录各自建立 name->path 映射（只扫一次，之后命中缓存）
            _report_path_cache['data'] = {}
            for root in report_roots():
                if not root.exists():
                    continue
                try:
                    rr = root.resolve()
                    for x in root.rglob('*'):
                        if not x.is_file():
                            continue
                        xp = x.resolve()
                        if not (xp == rr or rr in xp.parents):
                            continue
                        _report_path_cache['data'][x.name] = xp
                except Exception:
                    pass
            _report_path_cache['key'] = key
        for name in candidates:
            hit = _report_path_cache['data'].get(name)
            if hit:
                return hit
        return None

    # 3) 含子路径的候选，做有限 rglob 回退（少见路径）
    for root in report_roots():
        if not root.exists():
            continue
        try:
            rr = root.resolve()
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


EN_ZH_ADVICE = {
    'Extremely high risk, recommend avoiding operations': '风险极高：建议暂时回避，等待风险释放后再观察。',
    'Recommend strict position control, set stop loss': '风险较高：建议严格控制仓位，并设置止损位。',
    'Can participate with appropriate position control': '风险适中：可在控制仓位的前提下适度参与。',
    'Low risk, suitable for stable holding': '风险较低：适合稳健持有，并持续跟踪趋势变化。',
}

RISK_LEVEL_ZH = {
    'Extreme Risk': '极高',
    'High Risk': '高',
    'Medium Risk': '中',
    'Low Risk': '低',
    'Very Low Risk': '极低',
    'Extremely High Risk': '极高',
}


def translate_advice(advice):
    text = unescape(re.sub(r'<[^>]+>', '', str(advice or ''))).replace('&nbsp;', ' ').strip()
    if not text:
        return text
    if re.search(r'[\u4e00-\u9fff]', text):
        return text
    if text in EN_ZH_ADVICE:
        return EN_ZH_ADVICE[text]
    low = text.lower()
    if 'avoid' in low or 'extreme' in low:
        return '风险极高：建议暂时回避，等待风险释放后再观察。'
    if 'strict position control' in low or 'stop loss' in low or 'high risk' in low:
        return '风险较高：建议严格控制仓位，并设置止损位。'
    if 'appropriate position' in low or 'participate' in low or 'medium risk' in low:
        return '风险适中：可在控制仓位的前提下适度参与。'
    if 'low risk' in low or 'stable holding' in low:
        return '风险较低：适合稳健持有，并持续跟踪趋势变化。'
    if 'buy' in low:
        return '建议：可结合趋势、成交量和风险指标审慎关注。'
    if 'sell' in low:
        return '建议：短线风险偏高，可考虑降低仓位或等待企稳。'
    if 'hold' in low:
        return '建议：可继续持有观察，注意仓位和止损纪律。'
    return '建议：请结合趋势、成交量、风险指标和基本面综合判断。'


def translate_risk_level(level):
    text = unescape(str(level or '')).strip()
    return RISK_LEVEL_ZH.get(text, text)


def report_stock_code(name):
    m = re.search(r'(?<!\d)(\d{6})(?!\d)', str(name or ''))
    return m.group(1) if m else ''


# 简单行业关键词映射（匹配股票名称中的关键词）
_INDUSTRY_KEYWORDS = {
    '科技': '科技', '电子': '电子', '芯片': '半导体', '半导体': '半导体',
    '光电': '光电', '光学': '光电',
    '医疗': '医疗', '医药': '医药', '药': '医药', '生物': '生物医药',
    '新能源': '新能源', '光伏': '光伏', '风能': '新能源', '锂': '锂电池',
    '汽车': '汽车', '汽配': '汽车零部件', '摩托': '汽车零部件',
    '机器': '机器人', '智能': '人工智能', '数据': '大数据',
    '通信': '通信', '信息': '信息技术', '软件': '软件',
    '军工': '军工', '航天': '航天军工', '航空': '航天军工',
    '家电': '家电', '电器': '家电', '日电': '家电', '海尔': '白色家电', '美菱': '白色家电',
    '地产': '房地产', '房产': '房地产', '置业': '房地产',
    '银行': '银行', '保险': '保险', '证券': '证券',
    '消费': '消费', '食品': '食品饮料', '饮料': '食品饮料', '乳业': '食品饮料',
    '化工': '化工', '化学': '化工', '材料': '新材料', '新材': '新材料', '纤维': '化工',
    '电力': '电力', '能源': '能源', '电驱': '电气设备', '电气': '电气设备',
    '环保': '环保', '环境': '环保', '水': '环保', '水务': '环保',
    '机械': '机械设备', '装备': '机械设备', '机床': '机械设备', '机电': '机械设备',
    '建设': '基建', '建筑': '基建', '筑': '基建',
    '传媒': '传媒', '影视': '传媒',
    '农牧': '农牧', '农业': '农牧',
    '有色': '有色金属', '钢铁': '钢铁',
    '纺织': '纺织服装', '服装': '纺织服装',
    '交运': '交通运输', '运输': '交通运输',
    '玻璃': '玻璃', '巨石': '玻璃纤维', '玻纤': '玻璃纤维',
    '动力': '动力设备', '驱动': '电气设备', '电驱': '电气设备',
    '微': '电子', '元件': '电子元件', '电路': '电子',
    '新材': '新材料', '材料': '新材料',
    '精密': '精密制造', '精测': '精密测量',
    '创投': '金融', '投资': '金融',
    '制药': '医药', '药业': '医药',
    '航空': '航天军工', '航天': '航天军工',
    '食品': '食品饮料', '饮品': '食品饮料', '糖': '食品饮料',
    '网络': '信息技术', '互联': '信息技术', '数码': '电子',
    '制造': '制造业', '工业': '工业',
    '电机': '电气设备', '电驱': '电气设备', '驱动': '电气设备',
    '膜': '环保设备', '水务': '环保', '水处理': '环保',
    '配件': '汽车零部件', '轴承': '机械设备',
    '蔚蓝': '生物医药',
    '维格': '电子', '迪凯': '电子', '派格': '信息技术',
    '研奥': '轨道交通', '醋化': '化工', '华康': '食品饮料',
    '劲拓': '机械设备', '诺瓦': '电子', '天通': '电子',
    '鼎泰': '机械设备', '深科': '电子', '鸿利': '电子',
    '凡拓': '信息技术', '佳创': '信息技术', '清越': '电子',
    '盛美': '半导体', '中科': '半导体', '华海': '半导体',
}

def get_stock_industry(code, name):
    """根据股票代码获取所属行业（东方财富API缓存 + 关键词备用）"""
    if not code or len(code) != 6:
        return []
    # 优先使用EM API缓存数据
    result = fetch_stock_industries([code])
    industry = result.get(code, '')
    if industry:
        return [industry]
    # API未覆盖时用关键词匹配备用
    if name:
        matched = [v for k, v in _INDUSTRY_KEYWORDS.items() if k in name and v]
        if matched:
            return [matched[0]]
    return ['']


def extract_html_metrics(path):
    metrics = {}
    try:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
        def clean(v):
            return unescape(re.sub(r'<[^>]+>', '', v or '').replace('&nbsp;', ' ')).strip()
        def stat(label):
            m = re.search(r'<div class="stat-label"[^>]*>[^<]*' + label + r'[^<]*</div>\s*<div class="stat-value[^"]*"[^>]*>(.*?)</div>', text, re.I | re.S)
            return clean(m.group(1)) if m else None
        def num(label, percent=False):
            v = stat(label)
            if not v:
                return None
            m = re.search(r'([+-]?\d+(?:\.\d+)?)', v.replace(',', ''))
            if not m:
                return None
            return round(float(m.group(1)), 2 if percent else 3)
        risk_level_raw = stat('Risk Level')
        if risk_level_raw:
            metrics['risk_level'] = translate_risk_level(risk_level_raw)
        risk_score_raw = stat('Risk Score')
        if risk_score_raw:
            m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*/\s*1(?:\.0)?', risk_score_raw)
            if m:
                metrics['risk_score'] = round(float(m.group(1)) * 100, 2)
            else:
                m = re.search(r'([+-]?\d+(?:\.\d+)?)', risk_score_raw.replace(',', ''))
                if m:
                    v = float(m.group(1))
                    metrics['risk_score'] = round(v * 100 if v <= 1 else v, 2)
        v = num('Total Return', percent=True)
        if v is not None:
            metrics['max_drawdown'] = v
            metrics['total_return'] = v
        v = num('Volatility', percent=True)
        if v is not None:
            metrics['volatility'] = v
        v = num('Sharpe Ratio')
        if v is not None:
            metrics['sharpe'] = v
        v = num('Win Rate', percent=True)
        if v is not None:
            metrics['win_rate'] = v
        advice = stat('📋 Investment Suggestion') or stat('Investment Suggestion')
        if advice:
            metrics['advice'] = translate_advice(advice)
    except Exception:
        pass
    return metrics


def extract_html_total_return(path):
    return extract_html_metrics(path).get('total_return')

def report_display_name(filename):
    m = re.match(r'^(\d{6})_([^_]+)', filename or '')
    return f"{m.group(1)}_{m.group(2)}" if m else (filename or '')

def report_metrics_from_name(name, stat, path=None, html_only=False):
    html_metrics = extract_html_metrics(path) if path else {}
    if html_only:
        # HTML 已包含全部指标，跳过 find_item()（读 CSV+HTTP 取行情+全量计算）
        metrics = {'risk_level':'低','risk_score':0,'volatility':None,'max_drawdown':html_metrics.get('total_return'),'total_return':html_metrics.get('total_return'),'sharpe':None,'win_rate':None,'advice':'请查阅报告详情'}
        metrics.update({k:(translate_advice(v) if k == 'advice' else translate_risk_level(v) if k == 'risk_level' else v) for k,v in html_metrics.items() if v is not None and v != ''})
        return metrics
    code = report_stock_code(name) or normalize_code(name)
    item, rows = find_item(code) if code and len(code)==6 else (None, [])
    vol = item.get('volatility') if item else None
    dd = item.get('drawdown') if item else None
    total_return = html_metrics.get('total_return')
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
    metrics = {'risk_level':risk_level,'risk_score':risk_score,'volatility':vol,'max_drawdown':total_return,'total_return':total_return,'sharpe':sharpe,'win_rate':win,'advice':translate_advice(advice)}
    metrics.update({k:(translate_advice(v) if k == 'advice' else translate_risk_level(v) if k == 'risk_level' else v) for k,v in html_metrics.items() if v is not None and v != ''})
    return metrics
