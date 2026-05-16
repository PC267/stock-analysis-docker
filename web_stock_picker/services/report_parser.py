# -*- coding: utf-8 -*-
import re
from pathlib import Path
from statistics import pstdev
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
from .scoring import normalize_code, find_item, avg

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


def extract_html_metrics(path):
    metrics = {}
    try:
        text = Path(path).read_text(encoding='utf-8', errors='ignore')
        def clean(v):
            return re.sub(r'<[^>]+>', '', v or '').replace('&nbsp;', ' ').strip()
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
            level_map = {'Extreme Risk':'极高', 'High Risk':'高', 'Medium Risk':'中', 'Low Risk':'低', 'Very Low Risk':'极低'}
            metrics['risk_level'] = level_map.get(risk_level_raw, risk_level_raw)
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
            metrics['advice'] = advice
    except Exception:
        pass
    return metrics


def extract_html_total_return(path):
    return extract_html_metrics(path).get('total_return')

def report_display_name(filename):
    m = re.match(r'^(\d{6})_([^_]+)', filename or '')
    return f"{m.group(1)}_{m.group(2)}" if m else (filename or '')

def report_metrics_from_name(name, stat, path=None):
    html_metrics = extract_html_metrics(path) if path else {}
    code = normalize_code(name)
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
    metrics = {'risk_level':risk_level,'risk_score':risk_score,'volatility':vol,'max_drawdown':total_return,'total_return':total_return,'sharpe':sharpe,'win_rate':win,'advice':advice}
    metrics.update({k:v for k,v in html_metrics.items() if v is not None and v != ''})
    return metrics
