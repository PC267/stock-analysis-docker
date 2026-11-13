"""
优化要点：
- 删除重复的 get_stocks_by_concept 定义（只保留带异常保护的完整版）
- session 复用连接
- 请求返回内容不再缓存到局部变量之外
"""
import requests
import pandas as pd
import os
from typing import List, Tuple
import time


def get_root_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) == "model":
        return os.path.dirname(script_dir)
    return script_dir


_session = requests.Session()
_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})


def get_concept_fund_flow_top3() -> List[Tuple[str, str]]:
    url = "https://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "fid": "f174",
        "po": "1",
        "pz": "50",
        "pn": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": "8dec03ba335b81bf4ebdf7b29ec27d15",
        "fs": "m:90+t:3",
        "fields": "f12,f14,f2,f160,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f260,f261,f124,f1,f13",
    }
    headers = {"Referer": "https://data.eastmoney.com/bkzj/hy.html"}

    try:
        response = _session.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        concept_list = data.get("data", {}).get("diff", [])
        if not concept_list:
            print("未获取到概念板块数据")
            return []

        def get_flow_value(item):
            v = item.get("f174", "0")
            if v in ("-", "", None):
                return 0.0
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        sorted_concepts = sorted(concept_list, key=get_flow_value, reverse=True)
        result = [
            (item.get("f12", ""), item.get("f14", ""))
            for item in sorted_concepts[:3]
        ]
        print("获取到资金流入前三的概念板块:")
        for i, (code, name) in enumerate(result, 1):
            print(f"{i}. {name} ({code})")
        return result
    except Exception as e:
        print(f"获取概念板块数据失败: {e}")
        return []


def get_stocks_by_concept(concept_code: str) -> List[Tuple[str, str]]:
    url = "https://push2delay.eastmoney.com/api/qt/clist/get"
    params = {
        "fid": "f174",
        "po": "1",
        "pz": "100",
        "pn": "1",
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": "8dec03ba335b81bf4ebdf7b29ec27d15",
        "fs": f"b:{concept_code}",
        "fields": "f12,f14,f2,f160,f174,f175,f176,f177,f178,f179,f180,f181,f182,f183,f260,f261,f124,f1,f13",
    }
    headers = {"Referer": f"https://data.eastmoney.com/bkzj/{concept_code}.html"}

    try:
        response = _session.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        stock_list = data.get("data", {}).get("diff", [])
        if not stock_list:
            print(f"未获取到概念{concept_code}下的股票数据")
            return []

        def get_flow_value(item):
            v = item.get("f174", "0")
            if v in ("-", "", None):
                return 0.0
            try:
                return float(v)
            except (ValueError, TypeError):
                return 0.0

        sorted_stocks = sorted(stock_list, key=get_flow_value, reverse=True)
        top20 = sorted_stocks[:20]

        stocks = []
        for item in top20:
            code = item.get("f12", "")
            name = item.get("f14", "")
            if code and name:
                stocks.append((code, name))

        print(f"  按资金流入排序获取前20只股票，实际获取 {len(stocks)} 只")
        if stocks:
            total_flow = sum(get_flow_value(item) for item in top20)
            print(f"  前20只股票总资金流入: {total_flow:,.0f}")
        return stocks
    except Exception as e:
        print(f"获取概念{concept_code}股票数据失败: {e}")
        return []


def save_all_stocks(stocks: List[Tuple[str, str]]):
    if not stocks:
        print("没有股票数据需要保存")
        return

    root_dir = get_root_directory()
    results_dir = os.path.join(root_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, "Stock_Data.txt")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for code, name in stocks:
                f.write(f"{code},{name}\n")
        print(f"成功保存 {len(stocks)} 只股票数据到: {file_path}")
    except Exception as e:
        print(f"保存文件失败: {e}")


def main():
    print("开始获取概念板块资金流入前三的股票数据")
    print("=" * 60)
    root_dir = get_root_directory()
    print(f"项目根目录: {root_dir}")
    print(f"结果将保存到: {os.path.join(root_dir, 'results', 'Stock_Data.txt')}\n")

    print("第一步：获取概念板块资金流入前三")
    top3 = get_concept_fund_flow_top3()
    if not top3:
        print("未获取到概念板块数据，程序退出")
        return

    all_stocks = []
    stock_set = set()
    for concept_code, concept_name in top3:
        print(f"\n正在处理概念板块: {concept_name} ({concept_code})")
        stocks = get_stocks_by_concept(concept_code)
        if stocks:
            for code, name in stocks:
                if code not in stock_set:
                    stock_set.add(code)
                    all_stocks.append((code, name))
            print(f"  概念{concept_name}下获取 {len(stocks)} 只股票，累计去重 {len(all_stocks)}")
        time.sleep(1)

    if all_stocks:
        print("\n" + "=" * 60)
        save_all_stocks(all_stocks)
        print("\n前10只股票示例:")
        for i, (code, name) in enumerate(all_stocks[:10], 1):
            print(f"{i}. {code},{name}")
    else:
        print("\n未获取到任何股票数据")

    print("\n" + "=" * 60)
    print("程序执行完成！")


if __name__ == "__main__":
    main()