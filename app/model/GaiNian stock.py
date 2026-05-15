import requests
import json
import pandas as pd
import os
from typing import List, Tuple
import time


def get_root_directory():
    """
    获取项目根目录
    """
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 如果脚本在model目录中，返回上一级目录（项目根目录）
    if os.path.basename(script_dir) == "model":
        root_dir = os.path.dirname(script_dir)
    else:
        # 否则使用当前目录作为根目录
        root_dir = script_dir

    return root_dir


def get_concept_fund_flow_top3() -> List[Tuple[str, str]]:
    """
    获取东方财富网概念板块资金流入前三的概念代码

    Returns:
        List[Tuple[str, str]]: 包含(概念代码, 概念名称)的列表
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get"

    params = {
        'fid': 'f174',#当日排行f62,五日排行f164,十日排行f174
        'po': '1',
        'pz': '50',
        'pn': '1',
        'np': '1',
        'fltt': '2',
        'invt': '2',
        'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
        'fs': 'm:90+t:3',
        'fields': 'f12,f14,f62'
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://data.eastmoney.com/bkzj/hy.html'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        concept_list = data.get('data', {}).get('diff', [])

        if not concept_list:
            print("未获取到概念板块数据")
            return []

        # 按资金流入排序
        sorted_concepts = sorted(concept_list,
                                 key=lambda x: float(x.get('f62', 0)),
                                 reverse=True)

        # 获取前三名
        top3 = sorted_concepts[:3]

        result = []
        for item in top3:
            concept_code = item.get('f12', '')
            concept_name = item.get('f14', '')
            result.append((concept_code, concept_name))

        print(f"获取到资金流入前三的概念板块:")
        for i, (code, name) in enumerate(result, 1):
            print(f"{i}. {name} ({code})")

        return result

    except Exception as e:
        print(f"获取概念板块数据失败: {e}")
        return []


def get_stocks_by_concept(concept_code: str) -> List[Tuple[str, str]]:
    """
    根据概念代码获取该概念下的股票列表（按资金流入排序取前20）

    Args:
        concept_code: 概念代码，如BK1136

    Returns:
        List[Tuple[str, str]]: 包含(股票代码, 股票名称)的列表
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get"

    params = {
        'fid': 'f174',  # 按当日资金流入排序
        'po': '1',
        'pz': '100',  # 这里可以保持不变或改为20，但要确保能获取足够多股票用于排序
        'pn': '1',
        'np': '1',
        'fltt': '2',
        'invt': '2',
        'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
        'fs': f'b:{concept_code}',
        'fields': 'f12,f14,f62'  # 添加f62字段获取资金流入数据用于排序
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': f'https://data.eastmoney.com/bkzj/{concept_code}.html'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        stock_list = data.get('data', {}).get('diff', [])

        if not stock_list:
            print(f"未获取到概念{concept_code}下的股票数据")
            return []

        # 按资金流入排序（降序）
        sorted_stocks = sorted(stock_list,
                               key=lambda x: float(x.get('f62', 0)),
                               reverse=True)

        # 取前20只股票
        top20_stocks = sorted_stocks[:10]

        stocks = []
        for item in top20_stocks:
            stock_code = item.get('f12', '')
            stock_name = item.get('f14', '')

            if stock_code and stock_name:
                stocks.append((stock_code, stock_name))

        print(f"  按资金流入排序获取前10只股票，实际获取 {len(stocks)} 只")

        return stocks

    except Exception as e:
        print(f"获取概念{concept_code}股票数据失败: {e}")
        # 返回空列表
        return []

def get_stocks_by_concept(concept_code: str) -> List[Tuple[str, str]]:
    """
    根据概念代码获取该概念下的股票列表（按资金流入排序取前20）

    Args:
        concept_code: 概念代码，如BK1136

    Returns:
        List[Tuple[str, str]]: 包含(股票代码, 股票名称)的列表
    """
    url = "https://push2.eastmoney.com/api/qt/clist/get"

    params = {
        'fid': 'f174',  # 按当日资金流入排序
        'po': '1',
        'pz': '100',  # 获取足够多的股票用于排序
        'pn': '1',
        'np': '1',
        'fltt': '2',
        'invt': '2',
        'ut': '8dec03ba335b81bf4ebdf7b29ec27d15',
        'fs': f'b:{concept_code}',
        'fields': 'f12,f14,f62'  # 添加f62字段获取资金流入数据用于排序
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': f'https://data.eastmoney.com/bkzj/{concept_code}.html'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        stock_list = data.get('data', {}).get('diff', [])

        if not stock_list:
            print(f"未获取到概念{concept_code}下的股票数据")
            return []

        # 改进的排序逻辑，处理无效数据
        def get_flow_value(item):
            """安全获取资金流入值"""
            flow_value = item.get('f62', '0')
            if flow_value == '-' or flow_value == '' or flow_value is None:
                return 0.0  # 将无效值设为0
            try:
                return float(flow_value)
            except (ValueError, TypeError):
                return 0.0

        # 按资金流入排序（降序）
        sorted_stocks = sorted(stock_list,
                               key=get_flow_value,
                               reverse=True)

        # 取前20只股票
        top20_stocks = sorted_stocks[:20]

        stocks = []
        for item in top20_stocks:
            stock_code = item.get('f12', '')
            stock_name = item.get('f14', '')
            flow_value = get_flow_value(item)

            if stock_code and stock_name:
                stocks.append((stock_code, stock_name))
                # 可选：显示每只股票的资金流入
                if flow_value != 0:
                    print(f"    {stock_code} {stock_name}: {flow_value:,.0f}")

        print(f"  按资金流入排序获取前20只股票，实际获取 {len(stocks)} 只")

        # 显示资金流入信息
        if stocks:
            total_flow = sum(get_flow_value(item) for item in top20_stocks)
            print(f"  前20只股票总资金流入: {total_flow:,.0f}")

            # 显示前三名
            print(f"  资金流入前三:")
            for i, item in enumerate(top20_stocks[:3], 1):
                code = item.get('f12', '')
                name = item.get('f14', '')
                flow = get_flow_value(item)
                print(f"    {i}. {code} {name}: {flow:,.0f}")

        return stocks

    except Exception as e:
        print(f"获取概念{concept_code}股票数据失败: {e}")
        return []

def save_all_stocks(stocks: List[Tuple[str, str]]):
    """
    保存股票列表到文件

    Args:
        stocks: 包含(股票代码, 股票名称)的列表
    """
    if not stocks:
        print("没有股票数据需要保存")
        return

    # 获取项目根目录
    root_dir = get_root_directory()
    
    # 创建results目录（如果不存在）
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 文件路径
    file_path = os.path.join(results_dir, 'Stock_Data.txt')
    
    try:
        # 保存股票数据
        with open(file_path, 'w', encoding='utf-8') as f:
            for stock_code, stock_name in stocks:
                f.write(f"{stock_code},{stock_name}\n")
        
        print(f"成功保存 {len(stocks)} 只股票数据到: {file_path}")

    except Exception as e:
        print(f"保存文件失败: {e}")

def main():
    """
    主函数：获取概念板块资金流入前三，然后获取对应股票数据并保存
    """
    print("开始获取概念板块资金流入前三的股票数据")
    print("=" * 60)

    # 显示项目根目录信息
    root_dir = get_root_directory()
    print(f"项目根目录: {root_dir}")
    print(f"结果将保存到: {os.path.join(root_dir, 'results', 'Stock_Data.txt')}")
    print()

    # 1. 获取概念板块资金流入前三
    print("第一步：获取概念板块资金流入前三")
    top3_concepts = get_concept_fund_flow_top3()

    if not top3_concepts:
        print("未获取到概念板块数据，程序退出")
        return

    # 2. 为每个概念获取股票数据
    all_stocks = []
    stock_set = set()  # 用于去重

    for concept_code, concept_name in top3_concepts:
        print(f"\n正在处理概念板块: {concept_name} ({concept_code})")

        # 获取该概念下的股票
        stocks = get_stocks_by_concept(concept_code)

        if stocks:
            # 添加股票到总列表（去重）
            for stock_code, stock_name in stocks:
                if stock_code not in stock_set:
                    stock_set.add(stock_code)
                    all_stocks.append((stock_code, stock_name))

            print(f"  概念{concept_name}下获取到 {len(stocks)} 只股票")
            print(f"  去重后累计 {len(all_stocks)} 只股票")

        # 添加延迟，避免请求过于频繁
        time.sleep(1)

    # 3. 保存所有股票数据到文件
    if all_stocks:
        print("\n" + "=" * 60)
        print("正在保存股票数据...")

        save_all_stocks(all_stocks)

        # 显示前10只股票示例
        print("\n前10只股票示例:")
        for i, (code, name) in enumerate(all_stocks[:10], 1):
            print(f"{i}. {code},{name}")

        if len(all_stocks) > 10:
            print(f"... 还有{len(all_stocks) - 10}只股票")
    else:
        print("\n未获取到任何股票数据")

    print("\n" + "=" * 60)
    print("程序执行完成！")


# 使用示例
if __name__ == "__main__":
    main()