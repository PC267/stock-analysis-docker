"""
优化要点：
- FIELD_MAPPING 统一为模块级常量，避免 parse/save 两处重复定义
- 大量调试 print 改为 DEBUG 开关控制（CIA_DEBUG=1 打开）
- 调试文件写入改为可选（CIA_SAVE_DEBUG=1 打开），不再默认写盘
- 不再同时保留 raw_content + content，解码后立即释放原始字节
- 字段统计一次 notna().sum() 完成
- 主流程结束后 del 大对象 + gc.collect()
"""
import requests
import json
import re
from typing import Dict, Any
import pandas as pd
import brotli
import gzip
import time
import random
import subprocess
import sys
import os
import gc

DEBUG = os.environ.get("CIA_DEBUG", "0") == "1"
SAVE_DEBUG_FILES = os.environ.get("CIA_SAVE_DEBUG", "0") == "1"

FIELD_MAPPING = {
    "f12": "股票代码",
    "f14": "股票名称",
    "f2": "最新价",
    "f3": "涨跌幅%",
    "f4": "涨跌额",
    "f127": "涨跌幅",
    "f267": "主力流入净额",
    "f268": "主力流入净额占比%",
    "f269": "超大单流入净额",
    "f270": "超大单流入占比%",
    "f271": "大单流入净额",
    "f272": "大单流入占比%",
    "f273": "中单流入净额",
    "f274": "中单流入占比%",
    "f275": "小单流入净额",
    "f276": "小单流入占比%",
    "f124": "时间戳",
    "f1": "市场标识",
    "f13": "市场类型",
}

STAT_FIELDS = ["各项流入净额总和", "各项总和减主力流入净额"]


def _log_debug(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def fetch_eastmoney_stock_data(page_num=1, fid="f164"):
    url = "https://push2.eastmoney.com/api/qt/clist/get"
    params = {
        "cb": f"jQuery{int(time.time() * 1000)}_{int(random.random() * 1e15)}",
        "fid": fid,
        "po": "1",
        "pz": "50",
        "pn": str(page_num),
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": "8dec03ba335b81bf4ebdf7b29ec27d15",
        "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
        "fields": "f12,f14,f2,f3,f4,f127,f267,f268,f269,f270,f271,f272,f273,f274,f275,f276,f124,f1,f13",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Referer": "https://data.eastmoney.com/zjlx/detail.html",
        "Accept": "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    try:
        _log_debug("正在发送请求...")
        response = requests.get(url, params=params, headers=headers, timeout=15)
        _log_debug(f"状态码: {response.status_code}")

        if response.status_code != 200:
            print(f"请求失败: {response.status_code}")
            return None

        content_encoding = response.headers.get("Content-Encoding", "").lower()
        _log_debug(f"Content-Encoding: {content_encoding}")

        raw_content = response.content
        try:
            if "br" in content_encoding:
                content = brotli.decompress(raw_content).decode("utf-8")
            elif "gzip" in content_encoding:
                content = gzip.decompress(raw_content).decode("utf-8")
            else:
                content = response.text
        finally:
            del raw_content

        if SAVE_DEBUG_FILES:
            try:
                with open("../decoded_response.txt", "w", encoding="utf-8") as f:
                    f.write(content[:10000])
                print("✓ 已保存解码后响应到 decoded_response.txt")
            except Exception as e:
                _log_debug(f"保存调试文件失败: {e}")

        _log_debug(f"解码后内容长度: {len(content)}")
        if not content:
            return None

        # JSONP / 直接 JSON 兼容解析
        patterns = [
            (r"jQuery\d+_\d+\((.*?)\);?\s*$", True),
            (r"^({.*})$", False),
            (r"(\{.*\})", False),
        ]
        for i, (pattern, use_group1) in enumerate(patterns):
            match = re.search(pattern, content, re.DOTALL)
            if not match:
                continue
            try:
                json_str = match.group(1) if use_group1 else match.group(0)
                json_str = json_str.rstrip(";")
                data = json.loads(json_str)
                _log_debug(f"✓ 使用模式{i + 1}解析成功")
                return parse_stock_data(data)
            except json.JSONDecodeError as e:
                _log_debug(f"! 模式{i + 1}解析失败: {e}")
                continue

        # 兜底：直接尝试解析
        try:
            data = json.loads(content)
            _log_debug("✓ 直接解析成功")
            return parse_stock_data(data)
        except json.JSONDecodeError as e:
            print(f"! 所有解析模式都失败: {e}")
            return None

    except Exception as e:
        print(f"请求异常: {str(e)}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None


def fetch_eastmoney_multiple_pages(start_page=1, end_page=4, fid="f164", delay=1):
    all_dfs = []

    print(f"\n{'=' * 60}")
    print("开始批量获取多页数据")
    print(f"页码范围: {start_page} - {end_page}")
    print(f"资金流类型: {fid}")
    print(f"{'=' * 60}")

    for page_num in range(start_page, end_page + 1):
        print(f"\n正在获取第 {page_num} 页数据...")
        df = fetch_eastmoney_stock_data(page_num=page_num, fid=fid)
        if df is not None and not df.empty:
            df["页码"] = page_num
            all_dfs.append(df)
            print(f"✓ 第{page_num}页成功获取 {len(df)} 条数据")
        else:
            print(f"✗ 第{page_num}页获取失败或数据为空")

        if page_num < end_page and delay > 0:
            _log_debug(f"等待 {delay} 秒...")
            time.sleep(delay)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        # 释放分页 DataFrame
        del all_dfs
        gc.collect()
        print(f"\n批量获取完成！总数据量: {len(combined_df)} 条")
        return combined_df

    print("\n✗ 未获取到任何数据")
    return pd.DataFrame()


def parse_stock_data(data: Dict[str, Any]) -> pd.DataFrame:
    _log_debug("开始解析数据...")

    if not data:
        return pd.DataFrame()

    if "data" not in data:
        _log_debug("数据格式异常，没有'data'字段")
        if isinstance(data, dict) and data:
            return pd.DataFrame([data])
        return pd.DataFrame()

    stock_data = data["data"]

    if isinstance(stock_data, dict) and "diff" in stock_data:
        stocks = stock_data["diff"]
        if isinstance(stocks, dict):
            stocks_list = list(stocks.values())
        elif isinstance(stocks, list):
            stocks_list = stocks
        else:
            return pd.DataFrame()
        df = pd.DataFrame(stocks_list)
    else:
        df = pd.DataFrame([stock_data])

    rename_dict = {k: v for k, v in FIELD_MAPPING.items() if k in df.columns}
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)

    for field_name in FIELD_MAPPING.values():
        if field_name not in df.columns:
            df[field_name] = None

    _log_debug(f"共获取 {len(df)} 条股票数据")

    if DEBUG and len(df) > 0:
        print(df.head(5).to_string(index=False))

    return df


def save_to_csv(df: pd.DataFrame):
    if df.empty:
        print("! 没有数据可保存")
        return

    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)
    csv_name = os.path.join(results_dir, "eastmoney_stock_data.csv")

    required_fields = list(FIELD_MAPPING.values())
    for field in required_fields:
        if field not in df.columns:
            df[field] = None

    # 计算统计列
    try:
        needed_fields = [
            "主力流入净额",
            "超大单流入净额",
            "大单流入净额",
            "中单流入净额",
            "小单流入净额",
        ]
        for field in needed_fields:
            if field not in df.columns:
                print(f"警告: 缺少字段 {field}，填充为0")
                df[field] = 0
            df[field] = pd.to_numeric(df[field], errors="coerce")

        df["各项流入净额总和"] = (
            df["主力流入净额"].fillna(0)
            + df["超大单流入净额"].fillna(0)
            + df["大单流入净额"].fillna(0)
            + df["中单流入净额"].fillna(0)
            + df["小单流入净额"].fillna(0)
        )
        df["各项总和减主力流入净额"] = (
            df["各项流入净额总和"] - df["主力流入净额"].fillna(0)
        )

        print("✓ 已计算统计列: 各项流入净额总和 / 各项总和减主力流入净额")
        print(f"  - 各项流入净额总和总计: {df['各项流入净额总和'].sum():,.0f}")
        print(f"  - 各项总和减主力流入净额总计: {df['各项总和减主力流入净额'].sum():,.0f}")
    except Exception as e:
        print(f"! 计算统计列出错: {e}")
        df["各项流入净额总和"] = None
        df["各项总和减主力流入净额"] = None

    final_fields = required_fields + STAT_FIELDS
    if "页码" in df.columns:
        final_fields.append("页码")
    for field in final_fields:
        if field not in df.columns:
            df[field] = None

    df_sorted = df[final_fields]
    df_sorted.to_csv(csv_name, index=False, encoding="utf-8-sig")

    print(f"\n✓ 数据已保存到 {csv_name}")
    print(f"✓ 文件大小: {os.path.getsize(csv_name) / 1024:.2f} KB")
    print(f"✓ 字段数: {len(df_sorted.columns)}, 记录数: {len(df_sorted)}")

    # 验证（读回后立即释放）
    try:
        df_verify = pd.read_csv(csv_name, encoding="utf-8-sig")
        print(f"读取验证: {len(df_verify)} 条记录, {len(df_verify.columns)} 字段")
        del df_verify
    except Exception as e:
        print(f"读取验证失败: {e}")


def run_financial_ranking():
    print("\n" + "=" * 60)
    print("开始运行Financial ranking.py程序...")
    print("=" * 60)

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        financial_ranking_path = os.path.join(current_dir, "Financial ranking.py")

        if not os.path.exists(financial_ranking_path):
            print(f"错误: Financial ranking.py不存在: {financial_ranking_path}")
            return False

        print(f"找到Financial ranking.py: {financial_ranking_path}")

        # 方法1: importlib 动态导入
        try:
            import importlib.util

            model_dir = os.path.join(current_dir, "model")
            if model_dir not in sys.path:
                sys.path.append(model_dir)

            module_name = "Financial_ranking"
            spec = importlib.util.spec_from_file_location(module_name, financial_ranking_path)
            financial_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = financial_module
            spec.loader.exec_module(financial_module)

            print("✓ 成功导入Financial ranking模块")

            if hasattr(financial_module, "main"):
                financial_module.main()
                print("✓ Financial ranking.py运行完成")
                return True
            else:
                print("错误: Financial ranking.py中没有找到main函数")
        except Exception as e:
            print(f"导入模块失败: {e}")
            print("尝试使用subprocess运行...")

            result = subprocess.run(
                [sys.executable, financial_ranking_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            if result.returncode == 0:
                print("✓ Financial ranking.py运行成功")
                print(result.stdout)
                return True
            else:
                print(f"错误: 返回码 {result.returncode}")
                print(result.stderr)
                return False
    except Exception as e:
        print(f"运行Financial ranking.py时出错: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return False


def main():
    print("开始获取东方财富股票数据...")
    print("=" * 50)

    df_to_process = fetch_eastmoney_multiple_pages()

    if df_to_process is not None and not df_to_process.empty:
        print("\n数据获取成功！")

        save_to_csv(df_to_process)

        # 简单分析
        print("\n" + "=" * 50)
        print("数据分析结果")
        print("=" * 50)
        print(f"总记录数: {len(df_to_process)}")

        if "页码" in df_to_process.columns:
            for page, count in df_to_process["页码"].value_counts().sort_index().items():
                print(f"  第{page}页: {count} 条")

        if "最新价" in df_to_process.columns:
            df_to_process["最新价"] = pd.to_numeric(df_to_process["最新价"], errors="coerce")
            price_valid = df_to_process["最新价"].dropna()
            if len(price_valid) > 0:
                print(f"有效价格: {len(price_valid)} 条, "
                      f"范围: {price_valid.min():.2f}-{price_valid.max():.2f}, "
                      f"均值: {price_valid.mean():.2f}")

        if "涨跌幅%" in df_to_process.columns:
            df_to_process["涨跌幅%"] = pd.to_numeric(df_to_process["涨跌幅%"], errors="coerce")
            change_valid = df_to_process["涨跌幅%"].dropna()
            if len(change_valid) > 0:
                print(f"涨跌幅范围: {change_valid.min():.2f}% - {change_valid.max():.2f}%")
                top_gain = df_to_process.nlargest(3, "涨跌幅%")
                print("涨幅前三:")
                for _, row in top_gain.iterrows():
                    print(f"  {row.get('股票代码', '')} {row.get('股票名称', '')}: {row.get('涨跌幅%', 0):.2f}%")

        if "主力流入净额" in df_to_process.columns:
            df_to_process["主力流入净额"] = pd.to_numeric(
                df_to_process["主力流入净额"], errors="coerce")
            inflow = df_to_process[df_to_process["主力流入净额"] > 0]
            outflow = df_to_process[df_to_process["主力流入净额"] < 0]
            print(f"主力净流入股票数: {len(inflow)}, 净流出: {len(outflow)}")
            if len(inflow) > 0:
                print("主力净流入前10:")
                for _, row in inflow.nlargest(10, "主力流入净额").iterrows():
                    print(f"  {row.get('股票代码', '')} {row.get('股票名称', '')}: "
                          f"{row.get('主力流入净额', 0):,.0f}")

        # 释放
        del df_to_process
        gc.collect()
    else:
        print("\n! 数据获取失败或为空")

    print("\n准备运行Financial ranking.py...")
    success = run_financial_ranking()

    if success:
        print("\n✓ 两个程序都运行完成！")
    else:
        print("\n! Financial ranking.py运行失败")

    print("\nCapital inflow analysis.py 执行完成！")
    gc.collect()


if __name__ == "__main__":
    print("检查依赖库...")
    for name in ("brotli", "pandas", "requests"):
        try:
            __import__(name)
            print(f"✓ {name} 已安装")
        except ImportError:
            print(f"! {name} 未安装")
    print("=" * 50)
    main()