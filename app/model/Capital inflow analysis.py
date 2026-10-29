import requests
import json
import re
from typing import Dict, Any
import pandas as pd
import brotli  # 需要安装：pip install brotli
import gzip
import io
import time  # 添加time模块
import random  # 添加random模块
import subprocess  # 添加subprocess模块
import sys  # 添加sys模块
import os  # 添加os模块


def fetch_eastmoney_stock_data(page_num=1, fid="f164"):
    """
    获取东方财富股票资金流向数据（处理Brotli压缩）
    """
    # 构建请求URL
    url = "https://push2.eastmoney.com/api/qt/clist/get"

    # 请求参数 - 包含回调参数，这样返回的是JSONP格式
    params = {
        "cb": f"jQuery{int(time.time() * 1000)}_{int(random.random() * 1e15)}",
        "fid": fid,  # f62是当日资金流入 #f267是三日资金流入 #f164是五日资金流入 #f174是十日资金流入
        "po": "1",
        "pz": "50",  # 先获取50条测试
        "pn": str(page_num),  # 动态设置页码数
        "np": "1",
        "fltt": "2",
        "invt": "2",
        "ut": "8dec03ba335b81bf4ebdf7b29ec27d15",
        "fs": "m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:7+f:!2,m:1+t:3+f:!2",
        # 修改：请求所有需要的字段
        "fields": "f12,f14,f2,f3,f4,f127,f267,f268,f269,f270,f271,f272,f273,f274,f275,f276,f124,f1,f13"
    }

    # 请求头 - 注意：Accept-Encoding中包含br
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
        # 发送请求
        print("正在发送请求...")
        print(f"请求URL: {url}")
        print(f"请求参数: {params}")
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=15
        )

        print(f"状态码: {response.status_code}")
        print(f"响应头 - Content-Type: {response.headers.get('Content-Type')}")
        print(f"响应头 - Content-Encoding: {response.headers.get('Content-Encoding')}")

        if response.status_code == 200:
            # 检查Content-Encoding
            content_encoding = response.headers.get('Content-Encoding', '').lower()
            print(f"Content-Encoding: {content_encoding}")

            # 获取原始内容
            raw_content = response.content

            # 根据编码方式解码
            if 'br' in content_encoding:
                print("检测到Brotli压缩，正在解压...")
                try:
                    # 使用brotli解压
                    decompressed = brotli.decompress(raw_content)
                    content = decompressed.decode('utf-8')
                    print("✓ Brotli解压成功")
                except Exception as e:
                    print(f"! Brotli解压失败: {e}")
                    # 尝试其他方式
                    content = response.text
            elif 'gzip' in content_encoding:
                print("检测到Gzip压缩，正在解压...")
                try:
                    decompressed = gzip.decompress(raw_content)
                    content = decompressed.decode('utf-8')
                    print("✓ Gzip解压成功")
                except Exception as e:
                    print(f"! Gzip解压失败: {e}")
                    content = response.text
            else:
                # 没有压缩或自动解码
                content = response.text

            # 保存原始内容用于调试
            try:
                with open('../raw_response.txt', 'wb') as f:
                    f.write(raw_content)
                print("✓ 原始响应已保存到 raw_response.txt")

                with open('../decoded_response.txt', 'w', encoding='utf-8') as f:
                    f.write(content[:10000])  # 保存前10000字符
                print("✓ 解码后响应已保存到 decoded_response.txt")
            except Exception as e:
                print(f"! 保存调试文件失败: {e}")

            print(f"解码后内容长度: {len(content)} 字符")
            if len(content) > 0:
                print(f"解码后内容前300字符:\n{content[:300]}")

            # 解析JSONP格式
            if content and len(content) > 0:
                # 查找JSONP格式的模式
                import re

                # 模式1: jQueryxxxxx_xxxxxx({...})
                pattern1 = r'jQuery\d+_\d+\((.*?)\);?\s*$'
                # 模式2: 直接JSON
                pattern2 = r'^({.*})$'
                # 模式3: 任何包含大括号的内容
                pattern3 = r'(\{.*\})'

                patterns = [pattern1, pattern2, pattern3]

                for i, pattern in enumerate(patterns):
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        try:
                            json_str = match.group(1) if i == 0 else match.group(0)
                            # 清理可能的尾部分号
                            json_str = json_str.rstrip(';')

                            print(f"尝试使用模式{i + 1}解析，JSON字符串长度: {len(json_str)}")
                            if len(json_str) > 500:
                                print(f"JSON前300字符: {json_str[:300]}")

                            data = json.loads(json_str)
                            print(f"✓ 使用模式{i + 1}解析成功")
                            return parse_stock_data(data)
                        except json.JSONDecodeError as e:
                            print(f"! 模式{i + 1}解析失败: {e}")
                            continue
                        except Exception as e:
                            print(f"! 模式{i + 1}其他错误: {e}")
                            continue

                print("! 所有解析模式都失败")

                # 尝试直接解析整个内容（如果不是JSONP）
                try:
                    data = json.loads(content)
                    print("✓ 直接解析整个内容成功")
                    return parse_stock_data(data)
                except json.JSONDecodeError as e:
                    print(f"! 直接解析也失败: {e}")

            return None
        else:
            print(f"请求失败: {response.status_code}")
            return None

    except Exception as e:
        print(f"请求异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def fetch_eastmoney_multiple_pages(start_page=1, end_page=4, fid="f164", delay=1):
    """
    批量获取多页东方财富股票资金流向数据

    参数:
    start_page: 起始页码，默认第1页
    end_page: 结束页码，默认第2页
    fid: 资金流类型，f62是当日资金流入，f267是三日资金流入
    delay: 请求间隔秒数，避免请求过快

    返回:
    所有页面的股票数据DataFrame
    """
    all_dfs = []

    print(f"\n{'=' * 60}")
    print(f"开始批量获取多页数据")
    print(f"页码范围: {start_page} - {end_page}")
    print(f"资金流类型: {'当日资金流入' if fid == 'f62' else '三日资金流入'}")
    print(f"{'=' * 60}")

    for page_num in range(start_page, end_page + 1):
        print(f"\n正在获取第 {page_num} 页数据...")

        # 使用现有的fetch_eastmoney_stock_data函数获取单页数据
        df = fetch_eastmoney_stock_data(page_num=page_num, fid=fid)

        if df is not None and not df.empty:
            # 添加页码列以便区分数据来源
            df["页码"] = page_num
            all_dfs.append(df)
            print(f"✓ 第{page_num}页成功获取 {len(df)} 条数据")
        else:
            print(f"✗ 第{page_num}页获取失败或数据为空")

        # 如果不是最后一页，添加延时
        if page_num < end_page and delay > 0:
            print(f"等待 {delay} 秒后获取下一页...")
            time.sleep(delay)

    # 合并所有页面的数据
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n{'=' * 60}")
        print(f"批量获取完成！")
        print(f"页码范围: {start_page} - {end_page}")
        print(f"资金流类型: {'当日资金流入' if fid == 'f62' else '三日资金流入'}")
        print(f"总数据量: {len(combined_df)} 条")
        print(f"数据分布:")
        for page in range(start_page, end_page + 1):
            page_data = combined_df[combined_df["页码"] == page]
            print(f"  第{page}页: {len(page_data)} 条数据")
        print(f"{'=' * 60}")
        return combined_df
    else:
        print(f"\n✗ 未获取到任何数据")
        return pd.DataFrame()


def parse_stock_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    解析股票数据
    """
    print("\n开始解析数据...")

    if not data:
        print("数据为空")
        return pd.DataFrame()

    print(f"数据顶级键: {list(data.keys())}")

    if "data" not in data:
        print("数据格式异常，没有'data'字段")
        # 可能是不同的结构，直接尝试解析
        if isinstance(data, dict) and len(data) > 0:
            print("尝试直接使用获取到的数据...")
            # 创建一个DataFrame
            df = pd.DataFrame([data])
            print(f"创建的DataFrame形状: {df.shape}")
            return df
        else:
            return pd.DataFrame()

    stock_data = data["data"]

    print(f"data字段类型: {type(stock_data)}")

    if isinstance(stock_data, dict):
        print(f"data字段中的键: {list(stock_data.keys())}")

        # 提取股票列表
        if "diff" in stock_data:
            stocks = stock_data["diff"]
            print(f"找到 diff 字段，类型: {type(stocks)}")

            if isinstance(stocks, dict):
                # 如果是字典格式，转换为列表
                stocks_list = list(stocks.values())
                print(f"转换为列表，共 {len(stocks_list)} 条记录")
                df = pd.DataFrame(stocks_list)
            elif isinstance(stocks, list):
                print(f"直接是列表，共 {len(stocks)} 条记录")
                df = pd.DataFrame(stocks)
            else:
                print(f"未知的diff类型: {type(stocks)}")
                return pd.DataFrame()
        else:
            print("没有找到diff字段，尝试使用其他字段")
            # 尝试直接将整个data转换为DataFrame
            df = pd.DataFrame([stock_data])
    else:
        print(f"data字段不是字典类型，直接转换")
        df = pd.DataFrame([stock_data])

    # 显示获取到的原始字段
    print(f"\n原始字段 ({len(df.columns)}个): {list(df.columns)}")

    # 字段映射（根据东方财富的字段说明）- 包含所有字段
    field_mapping = {
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
        "f13": "市场类型"  # 0:深市, 1:沪市
    }

    # 重命名列
    rename_dict = {}
    for col in df.columns:
        if isinstance(col, str) and col in field_mapping:
            rename_dict[col] = field_mapping[col]

    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        print(f"重命名了 {len(rename_dict)} 个字段")

    # 确保所有field_mapping中的字段都在DataFrame中（如果不存在则添加空列）
    for field_key, field_name in field_mapping.items():
        if field_name not in df.columns:
            df[field_name] = None
            print(f"添加缺失字段: {field_name}")

    # 显示基本信息
    print(f"\n共获取 {len(df)} 条股票数据")
    print(f"处理后字段 ({len(df.columns)}个): {list(df.columns)}")

    # 检查所有需要的字段是否都存在
    required_fields = list(field_mapping.values())
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        print(f"警告: 以下字段缺失: {missing_fields}")

    # 打印所有字段的统计信息
    print("\n字段统计信息:")
    for field in required_fields:
        if field in df.columns:
            non_null_count = df[field].notna().sum()
            non_null_percent = (non_null_count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {field}: {non_null_count}/{len(df)} 条非空数据 ({non_null_percent:.1f}%)")

    # 如果有数据，显示前几条
    if len(df) > 0:
        print("\n=== 数据预览 ===")

        # 显示所有数据（如果少于20条）
        if len(df) <= 10:
            print(df.to_string(index=False))
        else:
            # 显示前5条
            print("前5条数据:")
            print(df.head().to_string(index=False))

    return df


def save_to_csv(df: pd.DataFrame):
    """
    保存数据到CSV文件
    """
    if not df.empty:
        try:
            import os

            # 创建results目录如果不存在
            results_dir = "../results"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                print(f"✓ 创建目录: {results_dir}")

            csv_name = os.path.join(results_dir, "eastmoney_stock_data.csv")

            # 定义完整的字段映射，确保所有字段都在
            field_mapping = {
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
                "f13": "市场类型"
            }

            # 获取所有需要的字段名（按指定顺序）
            required_fields = list(field_mapping.values())

            # 确保DataFrame中包含所有需要的字段
            for field in required_fields:
                if field not in df.columns:
                    df[field] = None

            # 计算统计列：各项流入净额的总和 - 三日主力流入净额
            try:
                # 检查需要的字段是否存在
                needed_fields = [
                    "主力流入净额",
                    "超大单流入净额",
                    "大单流入净额",
                    "中单流入净额",
                    "小单流入净额"
                ]

                # 确保所有需要的字段都存在且为数值类型
                for field in needed_fields:
                    if field not in df.columns:
                        print(f"警告: 缺少字段 {field}，无法计算统计值")
                        df[field] = 0
                    else:
                        # 转换为数值类型
                        df[field] = pd.to_numeric(df[field], errors='coerce')

                # 计算总和
                df["各项流入净额总和"] = (
                        df["主力流入净额"].fillna(0) +
                        df["超大单流入净额"].fillna(0) +
                        df["大单流入净额"].fillna(0) +
                        df["中单流入净额"].fillna(0) +
                        df["小单流入净额"].fillna(0)
                )

                # 计算差值：总和 - 主力流入净额
                df["各项总和减主力流入净额"] = df["各项流入净额总和"] - df["主力流入净额"].fillna(0)

                print("✓ 已计算统计列:")
                print(f"  - 各项流入净额总和 = 主力 + 超大单 + 大单 + 中单 + 小单")
                print(f"  - 各项总和减主力流入净额 = 总和 - 主力流入净额")

                # 显示统计信息
                if len(df) > 0:
                    total_sum = df["各项流入净额总和"].sum()
                    total_diff = df["各项总和减主力流入净额"].sum()
                    print(f"  - 各项流入净额总和总计: {total_sum:,.0f}")
                    print(f"  - 各项总和减主力流入净额总计: {total_diff:,.0f}")

            except Exception as e:
                print(f"! 计算统计列时出错: {e}")
                import traceback
                traceback.print_exc()
                # 添加空列
                df["各项流入净额总和"] = None
                df["各项总和减主力流入净额"] = None

            # 重新排列列顺序，按照field_mapping的顺序，然后添加统计列
            final_fields = required_fields + ["各项流入净额总和", "各项总和减主力流入净额"]

            # 如果DataFrame中有页码列，也加入最终字段
            if "页码" in df.columns:
                final_fields.append("页码")

            # 确保所有字段都存在
            for field in final_fields:
                if field not in df.columns:
                    df[field] = None

            # 重新排列列顺序
            df_sorted = df[final_fields] if all(field in df.columns for field in final_fields) else df

            # 保存到CSV文件
            df_sorted.to_csv(csv_name, index=False, encoding='utf-8-sig')
            print(f"\n✓ 数据已保存到 {csv_name}")
            print(f"✓ 文件大小: {os.path.getsize(csv_name) / 1024:.2f} KB")
            print(f"✓ 包含字段数: {len(df_sorted.columns)}")
            print(f"✓ 包含记录数: {len(df_sorted)}")
            print(f"✓ 最后2个字段为新增统计列")
            if "页码" in df_sorted.columns:
                print(f"✓ 包含页码列，可用于区分数据来源")

            # 显示字段数据填充情况
            print("\n字段数据填充情况:")
            for field in final_fields:
                if field in df_sorted.columns:
                    non_null_count = df_sorted[field].notna().sum()
                    non_null_percent = (non_null_count / len(df_sorted)) * 100 if len(df_sorted) > 0 else 0
                    data_type = df_sorted[field].dtype if hasattr(df_sorted[field], 'dtype') else 'unknown'
                    print(f"  {field}: {non_null_count}/{len(df_sorted)} ({non_null_percent:.1f}%), 类型: {data_type}")

            # 验证保存的数据
            print("\n=== 验证保存的数据 ===")
            try:
                df_verify = pd.read_csv(csv_name, encoding='utf-8-sig')
                print(f"读取验证: 成功读取 {len(df_verify)} 条记录")
                print(f"读取验证: 字段数: {len(df_verify.columns)}")

                # 显示最后2个新增字段的样本数据
                if len(df_verify) > 0:
                    print(f"\n读取验证: 前3条数据的新增统计字段:")
                    new_fields = ["各项流入净额总和", "各项总和减主力流入净额"]
                    if all(field in df_verify.columns for field in new_fields):
                        sample_df = df_verify[["股票代码", "股票名称"] + new_fields].head(3)
                        print(sample_df.to_string(index=False,
                                                  float_format=lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN"))
                    else:
                        print("警告: 新增统计字段未找到")

                # 检查新增字段的计算
                if len(df_verify) > 0:
                    print(f"\n统计字段计算验证:")
                    if all(field in df_verify.columns for field in
                           ["主力流入净额", "超大单流入净额", "大单流入净额",
                            "中单流入净额", "小单流入净额", "各项流入净额总和"]):
                        sample_idx = 0
                        if sample_idx < len(df_verify):
                            row = df_verify.iloc[sample_idx]
                            print(f"第{sample_idx + 1}条数据验证:")
                            print(f"  主力流入净额: {row.get('主力流入净额', 0):,.0f}")
                            print(f"  超大单流入净额: {row.get('超大单流入净额', 0):,.0f}")
                            print(f"  大单流入净额: {row.get('大单流入净额', 0):,.0f}")
                            print(f"  中单流入净额: {row.get('中单流入净额', 0):,.0f}")
                            print(f"  小单流入净额: {row.get('小单流入净额', 0):,.0f}")
                            print(f"  计算的总和: {row.get('各项流入净额总和', 0):,.0f}")
                            print(f"  总和减主力: {row.get('各项总和减主力流入净额', 0):,.0f}")

            except Exception as e:
                print(f"读取验证失败: {e}")

        except Exception as e:
            print(f"! 保存CSV文件失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("! 没有数据可保存")


def run_financial_ranking():
    """
    运行Financial ranking.py程序
    """
    print("\n" + "=" * 60)
    print("开始运行Financial ranking.py程序...")
    print("=" * 60)

    try:
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        financial_ranking_path = os.path.join(current_dir, "Financial ranking.py")

        if not os.path.exists(financial_ranking_path):
            print(f"错误: Financial ranking.py文件不存在: {financial_ranking_path}")
            return False

        print(f"找到Financial ranking.py文件: {financial_ranking_path}")

        # 方法1: 直接导入模块（推荐）
        try:
            import importlib.util
            import sys

            # 将model目录添加到Python路径
            model_dir = os.path.join(current_dir, "model")
            if model_dir not in sys.path:
                sys.path.append(model_dir)

            # 动态导入模块
            module_name = "Financial_ranking"
            spec = importlib.util.spec_from_file_location(module_name, financial_ranking_path)
            financial_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = financial_module
            spec.loader.exec_module(financial_module)

            print("✓ 成功导入Financial ranking模块")

            # 运行模块的主函数
            if hasattr(financial_module, 'main'):
                financial_module.main()
                print("✓ Financial ranking.py运行完成")
                return True
            else:
                print("错误: Financial ranking.py中没有找到main函数")

        except Exception as e:
            print(f"导入模块失败: {e}")
            print("尝试使用subprocess运行...")

            # 方法2: 使用subprocess运行
            result = subprocess.run([sys.executable, financial_ranking_path],
                                    capture_output=True, text=True, encoding='utf-8')

            if result.returncode == 0:
                print("✓ Financial ranking.py运行成功")
                print("输出内容:")
                print(result.stdout)
                return True
            else:
                print(f"错误: Financial ranking.py运行失败，返回码: {result.returncode}")
                print(f"错误输出: {result.stderr}")
                return False

    except Exception as e:
        print(f"运行Financial ranking.py时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数
    """
    print("开始获取东方财富股票数据...")
    print("=" * 50)

    # 直接使用多页获取函数，使用其默认参数值
    # 不指定start_page和end_page，让函数使用默认值
    print("使用fetch_eastmoney_multiple_pages的默认参数获取多页数据")
    df_to_process = fetch_eastmoney_multiple_pages()

    if df_to_process is not None and not df_to_process.empty:
        print("\n" + "=" * 50)
        print("数据获取成功！")
        print("=" * 50)

        # 保存到CSV
        save_to_csv(df_to_process)

        # 进行数据分析
        print("\n" + "=" * 50)
        print("数据分析结果")
        print("=" * 50)

        print(f"总记录数: {len(df_to_process)}")

        # 如果有多页数据，显示分页统计
        if "页码" in df_to_process.columns:
            page_counts = df_to_process["页码"].value_counts().sort_index()
            print(f"数据分布（按页码）:")
            for page, count in page_counts.items():
                print(f"  第{page}页: {count} 条数据")

        # 检查关键字段是否存在，并转换为数值类型
        if "最新价" in df_to_process.columns:
            # 转换最新价为数值类型，错误值转换为NaN
            df_to_process["最新价"] = pd.to_numeric(df_to_process["最新价"], errors='coerce')

            # 过滤掉NaN值后再计算统计
            price_valid = df_to_process["最新价"].dropna()
            if len(price_valid) > 0:
                print(f"有效价格数据: {len(price_valid)} 条")
                print(f"股票价格范围: {price_valid.min():.2f} - {price_valid.max():.2f}")
                print(f"平均价格: {price_valid.mean():.2f}")
                print(f"中位价格: {price_valid.median():.2f}")
            else:
                print(f"价格数据无效或为空")

        if "涨跌幅%" in df_to_process.columns:
            # 转换涨跌幅为数值类型
            df_to_process["涨跌幅%"] = pd.to_numeric(df_to_process["涨跌幅%"], errors='coerce')

            change_valid = df_to_process["涨跌幅%"].dropna()
            if len(change_valid) > 0:
                print(f"有效涨跌幅数据: {len(change_valid)} 条")
                print(f"涨跌幅范围: {change_valid.min():.2f}% - {change_valid.max():.2f}%")

                # 找出涨跌幅最大的股票
                top_gain = df_to_process.nlargest(3, "涨跌幅%")
                print("\n涨幅前三:")
                for idx, row in top_gain.iterrows():
                    name = row.get("股票名称", "未知")
                    code = row.get("股票代码", "未知")
                    change = row.get("涨跌幅%", 0)
                    page_info = f" (第{row.get('页码', '?')}页)" if "页码" in row else ""
                    if pd.notna(change):
                        print(f"  {code} {name}: {change:.2f}%{page_info}")
                    else:
                        print(f"  {code} {name}: 数据无效{page_info}")
            else:
                print("涨跌幅数据无效或为空")

        # 检查是否有资金流向数据
        flow_fields = [col for col in df_to_process.columns if "流入" in col or "流出" in col]
        if flow_fields:
            print(f"\n资金流向相关字段: {', '.join(flow_fields)}")

            # 转换资金流相关字段为数值类型
            for field in flow_fields:
                if field in df_to_process.columns:
                    df_to_process[field] = pd.to_numeric(df_to_process[field], errors='coerce')

            if "主力流入净额" in df_to_process.columns:
                # 过滤掉NaN值
                inflow_valid = df_to_process["主力流入净额"].dropna()
                inflow = df_to_process[df_to_process["主力流入净额"] > 0]
                outflow = df_to_process[df_to_process["主力流入净额"] < 0]

                print(f"有效主力流入数据: {len(inflow_valid)} 条")
                print(f"主力净流入股票数: {len(inflow)}")
                print(f"主力净流出股票数: {len(outflow)}")

                if len(inflow) > 0:
                    print("\n主力净流入前10:")
                    top_inflow = inflow.nlargest(10, "主力流入净额")
                    for idx, row in top_inflow.iterrows():
                        name = row.get("股票名称", "未知")
                        code = row.get("股票代码", "未知")
                        amount = row.get("主力流入净额", 0)
                        page_info = f" (第{row.get('页码', '?')}页)" if "页码" in row else ""
                        if pd.notna(amount):
                            print(f"  {code} {name}: {amount:,.0f}{page_info}")
                        else:
                            print(f"  {code} {name}: 数据无效{page_info}")
    elif df_to_process is not None and df_to_process.empty:
        print("\n! 获取到空数据")
    else:
        print("\n! 数据获取失败")

    print("\n" + "=" * 50)
    print("Capital inflow analysis.py程序执行完成！")

    # 在Capital inflow analysis.py运行结束后立即运行Financial ranking.py
    print("\n" + "=" * 50)
    print("准备运行Financial ranking.py...")
    print("=" * 50)

    # 运行Financial ranking.py
    success = run_financial_ranking()

    if success:
        print("\n✓ 两个程序都运行完成！")
    else:
        print("\n! Financial ranking.py运行失败")


if __name__ == "__main__":
    # 检查依赖库
    print("检查依赖库...")
    try:
        import brotli

        print("✓ brotli 已安装")
    except ImportError:
        print("! brotli 未安装，将无法解压br格式")
        print("  请运行: pip install brotli")

    try:
        import pandas

        print("✓ pandas 已安装")
    except ImportError:
        print("! pandas 未安装")
        print("  请运行: pip install pandas")

    try:
        import requests

        print("✓ requests 已安装")
    except ImportError:
        print("! requests 未安装")
        print("  请运行: pip install requests")

    print("=" * 50)

    # 运行主程序
    main()