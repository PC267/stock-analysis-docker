# stock_analysis.py 修改版
import pandas as pd
import os
from datetime import datetime
import glob
import sys

# ========== 修改导入路径 ==========
# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加report_chart目录到路径
report_chart_dir = os.path.join(current_dir, "report_chart")
sys.path.insert(0, report_chart_dir)
import report_config

try:
    from config_loader import ConfigManager, ParameterConfigLoader
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False
    print("⚠️ 未找到config_loader模块，将使用默认配置")

# stock_analysis.py 修改版
import pandas as pd
import os
from datetime import datetime
import glob

# 导入配置加载器
try:
    from config_loader import ConfigManager, ParameterConfigLoader

    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False
    print("⚠️ 未找到config_loader模块，将使用默认配置")


def analyze_stock_reports(input_file=None, output_dir=None, config_file=None):
    """分析股票推荐报告的主函数"""

    # 加载配置
    config = None
    if HAS_CONFIG_LOADER and config_file:
        try:
            # 使用ParameterConfigLoader加载参数配置文件
            loader = ParameterConfigLoader(config_file)
            if loader.load_config_file():
                config = loader
                print(f"✓ 已加载参数配置文件: {config_file}")
        except Exception as e:
            print(f"❌ 加载参数配置文件失败: {e}")
            config = None

    # 如果没有指定输入文件，自动查找最新的风险等级汇总文件
    if input_file is None:
        # 使用硬编码目录
        default_dir = report_config.REPORT_DIR

        if os.path.exists(default_dir):
            print(f"📂 正在扫描目录: {default_dir}")

            # 查看目录下所有文件
            all_files = os.listdir(default_dir)
            print(f"找到 {len(all_files)} 个文件:")
            for f in all_files:
                print(f"  - {f}")

            # 方法1：先找精确匹配
            summary_files = glob.glob(os.path.join(default_dir, "*风险等级汇总*.xlsx"))
            print(f"精确匹配找到 {len(summary_files)} 个文件")

            # 方法2：如果没找到，尝试更宽泛的匹配
            if not summary_files:
                summary_files = glob.glob(os.path.join(default_dir, "*汇总*.xls*"))
                print(f"宽泛匹配找到 {len(summary_files)} 个文件")

            # 方法3：如果还是没找到，列出所有Excel文件
            if not summary_files:
                summary_files = glob.glob(os.path.join(default_dir, "*.xls*"))
                print(f"所有Excel文件: {len(summary_files)} 个")

            if summary_files:
                summary_files.sort(key=os.path.getmtime, reverse=True)
                input_file = summary_files[0]
                print(f"✅ 自动选择: {os.path.basename(input_file)}")
                print(f"   完整路径: {input_file}")
            else:
                print("❌ 未找到任何Excel文件")
                return None

    # 如果没有指定输出目录，使用D:\Computer\桌面\分析报告
    if output_dir is None:
        output_dir = report_config.REPORT_DIR
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_file):
        df = pd.read_excel(input_file)

        # 转换胜率为数值
        df['胜率_数值'] = pd.to_numeric(df['胜率'].astype(str).str.rstrip('%'), errors='coerce') / 100

        # 从配置获取排序规则
        sort_columns = ['风险等级', '夏普比率', '胜率_数值', '最大回撤', '波动率', '风险评分']
        sort_ascending = [True, False, False, True, True, True]

        # 尝试从配置获取排序规则
        if config:
            try:
                # 从通用参数获取排序权重建议
                sort_weight_rec = config.get_general_parameter("sort_weight")
                if sort_weight_rec:
                    # 这里可以解析配置中的排序权重
                    pass
            except:
                pass

        # 按配置的规则排序
        df_sorted = df.sort_values(sort_columns, ascending=sort_ascending)

        # 从配置获取推荐数量
        top_n = 5
        if config:
            try:
                # 尝试从配置获取推荐数量
                rec_count_rec = config.get_general_parameter("recommendation_count")
                if rec_count_rec:
                    top_n = int(rec_count_rec.recommended_value)
            except:
                pass

        top_n = min(top_n, len(df_sorted))
        top_stocks = df_sorted.head(top_n)

        # 生成时间戳 - 从配置获取格式或使用默认
        timestamp_format = "%Y%m%d_%H%M%S"
        if HAS_CONFIG_LOADER:
            try:
                manager = ConfigManager()
                full_config = manager.load_config()
                if full_config and hasattr(full_config.reporting, 'timestamp_format'):
                    timestamp_format = full_config.reporting.timestamp_format
            except:
                pass

        timestamp = datetime.now().strftime(timestamp_format)

        # 1. 保存为Excel文件
        excel_filename = f"股票推荐结果_{timestamp}.xlsx"
        excel_dir = report_config.REPORT_DIR  # 目录路径

        # 确保目录存在
        os.makedirs(excel_dir, exist_ok=True)

        # 完整的文件路径
        excel_path = os.path.join(excel_dir, excel_filename)
        top_stocks.to_excel(excel_path, index=False)
        print(f"✅ Excel文件已保存: {excel_path}")

        # 2. 保存为HTML报告
        html_filename = f"股票推荐结果_{timestamp}.html"

        # 完整的HTML文件路径
        html_path = os.path.join(excel_dir, html_filename)

        # 从配置获取报告样式
        report_title = "股票智能推荐报告"
        report_subtitle = "基于多维度风险评估模型"

        if config:
            try:
                # 尝试从配置获取报告标题
                title_rec = config.get_general_parameter("report_title")
                if title_rec:
                    report_title = str(title_rec.recommended_value)

                subtitle_rec = config.get_general_parameter("report_subtitle")
                if subtitle_rec:
                    report_subtitle = str(subtitle_rec.recommended_value)
            except:
                pass

        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .stock-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .stock-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.15); transform: translateY(-2px); transition: all 0.3s; }}
                .stock-rank {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .stock-name {{ font-size: 18px; font-weight: bold; color: #333; }}
                .stock-code {{ color: #666; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; }}
                .metric {{ background: #f8f9fa; padding: 8px 12px; border-radius: 6px; }}
                .risk-low {{ color: #10b981; font-weight: bold; }}
                .risk-medium {{ color: #f59e0b; font-weight: bold; }}
                .risk-high {{ color: #ef4444; font-weight: bold; }}
                .timestamp {{ color: #6b7280; font-size: 14px; margin-top: 20px; text-align: center; }}
                .summary {{ background: #f3f4f6; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 {report_title}</h1>
                <p>{report_subtitle}</p>
            </div>

            <div class="summary">
                <h3>📊 报告摘要</h3>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>总分析股票数: {len(df)} 只</p>
                <p>推荐数量: {top_n} 只</p>
            </div>

            <h2>🏆 推荐股票列表</h2>
        """

        # 添加股票卡片
        for i in range(top_n):
            stock = df_sorted.iloc[i]
            risk_class = ""
            if "低" in str(stock['风险等级']):
                risk_class = "risk-low"
            elif "中" in str(stock['风险等级']):
                risk_class = "risk-medium"
            else:
                risk_class = "risk-high"

            html_content += f"""
            <div class="stock-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span class="stock-rank">#{i + 1}</span>
                        <span class="stock-name">{stock['股票名称']}</span>
                        <span class="stock-code">({stock['股票代码']})</span>
                    </div>
                    <span class="{risk_class}">风险等级: {stock['风险等级']}</span>
                </div>

                <div class="metrics">
                    <div class="metric">🎯 胜率: {stock['胜率']}</div>
                    <div class="metric">📊 夏普比率: {stock['夏普比率']:.3f}</div>
                    <div class="metric">📈 波动率: {stock['波动率']:.3f}</div>
                    <div class="metric">📉 最大回撤: {stock['最大回撤']:.3f}</div>
                    <div class="metric">⚠️ 风险评分: {stock['风险评分']:.1f}</div>
                </div>
            </div>
            """

        # 从配置获取筛选规则描述
        sort_rules = [
            "按风险等级从低到高排序",
            "同风险等级下按夏普比率从高到低排序",
            "同夏普比率下按胜率从高到低排序",
            "同胜率下按最大回撤从小到大排序",
            "同回撤下按波动率从小到大排序",
            "最后按风险评分从小到大排序"
        ]

        if config:
            try:
                # 可以扩展从配置读取排序规则描述
                pass
            except:
                pass

        # HTML尾部
        html_content += f"""
            <div class="summary">
                <h3>🔍 筛选规则</h3>
                <ol>
        """

        for rule in sort_rules:
            html_content += f"                    <li>{rule}</li>\n"

        html_content += f"""
                </ol>
            </div>

            <div class="timestamp">
                <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>数据来源: {os.path.basename(input_file)} | 生成工具: 股票智能分析系统</p>
        """

        # 如果使用了配置，显示配置信息
        if config_file:
            html_content += f"                <p>参数配置: {os.path.basename(config_file)}</p>\n"

        html_content += """
            </div>
        </body>
        </html>
        """

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 控制台输出
        print("=" * 70)
        print(" " * 20 + f"📈 {report_title}")
        print("=" * 70)
        print(f"📁 数据源: {input_file}")
        if config_file:
            print(f"⚙️  参数配置: {config_file}")
        print(f"📊 总股票数: {len(df)}")
        print(f"🎯 推荐数量: {top_n}")
        print("=" * 70)

        print("\n🏆 推荐股票列表:")
        for i in range(min(top_n, len(df_sorted))):
            stock = df_sorted.iloc[i]
            print(f"\n{i + 1}. {stock['股票名称']}({stock['股票代码']})")
            print(f"   风险等级: {stock['风险等级']}")
            print(f"   胜率: {stock['胜率']}")
            print(f"   夏普比率: {stock['夏普比率']:.3f}")
            print(f"   波动率: {stock['波动率']:.3f}")
            print(f"   最大回撤: {stock['最大回撤']:.3f}")

        print("\n" + "=" * 70)
        print("💾 报告已保存到:")
        print(f"  1. Excel文件: {excel_path}")
        print(f"  2. HTML报告: {html_path}")
        print("=" * 70)

        return {
            'excel_path': excel_path,
            'html_path': html_path,
            'top_stocks': top_stocks,
            'total_stocks': len(df),
            'input_file': input_file,
            'config_file': config_file
        }
    else:
        print(f"❌ 文件不存在: {input_file}")
        print("请检查文件路径是否正确")
        return None


if __name__ == "__main__":
    # 作为独立脚本运行时
    # 可以指定参数配置文件
    config_file_path = None

    # 尝试自动查找参数配置文件
    if HAS_CONFIG_LOADER:
        # 在当前目录或上级目录查找参数配置文件
        possible_paths = [
            "parameter_recommendations.txt",
            "config/parameter_recommendations.txt",
            "../parameter_recommendations.txt",
            "backtest_results/parameter_recommendations/parameter_recommendations.txt"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_file_path = path
                print(f"📂 发现参数配置文件: {path}")
                break

    analyze_stock_reports(config_file=config_file_path)  # stock_analysis.py 修改版
try:
    from config_loader import ConfigManager, ParameterConfigLoader
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False
    print("⚠️ 未找到config_loader模块，将使用默认配置")

# stock_analysis.py 修改版

# 导入配置加载器
try:
    from config_loader import ConfigManager, ParameterConfigLoader

    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False
    print("⚠️ 未找到config_loader模块，将使用默认配置")


def analyze_stock_reports(input_file=None, output_dir=None, config_file=None):
    """分析股票推荐报告的主函数"""

    # 加载配置
    config = None
    if HAS_CONFIG_LOADER and config_file:
        try:
            # 使用ParameterConfigLoader加载参数配置文件
            loader = ParameterConfigLoader(config_file)
            if loader.load_config_file():
                config = loader
                print(f"✓ 已加载参数配置文件: {config_file}")
        except Exception as e:
            print(f"❌ 加载参数配置文件失败: {e}")
            config = None

    # 如果没有指定输入文件，自动查找最新的风险等级汇总文件
    if input_file is None:
        # 使用硬编码目录
        default_dir = report_config.REPORT_DIR

        if os.path.exists(default_dir):
            print(f"📂 正在扫描目录: {default_dir}")

            # 查看目录下所有文件
            all_files = os.listdir(default_dir)
            print(f"找到 {len(all_files)} 个文件:")
            for f in all_files:
                print(f"  - {f}")

            # 方法1：先找精确匹配
            summary_files = glob.glob(os.path.join(default_dir, "*风险等级汇总*.xlsx"))
            print(f"精确匹配找到 {len(summary_files)} 个文件")

            # 方法2：如果没找到，尝试更宽泛的匹配
            if not summary_files:
                summary_files = glob.glob(os.path.join(default_dir, "*汇总*.xls*"))
                print(f"宽泛匹配找到 {len(summary_files)} 个文件")

            # 方法3：如果还是没找到，列出所有Excel文件
            if not summary_files:
                summary_files = glob.glob(os.path.join(default_dir, "*.xls*"))
                print(f"所有Excel文件: {len(summary_files)} 个")

            if summary_files:
                summary_files.sort(key=os.path.getmtime, reverse=True)
                input_file = summary_files[0]
                print(f"✅ 自动选择: {os.path.basename(input_file)}")
                print(f"   完整路径: {input_file}")
            else:
                print("❌ 未找到任何Excel文件")
                return None

    # 如果没有指定输出目录，使用D:\Computer\桌面\分析报告
    if output_dir is None:
        output_dir = report_config.REPORT_DIR
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(input_file):
        df = pd.read_excel(input_file)

        # 转换胜率为数值
        df['胜率_数值'] = pd.to_numeric(df['胜率'].astype(str).str.rstrip('%'), errors='coerce') / 100

        # 从配置获取排序规则
        sort_columns = ['风险等级', '夏普比率', '胜率_数值', '最大回撤', '波动率', '风险评分']
        sort_ascending = [True, False, False, True, True, True]

        # 尝试从配置获取排序规则
        if config:
            try:
                # 从通用参数获取排序权重建议
                sort_weight_rec = config.get_general_parameter("sort_weight")
                if sort_weight_rec:
                    # 这里可以解析配置中的排序权重
                    pass
            except:
                pass

        # 按配置的规则排序
        df_sorted = df.sort_values(sort_columns, ascending=sort_ascending)

        # 从配置获取推荐数量
        top_n = 5
        if config:
            try:
                # 尝试从配置获取推荐数量
                rec_count_rec = config.get_general_parameter("recommendation_count")
                if rec_count_rec:
                    top_n = int(rec_count_rec.recommended_value)
            except:
                pass

        top_n = min(top_n, len(df_sorted))
        top_stocks = df_sorted.head(top_n)

        # 生成时间戳 - 从配置获取格式或使用默认
        timestamp_format = "%Y%m%d_%H%M%S"
        if HAS_CONFIG_LOADER:
            try:
                manager = ConfigManager()
                full_config = manager.load_config()
                if full_config and hasattr(full_config.reporting, 'timestamp_format'):
                    timestamp_format = full_config.reporting.timestamp_format
            except:
                pass

        timestamp = datetime.now().strftime(timestamp_format)

        # 1. 保存为Excel文件
        excel_filename = f"股票推荐结果_{timestamp}.xlsx"
        excel_dir = report_config.REPORT_DIR  # 目录路径

        # 确保目录存在
        os.makedirs(excel_dir, exist_ok=True)

        # 完整的文件路径
        excel_path = os.path.join(excel_dir, excel_filename)
        top_stocks.to_excel(excel_path, index=False)
        print(f"✅ Excel文件已保存: {excel_path}")

        # 2. 保存为HTML报告
        html_filename = f"股票推荐结果_{timestamp}.html"

        # 完整的HTML文件路径
        html_path = os.path.join(excel_dir, html_filename)

        # 从配置获取报告样式
        report_title = "股票智能推荐报告"
        report_subtitle = "基于多维度风险评估模型"

        if config:
            try:
                # 尝试从配置获取报告标题
                title_rec = config.get_general_parameter("report_title")
                if title_rec:
                    report_title = str(title_rec.recommended_value)

                subtitle_rec = config.get_general_parameter("report_subtitle")
                if subtitle_rec:
                    report_subtitle = str(subtitle_rec.recommended_value)
            except:
                pass

        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_title}</title>
            <style>
                body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .stock-card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
                .stock-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.15); transform: translateY(-2px); transition: all 0.3s; }}
                .stock-rank {{ font-size: 24px; font-weight: bold; color: #667eea; }}
                .stock-name {{ font-size: 18px; font-weight: bold; color: #333; }}
                .stock-code {{ color: #666; }}
                .metrics {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 10px; }}
                .metric {{ background: #f8f9fa; padding: 8px 12px; border-radius: 6px; }}
                .risk-low {{ color: #10b981; font-weight: bold; }}
                .risk-medium {{ color: #f59e0b; font-weight: bold; }}
                .risk-high {{ color: #ef4444; font-weight: bold; }}
                .timestamp {{ color: #6b7280; font-size: 14px; margin-top: 20px; text-align: center; }}
                .summary {{ background: #f3f4f6; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 {report_title}</h1>
                <p>{report_subtitle}</p>
            </div>

            <div class="summary">
                <h3>📊 报告摘要</h3>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>总分析股票数: {len(df)} 只</p>
                <p>推荐数量: {top_n} 只</p>
            </div>

            <h2>🏆 推荐股票列表</h2>
        """

        # 添加股票卡片
        for i in range(top_n):
            stock = df_sorted.iloc[i]
            risk_class = ""
            if "低" in str(stock['风险等级']):
                risk_class = "risk-low"
            elif "中" in str(stock['风险等级']):
                risk_class = "risk-medium"
            else:
                risk_class = "risk-high"

            html_content += f"""
            <div class="stock-card">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div>
                        <span class="stock-rank">#{i + 1}</span>
                        <span class="stock-name">{stock['股票名称']}</span>
                        <span class="stock-code">({stock['股票代码']})</span>
                    </div>
                    <span class="{risk_class}">风险等级: {stock['风险等级']}</span>
                </div>

                <div class="metrics">
                    <div class="metric">🎯 胜率: {stock['胜率']}</div>
                    <div class="metric">📊 夏普比率: {stock['夏普比率']:.3f}</div>
                    <div class="metric">📈 波动率: {stock['波动率']:.3f}</div>
                    <div class="metric">📉 最大回撤: {stock['最大回撤']:.3f}</div>
                    <div class="metric">⚠️ 风险评分: {stock['风险评分']:.1f}</div>
                </div>
            </div>
            """

        # 从配置获取筛选规则描述
        sort_rules = [
            "按风险等级从低到高排序",
            "同风险等级下按夏普比率从高到低排序",
            "同夏普比率下按胜率从高到低排序",
            "同胜率下按最大回撤从小到大排序",
            "同回撤下按波动率从小到大排序",
            "最后按风险评分从小到大排序"
        ]

        if config:
            try:
                # 可以扩展从配置读取排序规则描述
                pass
            except:
                pass

        # HTML尾部
        html_content += f"""
            <div class="summary">
                <h3>🔍 筛选规则</h3>
                <ol>
        """

        for rule in sort_rules:
            html_content += f"                    <li>{rule}</li>\n"

        html_content += f"""
                </ol>
            </div>

            <div class="timestamp">
                <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>数据来源: {os.path.basename(input_file)} | 生成工具: 股票智能分析系统</p>
        """

        # 如果使用了配置，显示配置信息
        if config_file:
            html_content += f"                <p>参数配置: {os.path.basename(config_file)}</p>\n"

        html_content += """
            </div>
        </body>
        </html>
        """

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # 控制台输出
        print("=" * 48)
        print(" " * 20 + f"📈 {report_title}")
        print("=" * 48)
        print(f"📁 数据源: {input_file}")
        if config_file:
            print(f"⚙️  参数配置: {config_file}")
        print(f"📊 总股票数: {len(df)}")
        print(f"🎯 推荐数量: {top_n}")
        print("=" * 48)

        print("\n🏆 推荐股票列表:")
        for i in range(min(top_n, len(df_sorted))):
            stock = df_sorted.iloc[i]
            print(f"\n{i + 1}. {stock['股票名称']}({stock['股票代码']})")
            print(f"   风险等级: {stock['风险等级']}")
            print(f"   胜率: {stock['胜率']}")
            print(f"   夏普比率: {stock['夏普比率']:.3f}")
            print(f"   波动率: {stock['波动率']:.3f}")
            print(f"   最大回撤: {stock['最大回撤']:.3f}")

        print("\n" + "=" * 48)
        print("💾 报告已保存到:")
        print(f"  1. Excel文件: {excel_path}")
        print(f"  2. HTML报告: {html_path}")
        print("=" * 48)

        return {
            'excel_path': excel_path,
            'html_path': html_path,
            'top_stocks': top_stocks,
            'total_stocks': len(df),
            'input_file': input_file,
            'config_file': config_file
        }
    else:
        print(f"❌ 文件不存在: {input_file}")
        print("请检查文件路径是否正确")
        return None


if __name__ == "__main__":
    # 作为独立脚本运行时
    # 可以指定参数配置文件
    config_file_path = None

    # 尝试自动查找参数配置文件
    if HAS_CONFIG_LOADER:
        # 在当前目录或上级目录查找参数配置文件
        possible_paths = [
            "parameter_recommendations.txt",
            "config/parameter_recommendations.txt",
            "../parameter_recommendations.txt",
            "backtest_results/parameter_recommendations/parameter_recommendations.txt"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                config_file_path = path
                print(f"📂 发现参数配置文件: {path}")
                break

    analyze_stock_reports(config_file=config_file_path)
    analyze_stock_reports(config_file=config_file_path)