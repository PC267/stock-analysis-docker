# main.py 开头部分
import warnings
import matplotlib
import numpy as np
from datetime import datetime
import atexit
import sys
import os
import pytz
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 保留 WARNING，但禁止 GPU 相关警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ========== 修改导入路径 ==========
# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加report_chart目录到路径
report_chart_dir = os.path.join(current_dir, "report_chart")
sys.path.insert(0, report_chart_dir)

# 现在可以导入report_config了
import report_config
# 设置保存路径 - 使用report_config模块中的配置
SAVE_PATH = report_config.REPORT_DIR


# 设置环境变量防止 Tkinter 初始化
os.environ['PYTHON_DISABLE_TK'] = '1'
os.environ['TK_SILENCE_DEPRECATION'] = '1'
# 获取UTC时间
utc_time = datetime.now(pytz.utc)

# 转换为北京时间（东八区）
beijing_timezone = pytz.timezone('Asia/Shanghai')
beijing_time = utc_time.astimezone(beijing_timezone)

def exit_handler():
    """程序退出时的清理函数"""
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except:
        pass

    # 确保退出
    sys.stdout.flush()
    sys.stderr.flush()


atexit.register(exit_handler)


# ========== 关键修改：使用非交互式后端 ==========
# 设置matplotlib后端为Agg（纯图像，无GUI）
matplotlib.use('Agg')  # 从 TkAgg 改为 Agg
matplotlib.interactive(False)  # 禁用交互模式

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

print("=" * 48)
print(" " * 13 + "📈 股票智能模拟分析系统 v7.0")
print("=" * 48)
print(" " * 15 + "基于baostock真实股票数据")
print("=" * 48)
print("\n🛠️  系统配置:")
print(f"  数据源: baostock (专业免费股票数据接口)")
print(f"  数据范围: 最近10年历史数据")
print(f"  缓存目录: stock_data/")
print(f"  当前时间: {beijing_time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 48)
# 确保当前目录在路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"当前工作目录: {os.getcwd()}")
print(f"脚本所在目录: {current_dir}")

# 检查可能缺失的模块
try:
    import psutil
except ImportError:
    print("警告: psutil 模块未安装，内存监控功能将不可用")
    psutil = None

# 尝试导入config
try:
    # 首先尝试直接导入
    from config_loader import Config
    print("✓ 成功导入 Config")
except ImportError as e:
    print(f"导入 Config 失败: {e}")
    # 尝试相对导入
    try:
        import config_loader
        Config = config_loader.Config  # 修正这里
        print("✓ 通过相对导入成功获取 Config")
    except Exception as e2:
        print(f"相对导入也失败: {e2}")
        # 创建一个简单的Config类作为备用
        class SimpleConfig:
            def __init__(self, stock_code, stock_name, days_to_simulate=20):
                self.stock_code = stock_code
                self.stock_name = stock_name
                self.days_to_simulate = days_to_simulate
                self.data_dir = "./stock_data"
                os.makedirs(self.data_dir, exist_ok=True)
        Config = SimpleConfig
        print("⚠️ 使用简化的 Config 类")

# 尝试导入其他模块

try:
    from data.data_fetcher import DataFetcher
    print("✓ 成功导入 DataFetcher")
except ImportError as e:
    print(f"导入 DataFetcher 失败: {e}")
    exit(1)

try:
    from data.data_processor import DataProcessor
    print("✓ 成功导入 DataProcessor")
except ImportError as e:
    print(f"导入 DataProcessor 失败: {e}")
    exit(1)

try:
    from model.evaluator import ModelEvaluator
    print("✓ 成功导入 ModelEvaluator")
except ImportError as e:
    print(f"导入 ModelEvaluator 失败: {e}")
    exit(1)


try:
    from model.factory import ModelFactory
    print("✓ 成功导入 ModelFactory")
except ImportError as e:
    print(f"导入 ModelFactory 失败: {e}")
    exit(1)

try:
    from model.simulator import StockSimulator
    print("✓ 成功导入 StockSimulator")
except ImportError as e:
    print(f"导入 StockSimulator 失败: {e}")
    exit(1)

try:
    from report_chart.chart_generator import ChartGenerator
    print("✓ 成功导入 ChartGenerator")
except ImportError as e:
    print(f"导入 ChartGenerator 失败: {e}")
    exit(1)

# 修改报告生成器的导入方式
try:
    # 尝试导入报告模版
    from report_chart.report_generator import ReportGenerator
    print("✓ 成功导入 ReportGenerator")
except ImportError as e2:
    print(f"导入 ReportGenerator 也失败: {e2}")
    try:
        # 尝试导入HTML版本
        from html_report_generator import ReportGenerator
        print("✓ 成功导入 HTML 版本的 ReportGenerator")
    except ImportError as e3:
        print(f"导入所有报告生成器均失败: {e3}")
        ReportGenerator = None
        print("⚠️ 报告生成功能将不可用")

print("\n" + "=" * 48)


def get_best_model_fallback(models, simulations):
    """备选方案：如果模型评估失败，使用简化的选择方法"""
    print("使用备选方案选择最优模型...")

    try:
        # 如果有模拟结果，选择第一个非空的
        if simulations and len(simulations) > 0:
            for model_name, sim_data in simulations.items():
                if sim_data is not None and len(sim_data) > 0:
                    print(f"备选选择模型: {model_name}")
                    return model_name, sim_data

        # 如果有模型，选择第一个
        if models and len(models) > 0:
            model_names = list(models.keys())
            print(f"备选选择模型: {model_names[0]}")
            return model_names[0], None

    except Exception as e:
        print(f"备选方案也失败: {e}")

    # 最后备选
    print("使用默认模型: arima")
    return "arima", None


# ========== 添加内存管理函数 ==========
def cleanup_resources():
    """清理内存和资源"""
    try:
        # 清理 matplotlib
        import matplotlib.pyplot as plt
        plt.close('all')
        print("✓ 清理 matplotlib 图形")
    except:
        pass

    try:
        # 强制垃圾回收
        import gc
        gc.collect()
        print("✓ 触发垃圾回收")
    except:
        pass


def analyze_single_stock(stock_code, stock_name, days_to_simulate, output_format='html', auto_open_html=True):
    """分析单只股票"""
    try:
        # 记录内存使用
        start_memory = 0
        process = None
        if psutil:
            import os as os_module
            process = psutil.Process(os_module.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"开始内存: {start_memory:.1f} MB")

        # 创建配置
        print("\n正在初始化配置...")
        config = Config(stock_code, stock_name, days_to_simulate)
        print(f"✓ 配置初始化完成")

        # 1. 获取数据
        print("\n1️⃣ 数据获取阶段:")
        print("-" * 40)
        try:
            data_fetcher = DataFetcher(config)
            stock_data = data_fetcher.fetch_stock_data()

            if stock_data is None or len(stock_data) == 0:
                print("❌ 数据获取失败，跳过此股票")
                return None

            print(f"✓ 数据获取成功，共 {len(stock_data)} 条记录")
        except Exception as e:
            print(f"❌ 数据获取阶段出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 2. 处理数据
        print("\n2️⃣ 数据处理阶段:")
        print("-" * 48)
        try:
            data_processor = DataProcessor()
            processed_data = data_processor.process_data(stock_data)
            print(f"✓ 数据处理完成")
        except Exception as e:
            print(f"❌ 数据处理阶段出错: {e}")
            return None
        # 3. 训练模型
        print("\n3️⃣ 模型训练阶段:")
        print("-" * 48)
        try:
            if processed_data is None:
                print("❌ 处理后的数据为None，无法训练模型")
                return None

            # 训练模型
            model_factory = ModelFactory(config)
            models = model_factory.train_all_models(processed_data)

            # 获取推荐的模型组合（基于评估分数）
            try:
                # 训练模型
                model_factory = ModelFactory(config)
                models = model_factory.train_all_models(processed_data)

                # 获取推荐的模型组合（基于市场状态）
                try:
                    # 使用 market_regime 获取推荐模型和权重
                    market_regime = getattr(model_factory, 'market_regime', 'default')
                    model_weight_config = model_factory._get_model_weight_config()

                    # 根据当前市场状态获取推荐模型
                    if market_regime in model_weight_config:
                        recommended_models = model_weight_config[market_regime]
                    else:
                        recommended_models = model_weight_config.get('default', [])

                    # 输出结果
                    print("✅ 推荐模型组合 (基于市场状态):")
                    print(f"  市场状态: {market_regime}")

                    if recommended_models:
                        for model_name, weight in recommended_models:
                            print(f"  {model_name:<15}: {weight * 100:.1f}%")
                    else:
                        # 如果无法获取推荐模型，使用默认组合
                        recommended_models = [('arima', 0.4), ('xgboost', 0.3), ('garch', 0.3)]
                        print(f"  ⚠️ 未找到针对'{market_regise}'的配置，使用默认组合")
                        for model_name, weight in recommended_models:
                            print(f"  {model_name:<15}: {weight * 100:.1f}%")
                except Exception as e:
                    print(f"⚠️ 获取推荐模型组合失败: {e}")
                    # 使用默认模型组合
                    recommended_models = [('arima', 0.4), ('xgboost', 0.3), ('garch', 0.3)]
                    print("✅ 默认模型组合:")
                    for model_name, weight in recommended_models:
                        print(f"  {model_name:<15}: {weight * 100:.1f}%")

                if models is None or len(models) == 0:
                    print("❌ 模型训练失败，没有可用的模型")
                    return None

                print(f"✓ 模型训练完成")
            except Exception as e:
                print(f"❌ 模型训练阶段出错: {e}")
                import traceback
                traceback.print_exc()
                return None
            
            # 输出结果
            print("✅ 推荐模型组合:")
            for model_name, weight in recommended_models:
                # 获取模型评分
                scores = model_factory.evaluate_models()
                score = scores.get(model_name, 0)
                print(f"  {model_name:<15}: {weight*100:.1f}% (评分: {score:.2f})")

            if models is None or len(models) == 0:
                print("❌ 模型训练失败，没有可用的模型")
                return None

            print(f"✓ 模型训练完成")
        except Exception as e:
            print(f"❌ 模型训练阶段出错: {e}")
            import traceback
            traceback.print_exc()
            return None

        # 4. 模拟分析
        print("\n4️⃣ 模拟分析阶段:")
        print("-" * 48)
        try:
            simulator = StockSimulator(config, models, processed_data)
            simulations, future_dates = simulator.simulate_future()
            print(f"✓ 模拟分析完成")
        except Exception as e:
            print(f"❌ 模拟分析阶段出错: {e}")
            return None

        # 5. 模型评估 - 修复部分开始
        print("\n5️⃣ 模型评估阶段:")
        print("-" * 48)
        best_model_name = None
        best_simulations = None

        try:
            evaluator = ModelEvaluator(config, models, simulations, processed_data)

            # 直接访问 select_best_model 属性获取最优模型
            try:
                best_model_info = evaluator.select_best_model

                if best_model_info is None:
                    print("⚠️ 模型评估返回 None，使用备选方案")
                    best_model_name, best_simulations = get_best_model_fallback(models, simulations)
                else:
                    # 解析返回结果 - 根据 model_evaluator.py，返回的是 (model_name, simulations_array)
                    if isinstance(best_model_info, tuple) and len(best_model_info) >= 2:
                        best_model_name, best_simulations = best_model_info[0], best_model_info[1]
                        print(f"✓ 模型评估完成，最优模型: {best_model_name}")
                    else:
                        print(f"⚠️ 返回格式异常: {type(best_model_info)}，使用备选方案")
                        best_model_name, best_simulations = get_best_model_fallback(models, simulations)

            except Exception as e:
                print(f"❌ 模型评估出错: {e}")
                print("使用备选方案继续...")
                best_model_name, best_simulations = get_best_model_fallback(models, simulations)

        except Exception as e:
            print(f"❌ 模型评估阶段初始化出错: {e}")
            print("使用备选方案继续...")
            best_model_name, best_simulations = get_best_model_fallback(models, simulations)

        # 确保有有效的模型名称
        if best_model_name is None:
            print("⚠️ 无法确定最优模型，使用备选方案")
            best_model_name, best_simulations = get_best_model_fallback(models, simulations)
        # 模型评估 - 修复部分结束

        # 6. 生成模拟K线
        print("\n6️⃣ 生成模拟数据:")
        print("-" * 48)
        try:
            # 确保best_simulations是numpy数组
            if isinstance(best_simulations, dict):
                print(f"警告: best_simulations 是字典类型，转换为数组")
                last_price = processed_data['processed']['close'].iloc[-1]
                best_simulations = np.full(config.days_to_simulate, last_price)

            simulated_data = simulator.generate_candlestick_data(best_model_name, future_dates)
            print(f"✓ 模拟数据生成完成")
        except Exception as e:
            print(f"❌ 模拟数据生成阶段出错: {e}")
            return None

        # 7. 可视化
        print("\n7️⃣ 可视化展示:")
        print("-" * 48)
        try:
            chart_generator = ChartGenerator(config)
            chart_generator.generate_all_charts(processed_data, simulated_data, best_model_name)
            print(f"✓ 可视化完成")
        except Exception as e:
            print(f"❌ 可视化阶段出错: {e}")

        # 8. 生成报告 - 使用新的报告生成器
        print("\n8️⃣ 生成分析报告:")
        print("-" * 48)
        try:
            if ReportGenerator is None:
                print("⚠️ 报告生成器不可用，跳过报告生成")
                return None
            else:
                # 创建报告生成器实例
                report_generator = ReportGenerator(config)

                # 检查是否是统一的报告生成器
                if hasattr(report_generator, '_generate_html_report'):
                    # 这是统一版本，可以指定输出格式
                    print(f"生成 {output_format.upper()} 格式报告...")

                    # 生成报告
                    result = report_generator.generate_report(
                        processed_data=processed_data,
                        simulated_data=simulated_data,
                        models=models,
                        best_model_name=best_model_name,
                        output_format=output_format  # 传递输出格式
                    )

                    if result:
                        if output_format == 'html':
                            print(f"✓ HTML报告生成完成")
                            report_path = result.get('html_report', '')
                        elif output_format == 'visual':
                            print(f"✓ 可视化报告生成完成")
                            report_path = result.get('visual_report', '')
                        elif output_format == 'text':
                            print(f"✓ 文本报告生成完成")
                            report_path = result.get('text_report', '')

                        if report_path and os.path.exists(report_path):
                            print(f"📁 报告保存位置: {report_path}")

                            # 根据设置决定是否打开报告
                            if output_format == 'html' and auto_open_html:
                                import webbrowser
                                try:
                                    webbrowser.open('file://' + os.path.abspath(report_path))
                                    print("✅ 正在浏览器中打开报告...")
                                except:
                                    print("⚠️ 无法在浏览器中打开报告")
                    else:
                        print("❌ 报告生成失败")
                        return None

                else:
                    result = report_generator.generate_report(
                        processed_data,
                        simulated_data,
                        models,
                        best_model_name
                    )

                    if result:
                        report_path = result.get('html_report', '') or result.get('visual_report', '')
                        if report_path:
                            print(f"📁 报告保存位置: {report_path}")
                    else:
                        print("❌ 报告生成失败")
                        return None

            # ========== 关键：分析完成后立即清理 ==========
            print("\n🧹 清理本次分析资源...")

            # 清理大对象
            try:
                del processed_data
                del simulated_data
                del models
                del chart_generator
                del report_generator
            except:
                pass

            # 调用清理函数
            cleanup_resources()

            # 记录内存使用
            if psutil and process:
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = end_memory - start_memory
                print(f"结束内存: {end_memory:.1f} MB, 本次分析占用: {memory_diff:.1f} MB")

            return report_path if 'report_path' in locals() else None

        except Exception as e:
            print(f"❌ 报告生成阶段出错: {e}")
            import traceback
            traceback.print_exc()
            # 即使出错也要清理
            cleanup_resources()
            return None

        print(f"\n{'=' * 48}")
        print(" " * 20 + "🎉 分析完成！")
        print(f"{'=' * 48}")

    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        # 出错时也清理
        cleanup_resources()
        return None


def main():
    """主函数"""
    print(" " * 20 + "📈 股票智能模拟分析系统 v7.0")
    print("=" * 48)
    print(" " * 15 + "基于真实数据 + 机器学习模型")
    print("=" * 48)

    print("\n📋 股票输入方式:")
    print("  1. 单只股票分析")
    print("  2. 批量导入股票")

    choice = input("请选择输入方式 (默认1): ").strip() or "1"

    stock_list = []
    is_batch_mode = False

    if choice == "2":
        # 批量导入模式
        is_batch_mode = True
        print("\n📄 批量导入股票:")
        print("  1. 从文件导入 (每行格式: 股票代码,股票名称)")
        print("  2. 手动输入多只股票")
        print("  3. 资金流入分析程序")

        batch_choice = input("请选择批量导入方式 (默认1): ").strip() or "1"

        if batch_choice == "1":
            # 从文件导入
            file_path = input("请输入文件路径 (默认: stocks.txt): ").strip() or "stocks.txt"
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line or line.startswith('#'):
                                continue
                            parts = line.split(',')
                            if len(parts) >= 2:
                                code = parts[0].strip()
                                name = parts[1].strip()
                                stock_list.append((code, name))
                            elif len(parts) == 1:
                                code = parts[0].strip()
                                name = f"股票{code}"
                                stock_list.append((code, name))
                    print(f"✓ 从文件 '{file_path}' 成功导入 {len(stock_list)} 只股票")
                else:
                    print(f"❌ 文件 '{file_path}' 不存在，切换到单只股票模式")
                    is_batch_mode = False
                    choice = "1"
            except Exception as e:
                print(f"❌ 读取文件失败: {e}，切换到单只股票模式")
                is_batch_mode = False
                choice = "1"

        elif batch_choice == "2":
            # 手动输入多只股票
            print("\n📝 手动输入多只股票 (输入空行结束):")
            print("  格式: 股票代码,股票名称 (例如: 000001,平安银行)")
            print("  或: 股票代码 (自动生成名称)")

            count = 1
            while True:
                stock_input = input(f"股票 {count} (输入空行结束): ").strip()
                if not stock_input:
                    break

                parts = stock_input.split(',')
                if len(parts) >= 2:
                    code = parts[0].strip()
                    name = parts[1].strip()
                    stock_list.append((code, name))
                elif len(parts) == 1:
                    code = parts[0].strip()
                    name = f"股票{code}"
                    stock_list.append((code, name))
                else:
                    print("❌ 格式错误，请重新输入")
                    continue

                count += 1

            if len(stock_list) > 0:
                print(f"✓ 成功输入 {len(stock_list)} 只股票")
            else:
                print("⚠️ 未输入任何股票，切换到单只股票模式")
                is_batch_mode = False
                choice = "1"

        elif batch_choice == "3":
            # 资金流入分析菜单
            print("\n📊 资金流入分析菜单")
            print("=" * 48)
            print("1. 个股资金流入分析")
            print("2. 行业资金流入分析")
            print("=" * 48)
            capital_choice = input("请选择分析类型 (1或2): ").strip()

            analysis_type = ""  # 初始化变量
            if capital_choice == "1":
                # 个股资金流入分析
                print("\n💰 开始个股资金流入分析...")
                print("=" * 48)
                script_name = "Capital inflow analysis.py"
                results_file = "Stock_Data.txt"
                analysis_type = "个股资金流入"
            elif capital_choice == "2":
                # 行业资金流入分析
                print("\n🏢 开始行业资金流入分析...")
                print("=" * 48)
                script_name = "GaiNian stock.py"
                results_file = "Stock_Data.txt"  # 根据您的说明，行业分析也生成Stock_Data.txt文件
                analysis_type = "行业资金流入"
            else:
                print("❌ 无效选择，切换到单只股票模式")
                is_batch_mode = False
                choice = "1"
                # 需要设置默认值，避免后续代码出错
                analysis_path = None
                result = None
                capital_choice = None
                analysis_type = "资金流入"  # 设置默认值
                
            if capital_choice in ["1", "2"]:
                try:
                    import subprocess
                    # 定义脚本路径 - 修复路径问题
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    
                    # 尝试多种可能的路径
                    possible_paths = [
                        os.path.join(current_dir, "model", script_name),  # model目录下
                        os.path.join(current_dir, script_name),  # 当前目录下
                        os.path.join(current_dir, "..", "model", script_name),  # 上级的model目录
                        script_name,  # 当前工作目录
                    ]
                    
                    analysis_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            analysis_path = os.path.abspath(path)
                            print(f"✓ 找到分析程序: {analysis_path}")
                            break
                    
                    if not analysis_path:
                        print(f"❌ {script_name} 程序不存在，尝试在以下位置查找:")
                        for i, path in enumerate(possible_paths, 1):
                            abs_path = os.path.abspath(path)
                            exists = "✓" if os.path.exists(path) else "✗"
                            print(f"  {i}. {exists} {abs_path}")
                        print("切换到单只股票模式")
                        is_batch_mode = False
                        choice = "1"
                    else:
                        print(f"✓ 找到分析程序: {analysis_path}")
                        print("正在运行分析程序...")
                        
                        # 运行分析程序
                        result = subprocess.run(
                            [sys.executable, analysis_path],
                            capture_output=True,
                            text=True,
                            encoding='utf-8',
                            cwd=os.path.dirname(analysis_path)  # 在程序所在目录运行
                        )
                        
                        if result.returncode == 0:
                            print("✓ 分析程序运行成功")
                            print("-" * 48)
                            
                            # 尝试查找结果文件 - 修复路径问题
                            possible_result_files = [
                                os.path.join(os.path.dirname(analysis_path), "results", results_file),
                                os.path.join(current_dir, "results", results_file),
                                os.path.join(current_dir, results_file),
                                os.path.join(os.path.dirname(analysis_path), results_file),
                                f"results/{results_file}",
                                results_file,
                            ]
                            
                            result_file_path = None
                            for file_path in possible_result_files:
                                if os.path.exists(file_path):
                                    result_file_path = os.path.abspath(file_path)
                                    print(f"✓ 找到分析结果文件: {result_file_path}")
                                    break
                            
                            if not result_file_path:
                                print("❌ 未找到分析结果文件，尝试在以下位置查找:")
                                for i, file_path in enumerate(possible_result_files, 1):
                                    abs_path = os.path.abspath(file_path)
                                    exists = "✓" if os.path.exists(file_path) else "✗"
                                    print(f"  {i}. {exists} {abs_path}")
                                print("请检查程序输出，确认文件生成位置")
                                print("切换到单只股票模式")
                                is_batch_mode = False
                                choice = "1"
                            else:
                                # 读取股票数据
                                stock_count = 0
                                try:
                                    with open(result_file_path, 'r', encoding='utf-8') as f:
                                        for line_num, line in enumerate(f, 1):
                                            line = line.strip()
                                            if not line:
                                                continue
                                            parts = line.split(',')
                                            if len(parts) >= 2:
                                                code = parts[0].strip()
                                                name = parts[1].strip()
                                                stock_list.append((code, name))
                                                stock_count += 1
                                            elif len(parts) == 1:
                                                code = parts[0].strip()
                                                name = f"股票{code}"
                                                stock_list.append((code, name))
                                                stock_count += 1
                                    print(f"✓ 从分析结果成功导入 {stock_count} 只股票")
                                    
                                    # 显示导入的股票
                                    if stock_count > 0:
                                        print("\n📋 导入的股票列表:")
                                        for idx, (code, name) in enumerate(stock_list[:10], 1):  # 只显示前10只
                                            print(f"  {idx:2d}. {code},{name}")
                                        if stock_count > 10:
                                            print(f"  ... 还有 {stock_count - 10} 只股票")
                                        
                                        # 显示股票来源信息
                                        print(f"\n📊 股票来源: {analysis_type}分析")
                                        print(f"📁 数据文件: {result_file_path}")
                                        print(f"📈 选择依据: 三日涨跌幅%从小到大 + 各项总和减主力流入净额从大到小")
                                        print(f"🎯 选择数量: 前{stock_count}只股票")
                                except Exception as e:
                                    print(f"❌ 读取分析结果文件失败: {e}")
                                    print("切换到单只股票模式")
                                    is_batch_mode = False
                                    choice = "1"
                                
                                # 显示程序输出
                                print(f"\n📄 {analysis_type}分析程序输出:")
                                print("-" * 48)
                                # 只显示重要的输出信息
                                output_lines = result.stdout.split('\n')
                                important_lines = []
                                for line in output_lines:
                                    if any(keyword in line for keyword in 
                                           ['成功', '保存', '创建', '开始', '完成', '错误', '失败', '警告', '读取', '处理', '计算']):
                                        important_lines.append(line)
                                
                                if important_lines:
                                    print('\n'.join(important_lines[:20]))  # 只显示前20行重要信息
                                else:
                                    print(result.stdout[:500])  # 只显示前500字符
                                if len(result.stdout) > 500:
                                    print(f"... (完整输出共{len(result.stdout)}字符)")
                                print("-" * 48)
                                
                        else:
                            print(f"❌ 分析程序运行失败，返回码: {result.returncode}")
                            print("错误信息:")
                            print(result.stderr[:500])  # 只显示前500字符错误信息
                            if len(result.stderr) > 500:
                                print(f"... (还有{len(result.stderr) - 500}字符)")
                            print("切换到单只股票模式")
                            is_batch_mode = False
                            choice = "1"
                            
                except Exception as e:
                    print(f"❌ 运行分析程序时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    print("切换到单只股票模式")
                    is_batch_mode = False
                    choice = "1"
        else:
            print("❌ 无效选择，使用默认选项1")
            is_batch_mode = False
            choice = "1"

    if choice == "1" or len(stock_list) == 0:
        # 单只股票模式
        is_batch_mode = False
        stock_code = input("股票代码 (例如: 000001): ").strip() or "000001"
        stock_name = input("股票名称 (例如: 平安银行): ").strip() or "平安银行"
        stock_list = [(stock_code, stock_name)]

    try:
        days_input = input("模拟天数 (默认30): ").strip()
        days_to_simulate = int(days_input) if days_input else 30
    except:
        days_to_simulate = 30

    # 设置批量处理参数
    if is_batch_mode:
        # 批量模式：统一设置参数
        print("\n📊 批量处理配置:")
        print("  可选报告格式: html (网页), text (文本), visual (图片)")
        output_format = input("报告格式 (默认html): ").strip().lower() or "html"

        auto_open_html = False  # 批量模式默认不自动打开
        if output_format == 'html':
            open_choice = input("分析完成后自动在浏览器中打开所有报告？(y/N): ").strip().lower()
            auto_open_html = open_choice in ['y', 'yes']

        # ========== 关键：如果股票数量多，询问是否分批 ==========
        if len(stock_list) > 5:
            print(f"\n⚠️  检测到要分析 {len(stock_list)} 只股票，数量较多")
            print("建议分批处理，每批5只，防止内存泄漏")

            batch_choice = input("是否分批处理？(Y/n): ").strip().lower()
            if batch_choice not in ['n', 'no']:
                # 分批处理
                batch_size = 5
                batches = [stock_list[i:i + batch_size]
                           for i in range(0, len(stock_list), batch_size)]

                report_paths = []

                for batch_num, batch in enumerate(batches, 1):
                    print(f"\n{'=' * 48}")
                    print(f"处理第 {batch_num}/{len(batches)} 批 ({len(batch)} 只股票)...")
                    print(f"{'=' * 48}")

                    for idx, (stock_code, stock_name) in enumerate(batch, 1):
                        print(f"\n📊 分析第 {idx}/{len(batch)} 只股票: {stock_name}({stock_code})")

                        try:
                            report_path = analyze_single_stock(
                                stock_code,
                                stock_name,
                                days_to_simulate,
                                output_format,
                                False  # 批量模式不自动打开
                            )
                            if report_path:
                                report_paths.append(report_path)
                                print(f"✓ 第 {idx} 只股票分析完成")
                            else:
                                print(f"⚠️ 第 {idx} 只股票分析失败")
                        except Exception as e:
                            print(f"❌ 第 {idx} 只股票分析失败: {e}")
                            continue

                    # 每批结束后清理
                    if batch_num < len(batches):
                        print(f"\n🧹 批次 {batch_num} 完成，清理资源...")
                        cleanup_resources()

                # 所有批次完成
                print(f"\n{'=' * 48}")
                print(" " * 20 + "🎉 批量分析完成！")
                print(f"{'=' * 48}")

                if report_paths:
                    print(f"\n📋 分析结果汇总:")
                    print(f"  成功分析: {len(report_paths)}/{len(stock_list)} 只股票")
                    print(f"  报告保存位置:")
                    for path in report_paths[:10]:  # 只显示前10个
                        print(f"    - {os.path.basename(path)}")
                    if len(report_paths) > 10:
                        print(f"    ... 还有 {len(report_paths) - 10} 个报告")

                # 批量模式：询问是否继续分析其他股票
                print("\n批量分析已完成！")
                choice = input("\n是否继续分析其他股票？(y/N): ").strip().lower()
                if choice in ['y', 'yes']:
                    main()
                else:
                    # ========== 新增：执行股票分析脚本 ==========
                    print("\n📊 正在生成股票推荐分析报告...")
                    try:
                        # 尝试导入stock_analysis模块
                        try:
                            import stock_analysis
                            stock_analysis_available = True
                        except ImportError:
                            print("⚠️ stock_analysis 模块未安装，跳过股票推荐报告生成")
                            stock_analysis_available = False

                        if stock_analysis_available:
                            print("=" * 48)
                            print(" " * 20 + "📈 开始生成股票推荐报告")
                            print("=" * 48)

                            # 指定具体的文件路径
                            summary_file = os.path.join(SAVE_PATH, "风险等级汇总.xlsx")
                            result = stock_analysis.analyze_stock_reports(input_file=summary_file)

                            if result:
                                print("✅ 股票推荐报告生成成功！")
                                print(f"📊 分析完成: {result['total_stocks']} 只股票")
                                print(f"📁 Excel报告: {result['excel_path']}")
                                print(f"🌐 HTML报告: {result['html_path']}")
                            else:
                                print("❌ 股票推荐报告生成失败")
                    except Exception as e:
                        print(f"❌ 生成股票推荐报告时出错: {e}")
                        import traceback
                        traceback.print_exc()

                    print("\n感谢使用股票智能模拟分析系统！")
                    print("再见！👋")
                return

        # 询问是否立即开始批量分析
        print(f"\n将要分析 {len(stock_list)} 只股票，配置如下:")
        print(f"  模拟天数: {days_to_simulate} 天")
        print(f"  报告格式: {output_format}")
        print(f"  自动打开报告: {'是' if auto_open_html else '否'}")

        confirm = input("是否开始批量分析？(Y/n): ").strip().lower()
        if confirm in ['n', 'no']:
            print("批量分析已取消")
            return
    else:
        # 单只股票模式：单独询问参数
        output_format = 'html'
        auto_open_html = True

    # 批量处理股票（如果没有分批）
    print(f"\n分析配置:")
    print(f"  股票数量: {len(stock_list)} 只")
    print(f"  模拟天数: {days_to_simulate} 天")
    print(f"  报告格式: {output_format}")

    report_paths = []  # 保存所有报告路径

    for idx, (stock_code, stock_name) in enumerate(stock_list, 1):
        print(f"\n{'=' * 48}")
        print(f"📊 开始分析第 {idx}/{len(stock_list)} 只股票:")
        print(f"  股票: {stock_name}({stock_code})")
        print(f"{'=' * 48}")

        try:
            report_path = analyze_single_stock(
                stock_code,
                stock_name,
                days_to_simulate,
                output_format,
                auto_open_html
            )
            if report_path:
                report_paths.append(report_path)
                print(f"✓ 第 {idx} 只股票分析完成")

                # ========== 关键：每分析5只股票清理一次 ==========
                if idx % 5 == 0 and idx < len(stock_list):
                    print(f"\n🧹 已分析 {idx} 只股票，清理中间资源...")
                    cleanup_resources()

            else:
                print(f"⚠️ 第 {idx} 只股票分析失败")
        except Exception as e:
            print(f"❌ 第 {idx} 只股票分析失败: {e}")
            continue

    # 批量分析完成后的处理
    print(f"\n{'=' * 48}")
    print(" " * 20 + "🎉 批量分析完成！")
    print(f"{'=' * 48}")

    if report_paths:
        print(f"\n📋 分析结果汇总:")
        print(f"  成功分析: {len(report_paths)}/{len(stock_list)} 只股票")
        print(f"  报告保存位置:")
        for path in report_paths[:10]:  # 只显示前10个
            print(f"    - {os.path.basename(path)}")
        if len(report_paths) > 10:
            print(f"    ... 还有 {len(report_paths) - 10} 个报告")

    # 批量模式：询问是否继续分析其他股票
    if is_batch_mode:
        print("\n批量分析已完成！")
        choice = input("\n是否继续分析其他股票？(y/N): ").strip().lower()
        if choice in ['y', 'yes']:
            main()
        else:
            # ========== 新增：执行股票分析脚本 ==========
            print("\n📊 正在生成股票推荐分析报告...")
            try:
                # 尝试导入stock_analysis模块
                try:
                    import stock_analysis
                    stock_analysis_available = True
                except ImportError:
                    print("⚠️ stock_analysis 模块未安装，跳过股票推荐报告生成")
                    stock_analysis_available = False

                if stock_analysis_available:
                    print("=" * 48)
                    print(" " * 20 + "📈 开始生成股票推荐报告")
                    print("=" * 48)

                    # 指定具体的文件路径
                    summary_file = os.path.join(SAVE_PATH, "风险等级汇总.xlsx")
                    result = stock_analysis.analyze_stock_reports(input_file=summary_file)

                    if result:
                        print("✅ 股票推荐报告生成成功！")
                        print(f"📊 分析完成: {result['total_stocks']} 只股票")
                        print(f"📁 Excel报告: {result['excel_path']}")
                        print(f"🌐 HTML报告: {result['html_path']}")
                    else:
                        print("❌ 股票推荐报告生成失败")
            except Exception as e:
                print(f"❌ 生成股票推荐报告时出错: {e}")
                import traceback
                traceback.print_exc()

            print("\n感谢使用股票智能模拟分析系统！")
            print("再见！👋")
    else:
        # 单只股票模式：保持原来的询问逻辑
        while True:
            try:
                choice = input("\n是否分析其他股票？(y/N): ").strip().lower()
                if choice in ['', 'n', 'no']:
                    print("\n感谢使用股票智能模拟分析系统！")
                    print("再见！👋")
                    break
                elif choice in ['y', 'yes']:
                    main()
                    break
                else:
                    print("请输入 y/yes 或 n/no，或直接按回车退出")
            except KeyboardInterrupt:
                print("\n\n程序已中断，感谢使用！")
                break


# ==================== 程序退出处理 ====================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 最终清理
        cleanup_resources()
        print("\n程序结束，感谢使用！")

        # 关键：使用 os._exit 而不是 sys.exit
        os._exit(0)  # 这会立即终止进程，跳过所有清理