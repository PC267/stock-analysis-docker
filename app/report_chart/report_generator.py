# report_generator.py - 优化版
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import warnings
import logging
import platform
import sys

# 添加report_config.py所在目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import report_config


# 设置保存路径 - 使用report_config模块中的配置
SAVE_PATH = report_config.REPORT_DIR

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略警告信息
warnings.filterwarnings('ignore', category=UserWarning)

# 优化的跨平台中文字体设置
def setup_chinese_font():
    """跨平台中文字体设置 - 优化版"""

    # 先重置为默认配置
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    # 获取当前操作系统
    system_name = platform.system()

    # 根据操作系统设置字体
    if system_name == "Windows":
        # Windows系统：优先使用微软雅黑，备选SimHei
        windows_fonts = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.sans-serif'] = windows_fonts
        print("Windows system font settings: Using Microsoft YaHei font")

    elif system_name == "Darwin":  # macOS
        # macOS系统：使用苹方字体
        mac_fonts = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'Arial']
        matplotlib.rcParams['font.sans-serif'] = mac_fonts
        print("macOS system font settings: Using PingFang SC font")

    else:  # Linux或其他系统
        # Linux系统：使用Noto字体
        linux_fonts = ['Noto Sans CJK SC',
                       'DejaVu Sans', 'Arial Unicode MS', 'Arial']
        matplotlib.rcParams['font.sans-serif'] = linux_fonts
        print(f"Linux system font settings: Using Noto Sans CJK SC font")

    # 确保正常显示负号
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置字体族
    matplotlib.rcParams['font.family'] = 'sans-serif'

    # 优化全局字体大小和间距配置
    matplotlib.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.constrained_layout.use': True,
        'figure.autolayout': True,
    })


# 高级字体回退方案
def advanced_font_setup():
    """高级字体设置，包含智能字体检测"""
    try:
        # 方法1：尝试标准字体设置
        setup_chinese_font()

        # 测试字体是否设置成功
        try:
            # 创建一个测试图形来验证字体设置
            fig, ax = plt.subplots(figsize=(3, 1))
            test_text = '测试中文: 你好世界! 123ABC'
            ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            plt.close(fig)
            print("Chinese font settings successful - Test passed")
            return True
        except Exception as e:
            print(f"Standard font test failed: {e}")
            raise

    except Exception as e:
        print(f"Standard font settings failed: {e}")
        print("Trying advanced font detection...")

        try:
            from matplotlib import font_manager

            # 获取系统所有字体
            font_files = font_manager.findSystemFonts()

            # 常见中文字体列表（按优先级排序）
            chinese_font_candidates = [
                'Microsoft YaHei',  # Windows 微软雅黑
                'SimHei',  # Windows 黑体
                'PingFang SC',  # macOS 苹方
                'Heiti SC',  # macOS 黑体
                'Noto Sans CJK SC',  # Linux/通用
                'Droid Sans Fallback',  # Android字体
                'WenQuanYi Micro Hei',  # 文泉驿微米黑
                'Source Han Sans SC',  # 思源黑体
                'Hiragino Sans GB',  # 冬青黑体
                'STXihei',  # 华文细黑
                'STHeiti',  # 华文黑体
                'SimSun',  # 宋体
                'NSimSun',  # 新宋体
                'FangSong',  # 仿宋
                'KaiTi',  # 楷体
            ]

            # 检查系统中有哪些字体可用
            available_fonts = []
            for font_file in font_files:
                try:
                    font_prop = font_manager.FontProperties(fname=font_file)
                    font_name = font_prop.get_name()
                    # 检查是否是中文字体候选
                    if font_name in chinese_font_candidates:
                        if font_name not in available_fonts:
                            available_fonts.append(font_name)
                except:
                    continue

            print(f"Detected Chinese fonts: {available_fonts}")

            if available_fonts:
                # 使用检测到的字体
                matplotlib.rcParams['font.sans-serif'] = available_fonts + \
                    ['DejaVu Sans', 'Arial']
                print(f"Using detected fonts: {available_fonts[0]}")
            else:
                # 没有检测到中文字体，使用英文字体
                matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
                print("No Chinese fonts detected, using English fonts")

            matplotlib.rcParams['axes.unicode_minus'] = False
            matplotlib.rcParams['font.family'] = 'sans-serif'

            # 测试字体是否设置成功
            fig, ax = plt.subplots(figsize=(3, 1))
            test_text = '测试中文: 你好世界! 123ABC'
            ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            plt.close(fig)
            print("Advanced font detection successful - Test passed")
            return True

        except Exception as e2:
            print(f"Advanced font detection failed: {e2}")
            # 最终回退方案
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            matplotlib.rcParams['axes.unicode_minus'] = False
            print("Using fallback font: DejaVu Sans")
            return False


# 初始化字体设置
def init_fonts():
    """初始化字体设置"""
    print("=" * 48)
    print("Initializing Chinese font settings...")
    print("=" * 48)

    try:
        success = advanced_font_setup()
        if success:
            print("Font initialization completed")
            return True
        else:
            raise Exception("Font settings failed")
    except Exception as e:
        print(f"Font initialization failed: {e}")
        print("Using minimal font configuration...")
        # 使用最通用的字体配置
        matplotlib.rcParams.update({
            'font.sans-serif': ['DejaVu Sans', 'Arial'],
            'axes.unicode_minus': False,
            'font.family': 'sans-serif'
        })
        return False


# 在模块导入时初始化字体
_font_initialized = False
if not _font_initialized:
    init_fonts()
    _font_initialized = True

class ReportGenerator:
    def __init__(self, config):
        self.config = config
        # 使用report_config模块中的保存路径
        self.save_path = SAVE_PATH
        # 确保保存目录存在
        os.makedirs(self.save_path, exist_ok=True)
        # 风险记录文件路径
        self.risk_summary_file = os.path.join(self.save_path, "风险等级汇总.xlsx")

    def generate_report(self, processed_data, simulated_data, models, best_model_name):
        """生成详细分析报告"""
        print("=" * 48)
        print(f"正在分析 {self.config.stock_name}({self.config.stock_code})...")
        print("=" * 48)

        try:
            # 检查数据是否存在
            if processed_data is None or 'processed' not in processed_data:
                print(f"错误: {self.config.stock_name} 处理数据不存在")
                return None

            hist_data = processed_data['processed']
            if hist_data is None or len(hist_data) == 0:
                print(f"错误: {self.config.stock_name} 历史数据为空")
                return None

            # 生成可视化报告（图片格式）
            visual_file = self._create_visual_report(hist_data, simulated_data, models, best_model_name)

            # 生成HTML报告
            html_file = self._create_html_report(hist_data, simulated_data, models, best_model_name, visual_file)

            # 计算综合风险评估
            risk_analysis = self._calculate_comprehensive_risk(simulated_data)

            # ==================== 修改：获取得分前三模型名字 ====================
            top_models_names = self._get_top_models_names(models)
            risk_analysis['top_models'] = top_models_names

            # 添加最优模型信息到risk_analysis（保持向后兼容）
            risk_analysis['best_model_name'] = best_model_name
            # ================================================================

            # 计算详细的股票统计指标
            stock_stats = self._calculate_stock_statistics(hist_data)
            risk_analysis['stock_stats'] = stock_stats

            # 将风险信息保存到Excel汇总表
            risk_analysis['simulated_data'] = simulated_data
            self._save_risk_to_excel(risk_analysis)

            print(f"{self.config.stock_name} 分析完成！")
            return {
                'visual_report': visual_file,
                'html_report': html_file,
                'risk_analysis': risk_analysis,
                'stock_stats': stock_stats
            }

        except Exception as e:
            print(f"{self.config.stock_name} 分析时出错: {e}")
            import traceback
            traceback.print_exc()
            self._save_error_report(e)
            return None

    def _get_top_models_names(self, models):
        """获取得分前三模型名字"""
        try:
            # 检查models参数
            if not isinstance(models, dict):
                return "未知"

            # 从models中提取模型评分信息
            model_scores = {}

            for model_name, model_info in models.items():
                if isinstance(model_info, dict):
                    # 尝试从模型信息中获取得分
                    if 'score' in model_info:
                        score = model_info['score']
                        if isinstance(score, (int, float)):
                            model_scores[model_name] = score
                    elif 'aic' in model_info and model_info['aic'] is not None:
                        # 使用AIC作为得分依据（AIC越小越好，取绝对值）
                        try:
                            aic_value = abs(float(model_info['aic']))
                            model_scores[model_name] = aic_value
                        except:
                            pass
                    elif 'rmse' in model_info and model_info['rmse'] is not None:
                        # 使用RMSE作为得分依据（RMSE越小越好，取倒数）
                        try:
                            rmse_value = float(model_info['rmse'])
                            if rmse_value > 0:
                                model_scores[model_name] = 1.0 / rmse_value
                        except:
                            pass
                    else:
                        # 没有评分信息，使用默认值
                        model_scores[model_name] = 1.0
                else:
                    # 模型信息不是字典，使用默认值
                    model_scores[model_name] = 1.0

            # 如果只有一个模型，直接返回
            if len(model_scores) <= 1:
                return list(model_scores.keys())[0] if model_scores else "未知"

            # 按得分排序（降序）
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

            # 获取前三模型名字
            top_models = []
            for i, (model_name, score) in enumerate(sorted_models[:3]):
                if i == 0:
                    # 第一个模型显示原始名字
                    top_models.append(model_name)
                else:
                    # 其他模型只显示名字，不显示得分
                    top_models.append(model_name)

            # 用逗号连接
            if len(top_models) == 1:
                return top_models[0]
            else:
                return ", ".join(top_models)

        except Exception as e:
            print(f"获取得分前三模型失败: {e}")
            return "未知"

    def _create_html_report(self, hist_data, simulated_data, models, best_model_name, visual_file):
        """创建HTML格式的报告"""
        try:
            # 计算风险评估
            risk_analysis = self._calculate_comprehensive_risk(simulated_data)

            # 计算股票统计指标
            stock_stats = self._calculate_stock_statistics(hist_data)

            # 准备数据
            recent_days = min(15, len(hist_data))
            historical_data = hist_data.iloc[-recent_days:]

            # 计算价格统计
            if simulated_data is not None and len(simulated_data) > 0:
                last_sim_price = simulated_data['close'].iloc[-1]
                first_sim_price = simulated_data['close'].iloc[0]
                total_return = ((last_sim_price / first_sim_price - 1) * 100)
            else:
                total_return = 0
                first_sim_price = hist_data['close'].iloc[-1] if len(hist_data) > 0 else 0
                last_sim_price = first_sim_price

            # 获取可视化图片文件名
            visual_filename = os.path.basename(visual_file) if visual_file else "report.png"

            # 创建HTML内容
            html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Analysis Report -({self.config.stock_code})</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
        }}

        body {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
            min-height: 100vh;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(to right, #2c3e50, #4a6491);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
            background-size: 20px 20px;
            animation: float 20s linear infinite;
        }}

        @keyframes float {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .stock-title {{
            font-size: 2.8em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            position: relative;
            z-index: 1;
        }}

        .stock-code {{
            font-size: 1.4em;
            opacity: 0.9;
            margin-bottom: 20px;
            position: relative;
            z-index: 1;
        }}

        .timestamp {{
            background: rgba(255,255,255,0.1);
            display: inline-block;
            padding: 10px 25px;
            border-radius: 50px;
            font-size: 1.1em;
            position: relative;
            z-index: 1;
            backdrop-filter: blur(10px);
        }}

        .content {{
            padding: 40px;
        }}

        .section {{
            margin-bottom: 50px;
            animation: fadeIn 1s ease-out;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .section-title {{
            color: #2c3e50;
            font-size: 1.8em;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid #3498db;
            position: relative;
        }}

        .section-title::after {{
            content: '';
            position: absolute;
            bottom: -3px;
            left: 0;
            width: 100px;
            height: 3px;
            background: #e74c3c;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-top: 20px;
        }}

        .stat-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .stat-card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 5px;
            background: linear-gradient(to right, #3498db, #2ecc71);
        }}

        .stat-label {{
            font-size: 1.1em;
            color: #7f8c8d;
            margin-bottom: 12px;
            font-weight: 500;
        }}

        .stat-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #2c3e50;
        }}

        .risk-highlight {{
            background: linear-gradient(135deg, #ff6b6b, #ee5a52);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            position: relative;
            overflow: hidden;
        }}

        .risk-highlight::before {{
            content: '⚠';
            position: absolute;
            right: 30px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 6em;
            opacity: 0.1;
        }}

        .suggestion-box {{
            background: linear-gradient(135deg, #2ecc71, #27ae60);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}

        .visualization {{
            text-align: center;
            margin: 40px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 15px;
        }}

        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}

        .visualization img:hover {{
            transform: scale(1.02);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }}

        .model-info {{
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
        }}

        .footer {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 30px;
            text-align: center;
            border-top: 5px solid #3498db;
        }}

        .disclaimer {{
            font-size: 0.9em;
            opacity: 0.7;
            margin-top: 20px;
            line-height: 1.5;
        }}

        /* 风险等级颜色 */
        .risk-level-Extreme Risk {{ color: #c0392b; font-weight: bold; }}
        .risk-level-High Risk {{ color: #e74c3c; font-weight: bold; }}
        .risk-level-Medium Risk {{ color: #f39c12; font-weight: bold; }}
        .risk-level-Low Risk {{ color: #27ae60; font-weight: bold; }}
        .risk-level-Very Low Risk {{ color: #16a085; font-weight: bold; }}

        @media (max-width: 768px) {{
            .container {{
                margin: 10px;
                border-radius: 15px;
            }}

            .header {{
                padding: 20px;
            }}

            .stock-title {{
                font-size: 2em;
            }}

            .content {{
                padding: 20px;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Area -->
        <div class="header">
            <h1 class="stock-title">{self.config.stock_name}</h1>
            <div class="stock-code">Stock Code: {self.config.stock_code}</div>
            <div class="timestamp">Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <!-- Content Area -->
        <div class="content">
            <!-- Risk Overview -->
            <div class="section">
                <h2 class="section-title">⚡ Risk Overview</h2>
                <div class="risk-highlight">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                        <div>
                            <div class="stat-label">Risk Level</div>
                            <div class="stat-value risk-level-{risk_analysis['risk_level_1']}">
                                {risk_analysis['risk_level_1']}
                            </div>
                        </div>
                        <div>
                            <div class="stat-label">Risk Score</div>
                            <div class="stat-value" style="color: white;">{risk_analysis['risk_score_1']:.3f}/1.0</div>
                        </div>
                        <div>
                            <div class="stat-label">Max Drawdown</div>
                            <div class="stat-value" style="color: white;">{risk_analysis['max_drawdown'] * 100:.2f}%</div>
                        </div>
                    </div>
                </div>

                <!-- Investment Suggestion -->
                <div class="suggestion-box">
                    <div class="stat-label" style="color: rgba(255,255,255,0.9);">📋 Investment Suggestion</div>
                    <div class="stat-value" style="font-size: 1.5em;">{risk_analysis['suggestion_1']}</div>
                </div>
            </div>

            <!-- Key Metrics -->
            <div class="section">
                <h2 class="section-title">📊 Key Metrics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Start Price</div>
                        <div class="stat-value">¥{first_sim_price:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">End Price</div>
                        <div class="stat-value">¥{last_sim_price:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Return</div>
                        <div class="stat-value" style="color: {'red' if total_return < 0 else 'green'}">
                            {total_return:+.2f}%
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Volatility</div>
                        <div class="stat-value">{risk_analysis['volatility'] * 100:.2f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Sharpe Ratio</div>
                        <div class="stat-value">{risk_analysis['sharpe_ratio']:.3f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value">{risk_analysis['win_rate'] * 100:.1f}%</div>
                    </div>
                </div>
            </div>

            <!-- Detailed Statistics -->
            <div class="section">
                <h2 class="section-title">📈 Detailed Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Current Price</div>
                        <div class="stat-value">¥{stock_stats.get('current_price', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Max Price</div>
                        <div class="stat-value">¥{stock_stats.get('max_price', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Min Price</div>
                        <div class="stat-value">¥{stock_stats.get('min_price', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average Price</div>
                        <div class="stat-value">¥{stock_stats.get('avg_price', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Price Std Dev</div>
                        <div class="stat-value">{stock_stats.get('price_std', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Annual Return</div>
                        <div class="stat-value">{stock_stats.get('annual_return', 0):.2%}</div>
                    </div>
                </div>
            </div>

            <!-- Model Information -->
            <div class="section">
                <h2 class="section-title">🤖 Model Analysis</h2>
                <div class="model-info">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                        <div>
                            <div class="stat-label" style="color: rgba(255,255,255,0.9);">Best Model</div>
                            <div class="stat-value" style="font-size: 1.8em; color: white;">{best_model_name}</div>
                        </div>
                        <div>
                            <div class="stat-label" style="color: rgba(255,255,255,0.9);">Analysis Period</div>
                            <div class="stat-value" style="font-size: 1.8em; color: white;">{len(historical_data)} days</div>
                        </div>
                    </div>
                    <div style="margin-top: 20px;">
                        <div class="stat-label" style="color: rgba(255,255,255,0.9);">Available Models</div>
                        <div style="color: white; font-size: 1.1em;">{', '.join(models.keys())}</div>
                    </div>
                </div>
            </div>

            <!-- Visualization Chart -->
            <div class="section">
                <h2 class="section-title">📈 Visualization Analysis</h2>
                <div class="visualization">
                    <p style="margin-bottom: 20px; color: #666; font-size: 1.1em;">
                        Comprehensive Analysis Chart (Click image to view larger version)
                    </p>
                    <img src="{visual_filename}" alt="Stock Analysis Chart" onclick="window.open(this.src, '_blank')" style="cursor: pointer;">
                    <p style="margin-top: 15px; color: #999; font-size: 0.9em;">
                        Includes price trends, K-line chart, volume comparison, and risk distribution analysis
                    </p>
                </div>
            </div>

            <!-- Data Summary -->
            <div class="section">
                <h2 class="section-title">📋 Data Summary</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Historical Data Days</div>
                        <div class="stat-value">{len(historical_data)} days</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Simulated Data Days</div>
                        <div class="stat-value">{len(simulated_data) if simulated_data is not None else 0} days</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Total Trading Days</div>
                        <div class="stat-value">{len(historical_data) + (len(simulated_data) if simulated_data is not None else 0)} days</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Data Start Date</div>
                        <div class="stat-value">{hist_data.index[0].strftime('%Y-%m-%d') if len(hist_data) > 0 else 'N/A'}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Data End Date</div>
                        <div class="stat-value">{hist_data.index[-1].strftime('%Y-%m-%d') if len(hist_data) > 0 else 'N/A'}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <div style="font-size: 1.2em; margin-bottom: 15px;">Analysis Report Generated</div>
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This report is generated based on historical data and model analysis, for reference and research only.<br>
                Investment involves risks, caution is required. Report generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                Generation method: Python Stock Analysis System v2.0 | © 2024 Quantitative Analysis Tool
            </div>
        </div>
    </div>

    <script>
        // 添加简单的交互效果
        document.addEventListener('DOMContentLoaded', function() {{
            // 为所有卡片添加点击效果
            const cards = document.querySelectorAll('.stat-card');
            cards.forEach(card => {{
                card.addEventListener('click', function() {{
                    this.style.transform = 'scale(0.98)';
                    setTimeout(() => {{
                        this.style.transform = '';
                    }}, 150);
                }});
            }});

            // 风险等级高亮效果
            const riskElements = document.querySelectorAll('[class*="risk-level-"]');
            riskElements.forEach(el => {{
                el.addEventListener('mouseenter', function() {{
                    this.style.textShadow = '0 0 10px currentColor';
                }});
                el.addEventListener('mouseleave', function() {{
                    this.style.textShadow = 'none';
                }});
            }});
        }});
    </script>
</body>
</html>
            """

            # 保存HTML文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_filename = f'{self.config.stock_code}_{self.config.stock_name}_HTML报告_{timestamp}.html'
            html_filepath = os.path.join(self.save_path, html_filename)

            with open(html_filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)

            print(f"HTML报告已生成: {html_filepath}")
            return html_filepath

        except Exception as e:
            print(f"创建HTML报告失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_comprehensive_risk(self, simulated_data):
        """综合风险评估"""
        if simulated_data is None or len(simulated_data) <= 1:
            return {
                'stock_code': self.config.stock_code,
                'stock_name': self.config.stock_name,
                'risk_level': '未知',
                'risk_level_1': 'Unknown',
                'volatility': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'risk_score': 0,
                'risk_score_1': 0,
                'suggestion': '数据不足',
                'suggestion_1': 'insufficient data',
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        # 计算收益率
        sim_returns = simulated_data['close'].pct_change().dropna()
        if len(sim_returns) == 0:
            return {
                'stock_code': self.config.stock_code,
                'stock_name': self.config.stock_name,
                'risk_level': '未知',
                'risk_level_1': 'Unknown',
                'volatility': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'risk_score': 0,
                'risk_score_1': 0,
                'suggestion': '数据不足',
                'suggestion_1': 'insufficient data',
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        # 1. 波动率（年化）
        volatility = sim_returns.std() * np.sqrt(252)

        # 2. 最大回撤
        cum_returns = (1 + sim_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # 3. 夏普比率（简化版，假设无风险利率为0）
        sharpe_ratio = sim_returns.mean() / sim_returns.std() if sim_returns.std() > 0 else 0

        # 4. 胜率
        win_rate = (sim_returns > 0).sum() / len(sim_returns)

        # 综合风险评分_1
        risk_score_1 = (
                min(volatility * 10, 1) * 0.4 +
                min(max_drawdown * 20, 1) * 0.3 +
                (1 - min(sharpe_ratio * 5, 1)) * 0.2 +
                (1 - win_rate) * 0.1
        )

        # 风险等级划分
        if risk_score_1 >= 0.7:
            risk_level_1 = "Extreme Risk"
            suggestion_1 = "Extremely high risk, recommend avoiding operations"
        elif risk_score_1 >= 0.5:
            risk_level_1 = "High Risk"
            suggestion_1 = "Recommend strict position control, set stop loss"
        elif risk_score_1 >= 0.3:
            risk_level_1 = "Medium Risk"
            suggestion_1 = "Recommend batch position building, control position within 50%"            
        elif risk_score_1 >= 0.1:    
            risk_level_1 = "Low Risk"
            suggestion_1 = "Risk controllable, suitable for stable investment"            
        else :    
            risk_level_1 = "Very Low Risk"
            suggestion_1 = "Very low risk, suitable for conservative investment"
        # 综合风险评分
        risk_score = (
                min(volatility * 10, 1) * 0.4 +
                min(max_drawdown * 20, 1) * 0.3 +
                (1 - min(sharpe_ratio * 5, 1)) * 0.2 +
                (1 - win_rate) * 0.1
        )

        # 风险等级划分
        if risk_score >= 0.7:
            risk_level = "极高风险"
            suggestion = "风险极高，建议避免操作"
        elif risk_score >= 0.5:
            risk_level = "高风险"
            suggestion = "建议严格控制仓位，设置止损"
        elif risk_score >= 0.3:
            risk_level = "中风险"
            suggestion = "建议分批建仓，控制仓位在50%以内"        
        elif risk_score >= 0.1:
            risk_level = "低风险"
            suggestion = "风险可控，适合稳健投资"         
        else : 
            risk_level = "极低风险"
            suggestion = "风险极低，适合保守型投资"
        return {
            'stock_code': self.config.stock_code,
            'stock_name': self.config.stock_name,
            'risk_level': risk_level,
            'risk_score_1': risk_score_1,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'risk_score': risk_score,
            'risk_level_1': risk_level_1,
            'suggestion': suggestion,
            'suggestion_1': suggestion_1,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _calculate_stock_statistics(self, df):
        """计算股票统计指标（从第二个文件优化）"""
        stats = {}

        # 价格统计
        stats['current_price'] = df['close'].iloc[-1]
        stats['max_price'] = df['close'].max()
        stats['max_price_date'] = df['close'].idxmax().strftime('%Y-%m-%d') if not df.empty else 'N/A'
        stats['min_price'] = df['close'].min()
        stats['min_price_date'] = df['close'].idxmin().strftime('%Y-%m-%d') if not df.empty else 'N/A'
        stats['avg_price'] = df['close'].mean()
        stats['price_std'] = df['close'].std()

        # 收益率统计
        if len(df) > 0:
            total_return = df['close'].iloc[-1] / df['close'].iloc[0] - 1
        else:
            total_return = 0
        stats['total_return'] = total_return

        # 计算年化收益率
        if len(df) > 1:
            days = (df.index[-1] - df.index[0]).days
            if days > 0:
                annual_return = (1 + total_return) ** (365 / days) - 1
            else:
                annual_return = 0
        else:
            annual_return = 0
        stats['annual_return'] = annual_return

        # 日收益率统计
        daily_returns = df['close'].pct_change().dropna()
        if len(daily_returns) > 0:
            stats['daily_return_mean'] = daily_returns.mean()
            stats['daily_return_std'] = daily_returns.std()
        else:
            stats['daily_return_mean'] = 0
            stats['daily_return_std'] = 0

        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03 / 252
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            excess_returns = daily_returns - risk_free_rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        stats['sharpe_ratio'] = sharpe_ratio

        # 最大回撤
        if len(daily_returns) > 0:
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            stats['max_drawdown'] = drawdown.min()
        else:
            stats['max_drawdown'] = 0

        # 成交量统计
        if 'volume' in df.columns:
            stats['avg_volume'] = df['volume'].mean()
            stats['volume_std'] = df['volume'].std()
            # 量价相关性
            stats['volume_price_corr'] = df['volume'].corr(df['close'])
        else:
            stats['avg_volume'] = 0
            stats['volume_std'] = 0
            stats['volume_price_corr'] = 0

        # 技术指标当前值（如果有）
        stats['rsi_current'] = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns and len(df) > 0 else 0
        stats['macd_current'] = df['macd'].iloc[-1] if 'macd' in df.columns and len(df) > 0 else 0

        # 布林带位置（如果有）
        if all(col in df.columns for col in ['boll_upper', 'boll_lower']) and len(df) > 0:
            boll_position = (df['close'].iloc[-1] - df['boll_lower'].iloc[-1]) / \
                            (df['boll_upper'].iloc[-1] - df['boll_lower'].iloc[-1])
            stats['boll_position'] = boll_position
        else:
            stats['boll_position'] = 0

        # 波动率
        if len(daily_returns) >= 20:
            stats['volatility_20d'] = daily_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        else:
            stats['volatility_20d'] = 0

        return stats

    def _save_risk_to_excel(self, risk_analysis):
        """将风险信息保存到Excel汇总表"""
        try:
            # 从risk_analysis中提取模拟数据的最高价和最低价
            if 'simulated_data' in risk_analysis and risk_analysis['simulated_data'] is not None:
                sim_data = risk_analysis['simulated_data']
                sim_high_price = sim_data['high'].max() if len(sim_data) > 0 else 0
                sim_low_price = sim_data['low'].min() if len(sim_data) > 0 else 0
                sim_price_diff = sim_high_price - sim_low_price
                sim_days = len(sim_data)
            else:
                # 如果没有模拟数据，使用默认值
                sim_high_price = 0
                sim_low_price = 0
                sim_price_diff = 0
                sim_days = 0

            # ==================== 修改：获取得分前三模型名字 ====================
            # 从risk_analysis中提取最优模型信息
            top_models = risk_analysis.get('top_models', '未知模型')

            # 准备要保存的数据
            data_to_save = {
                '股票代码': risk_analysis['stock_code'],
                '股票名称': risk_analysis['stock_name'],
                '风险等级': risk_analysis['risk_level'],
                '风险评分': f"{risk_analysis['risk_score']:.3f}",
                '最优模型': top_models,  # 使用得分前三模型名字
                '波动率': f"{risk_analysis['volatility']:.4f}",
                '最大回撤': f"{risk_analysis['max_drawdown']:.4f}",
                '夏普比率': f"{risk_analysis['sharpe_ratio']:.4f}",
                '胜率': f"{risk_analysis['win_rate']:.2%}",
                '分析建议': risk_analysis['suggestion'],
                '分析时间': risk_analysis['analysis_time'],
                '模拟天数': sim_days,
                '模拟最高价': f"{sim_high_price:.4f}",
                '模拟最低价': f"{sim_low_price:.4f}",
                '模拟价格差': f"{sim_price_diff:.4f}"
            }
            # ================================================================

            df_new = pd.DataFrame([data_to_save])

            # 检查Excel文件是否存在
            if os.path.exists(self.risk_summary_file):
                try:
                    # 读取现有文件
                    df_existing = pd.read_excel(self.risk_summary_file)

                    # 检查是否已有相同股票代码的记录
                    mask = df_existing['股票代码'] == risk_analysis['stock_code']
                    if mask.any():
                        # 更新现有记录
                        for col in df_new.columns:
                            if col in df_existing.columns:
                                df_existing.loc[mask, col] = df_new.iloc[0][col]
                        # 添加可能的新列
                        new_cols = [col for col in df_new.columns if col not in df_existing.columns]
                        if new_cols:
                            for col in new_cols:
                                df_existing[col] = None
                                df_existing.loc[mask, col] = df_new.iloc[0][col]
                    else:
                        # 添加新记录，确保列顺序一致
                        if set(df_new.columns) != set(df_existing.columns):
                            # 合并列
                            all_cols = list(df_existing.columns) + [col for col in df_new.columns if
                                                                    col not in df_existing.columns]
                            df_existing = df_existing.reindex(columns=all_cols)
                            df_new = df_new.reindex(columns=all_cols)
                        # 添加新记录
                        df_existing = pd.concat([df_existing, df_new], ignore_index=True)

                    df_to_save = df_existing
                except Exception as e:
                    print(f"读取现有Excel文件出错，创建新文件: {e}")
                    df_to_save = df_new
            else:
                df_to_save = df_new

            # 保存到Excel
            try:
                df_to_save.to_excel(self.risk_summary_file, index=False)
                print(f"{risk_analysis['stock_name']} 风险信息已保存到Excel汇总表")
            except Exception as e:
                print(f"保存Excel文件失败: {e}")
                # 尝试保存为CSV作为备选方案
                csv_file = self.risk_summary_file.replace('.xls', '.csv')
                df_to_save.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"风险信息已保存到CSV文件: {csv_file}")

        except Exception as e:
            print(f"保存风险信息时出错: {e}")

    def _create_visual_report(self, hist_data, simulated_data, models, best_model_name):
        """创建可视化报告图表 - 批量模式下简化"""
        try:
            # 清空之前的图形
            plt.close('all')

            # 创建图形
            fig = plt.figure(figsize=(22, 18), facecolor='#f5f5f5')

            # 设置全局标题
            fig.suptitle(f'Stock Analysis Report - ({self.config.stock_code})\n'
                         f'(For learning and research purposes only, consequences for other uses)',
                         fontsize=20, fontweight='bold', y=0.98, color='#1A237E')

            # 准备数据
            recent_days = min(15, len(hist_data))
            historical_data = hist_data.iloc[-recent_days:]

            # 1. 分析数据统计（顶部）
            ax_stats = plt.subplot(3, 2, (1, 2))
            ax_stats.set_facecolor('#E8F5E9')
            ax_stats.axis('off')

            # 计算综合风险评估
            risk_analysis = self._calculate_comprehensive_risk(simulated_data)

            # 计算股票统计指标
            stock_stats = self._calculate_stock_statistics(hist_data)

            if simulated_data is not None and len(simulated_data) > 0:
                last_sim_price = simulated_data['close'].iloc[-1]
                first_sim_price = simulated_data['close'].iloc[0]
                total_return = ((last_sim_price / first_sim_price - 1) * 100)
            else:
                total_return = 0
                first_sim_price = hist_data['close'].iloc[-1] if len(
                    hist_data) > 0 else 0
                last_sim_price = first_sim_price

            # 创建统计表格
            stats_data = [
                ["Analysis Data Statistics", ""],
                ["-" * 48, ""],
                ["Start Price", f"{first_sim_price:.2f} CNY"],
                ["End Price", f"{last_sim_price:.2f} CNY"],
                ["Total Return", f"{total_return:+.2f}%"],
                ["Risk Level", f"{risk_analysis['risk_level_1']}"],
                ["Risk Score", f"{risk_analysis['risk_score_1']:.3f}/1.0"],
                ["Volatility", f"{risk_analysis['volatility'] * 100:.2f}%"],
                ["Max Drawdown", f"{risk_analysis['max_drawdown'] * 100:.2f}%"],
                ["Sharpe Ratio", f"{risk_analysis['sharpe_ratio']:.3f}"],
                ["Win Rate", f"{risk_analysis['win_rate'] * 100:.1f}%"],
                ["Analysis Suggestion", f"{risk_analysis['suggestion_1']}"],
                ["Annual Return", f"{stock_stats.get('annual_return', 0):.2%}"],
                ["Current Price", f"{stock_stats.get('current_price', 0):.2f} CNY"],
                ["Average Price", f"{stock_stats.get('avg_price', 0):.2f} CNY"],
            ]

            # 创建表格
            table = ax_stats.table(cellText=stats_data,
                                   cellLoc='left',
                                   loc='center',
                                   colWidths=[0.35, 0.4])

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 2.2)

            # 设置单元格样式
            for i in range(len(stats_data)):
                for j in range(2):
                    cell = table[(i, j)]
                    cell.set_edgecolor('#C8E6C9')

                    if i == 0:
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(
                            weight='bold', color='white', fontsize=13)
                    elif i == 1:
                        cell.set_facecolor('#E8F5E9')
                        cell.set_text_props(fontsize=9)
                    else:
                        if i % 2 == 0:
                            cell.set_facecolor('#F1F8E9')
                        else:
                            cell.set_facecolor('#E8F5E9')

                        # 风险等级单元格高亮
                        if i == 5:  # 风险等级
                            risk_color = {
                                'Extreme Risk': '#B71C1C',
                                'High Risk': '#F44336',
                                'Medium Risk': '#FF9800',
                                'Low Risk': '#4CAF50',
                                'Very Low Risk': '#2E7D32',
                                'Unknown': '#757575'
                            }.get(risk_analysis['risk_level'], '#757575')
                            cell.set_facecolor(risk_color)
                            cell.set_text_props(color='white', weight='bold')
                        elif i == 6:  # 风险评分
                            score = risk_analysis['risk_score']
                            if score >= 0.7:
                                color = '#F44336'
                            elif score >= 0.5:
                                color = '#FF9800'
                            elif score >= 0.3:
                                color = '#FFC107'
                            elif score >= 0.1:
                                color = '#8BC34A'
                            else:
                                color = '#4CAF50'
                            cell.set_facecolor(color)
                            cell.set_text_props(color='white', weight='bold')
                        elif i == 11:  # 分析建议
                            cell.set_facecolor('#E3F2FD')
                            cell.set_text_props(color='#1565C0')
                        elif i >= 12:  # 新增统计指标
                            cell.set_facecolor('#F5F5F5')
                            cell.set_text_props(color='#333333')

            # 2. 价格走势图
            ax1 = plt.subplot(3, 2, 3)
            ax1.set_facecolor('white')

            # 历史价格
            ax1.plot(historical_data.index, historical_data['close'],
                     color='#2E7D32', linewidth=3, label='Historical Close Price',
                     marker='o', markersize=6, markerfacecolor='white', markeredgewidth=2)

            # 分析价格
            if simulated_data is not None and len(simulated_data) > 0:
                ax1.plot(simulated_data.index, simulated_data['close'],
                         color='#1565C0', linewidth=3, label='Simulated Close Price',
                         marker='s', markersize=6, markerfacecolor='white', markeredgewidth=2)

                # 分界线
                ax1.axvline(x=historical_data.index[-1], color='#D32F2F',
                            linestyle='--', linewidth=2, alpha=0.7, label='Analysis Start')

            ax1.set_title('Right of red line shows price trend analysis',
                          fontsize=13, fontweight='bold', pad=15, color='#333333')
            ax1.set_ylabel('Price (CNY)', fontsize=12, fontweight='bold')
            ax1.legend(loc='best', fontsize=10, framealpha=0.9)
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.tick_params(axis='x', rotation=45, labelsize=10)
            ax1.tick_params(axis='y', labelsize=10)

            # 3. K线图
            ax2 = plt.subplot(3, 2, 4)
            ax2.set_facecolor('white')

            # 绘制历史K线
            for idx in range(len(historical_data)):
                row = historical_data.iloc[idx]
                # 历史K线颜色设置
                # 红涨绿跌
                color = '#F44336' if row['close'] >= row['open'] else '#4CAF50'

                # 影线
                ax2.plot([idx, idx], [row['low'], row['high']],
                         color=color, linewidth=1.5, alpha=0.8)

                # 实体
                body_width = 0.6
                body_top = max(row['open'], row['close'])
                body_bottom = min(row['open'], row['close'])
                body_height = body_top - body_bottom

                if body_height > 0:
                    ax2.add_patch(plt.Rectangle(
                        (idx - body_width / 2, body_bottom),
                        body_width, body_height,
                        facecolor=color, edgecolor=color, alpha=0.7
                    ))

            # 绘制分析K线
            if simulated_data is not None and len(simulated_data) > 0:
                offset = len(historical_data)
                for idx in range(len(simulated_data)):
                    row = simulated_data.iloc[idx]
                    # 模拟K线颜色设置
                    # 红涨绿跌
                    color = '#F44336' if row['close'] >= row['open'] else '#4CAF50'

                    # 影线
                    ax2.plot([offset + idx, offset + idx], [row['low'], row['high']],
                             color=color, linewidth=1.5, alpha=0.8)

                    # 实体
                    body_width = 0.6
                    body_top = max(row['open'], row['close'])
                    body_bottom = min(row['open'], row['close'])
                    body_height = body_top - body_bottom

                    if body_height > 0:
                        ax2.add_patch(plt.Rectangle(
                            (offset + idx - body_width / 2, body_bottom),
                            body_width, body_height,
                            facecolor=color, edgecolor=color, alpha=0.7
                        ))

            # 设置x轴标签
            all_indices = list(range(len(historical_data)))
            date_labels = [d.strftime('%m-%d') for d in historical_data.index]
            if simulated_data is not None:
                all_indices += list(range(len(historical_data),
                                    len(historical_data) + len(simulated_data)))
                date_labels += [d.strftime('%m-%d')
                                for d in simulated_data.index]

            step = max(1, len(all_indices) // 8)
            ax2.set_xticks(all_indices[::step])
            ax2.set_xticklabels(date_labels[::step], rotation=45, fontsize=10)

            if simulated_data is not None:
                ax2.axvline(x=len(historical_data) - 0.5, color='#D32F2F',
                            linestyle='--', linewidth=2, alpha=0.7)

            ax2.set_title('K-line Chart (Right of red line shows simulated K-line)',
                          fontsize=13, fontweight='bold', pad=15, color='#333333')
            ax2.set_ylabel('Price (CNY)', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')

            # 4. 成交量图
            ax3 = plt.subplot(3, 2, 5)
            ax3.set_facecolor('white')

            # 历史成交量
            hist_indices = list(range(len(historical_data)))
            hist_volumes = historical_data['volume'].values

            colors1 = ['#4CAF50' if historical_data['close'].iloc[i] >= historical_data['open'].iloc[i]
                       else '#F44336' for i in range(len(historical_data))]

            ax3.bar(hist_indices, hist_volumes, color=colors1, alpha=0.7, width=0.6,
                    label='Historical Volume')

            # 分析成交量
            if simulated_data is not None and len(simulated_data) > 0:
                sim_indices = list(range(len(historical_data), len(
                    historical_data) + len(simulated_data)))
                sim_volumes = simulated_data['volume'].values

                colors2 = ['#2196F3' if simulated_data['close'].iloc[i] >= simulated_data['open'].iloc[i]
                           else '#1976D2' for i in range(len(simulated_data))]

                ax3.bar(sim_indices, sim_volumes, color=colors2, alpha=0.7, width=0.6,
                        label='Simulated Volume')

                # 分界线
                ax3.axvline(x=len(historical_data) - 0.5, color='#D32F2F',
                            linestyle='--', linewidth=2, alpha=0.7)

            # 设置x轴
            ax3.set_xticks(all_indices[::step])
            ax3.set_xticklabels(date_labels[::step], rotation=45, fontsize=10)

            ax3.set_title('Right of red line shows volume comparison analysis',
                          fontsize=13, fontweight='bold', pad=15, color='#333333')
            ax3.set_ylabel('Volume', fontsize=12, fontweight='bold')
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3, axis='y', linestyle='--')

            # 5. 风险分析饼图
            ax4 = plt.subplot(3, 2, 6)
            ax4.set_facecolor('white')

            # 风险分析数据
            if simulated_data is not None and len(simulated_data) > 0:
                sim_returns = simulated_data['close'].pct_change().dropna()
                if len(sim_returns) > 0:
                    # 定义涨跌阈值
                    up_threshold = 0.01
                    down_threshold = -0.01

                    # 计算各类天数
                    up_days = sum(sim_returns > up_threshold)
                    down_days = sum(sim_returns < down_threshold)
                    neutral_days = len(sim_returns) - up_days - down_days
                    total_days = len(sim_returns)

                    if total_days > 0:
                        risk_labels = [
                            f'Up Days\n({up_days} days)',
                            f'Down Days\n({down_days} days)',
                            f'Neutral Days\n({neutral_days} days)'
                        ]

                        risk_sizes = [up_days, down_days, neutral_days]
                        risk_colors = ['#F44336', '#4CAF50',  '#FFC107']
                        explode = (0.05, 0.05, 0)

                        # 绘制饼图
                        wedges, texts, autotexts = ax4.pie(
                            risk_sizes,
                            labels=risk_labels,
                            colors=risk_colors,
                            autopct=lambda
                            pct: f'{pct:.1f}%\n({int(round(pct / 100 * total_days))} days)' if pct > 0 else '',
                            startangle=90,
                            explode=explode,
                            wedgeprops={
                                'edgecolor': 'white',
                                'linewidth': 2,
                                'alpha': 0.8
                            }
                        )

                        # 添加环形效果
                        centre_circle = plt.Circle((0, 0), 0.4, fc='white')
                        ax4.add_artist(centre_circle)

                        # 添加中心统计信息
                        avg_return = sim_returns.mean() * 100
                        total_return_pct = (
                            (simulated_data['close'].iloc[-1] / simulated_data['close'].iloc[0] - 1) * 100)

                        centre_text = f'Total {total_days} days\n'
                        centre_text += f'Average {avg_return:+.1f}%\n'
                        centre_text += f'Cumulative {total_return_pct:+.1f}%'

                        ax4.text(0, 0, centre_text,
                                 ha='center', va='center',
                                 fontsize=11, fontweight='bold',
                                 color='#333333')
                    else:
                        ax4.text(0.5, 0.5, 'Insufficient return data',
                                 ha='center', va='center',
                                 fontsize=13, fontweight='bold',
                                 color='#666666')
                else:
                    ax4.text(0.5, 0.5, 'Insufficient return data',
                             ha='center', va='center',
                             fontsize=13, fontweight='bold',
                             color='#666666')
            else:
                ax4.text(0.5, 0.5, 'Simulated data is empty',
                         ha='center', va='center',
                         fontsize=13, fontweight='bold',
                         color='#666666')

            ax4.set_title('Risk Analysis - Trading Day Type Distribution',
                          fontsize=13, fontweight='bold', pad=15, color='#333333')
            ax4.set_aspect('equal')

            # 调整子图间距
            plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)
            plt.subplots_adjust(top=0.92, bottom=0.1, left=0.1,
                                right=0.95, hspace=0.3, wspace=0.25)

            # 保存图片到桌面路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{self.config.stock_code}_{timestamp}.png'
            filepath = os.path.join(self.save_path, filename)

            # 保存图片
            plt.savefig(filepath, dpi=200, bbox_inches='tight',
                        facecolor='#f5f5f5', pad_inches=0.2)

            # 关闭图形
            plt.close(fig)

            return filepath

        except Exception as e:
            print(f"Failed to create visual report: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_error_report(self, error):
        """保存错误报告"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config.stock_code}_{self.config.stock_name}_错误报告_{timestamp}.txt"
            filepath = os.path.join(self.save_path, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"股票分析报告生成失败\n")
                f.write("=" * 48 + "\n")
                f.write(f"股票代码: {self.config.stock_code}\n")
                f.write(f"股票名称: {self.config.stock_name}\n")
                f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"错误信息: {str(error)}\n")
            print(f"错误报告已保存: {filepath}")
        except Exception as e:
            print(f"无法保存错误报告: {e}")


# 配置类示例
class ReportConfig:
    def __init__(self, stock_code, stock_name):
        self.stock_code = stock_code
        self.stock_name = stock_name


# 批量分析管理器
class BatchAnalysisManager:
    def __init__(self):
        self.save_path = SAVE_PATH
        os.makedirs(self.save_path, exist_ok=True)
        self.risk_summary_file = os.path.join(self.save_path, "风险等级汇总.xlsx")

    def run_batch_analysis(self, stock_list, get_data_func, get_models_func):
        """批量分析股票"""
        print("=" * 48)
        print("开始批量股票分析")
        print("=" * 48)

        all_results = []

        for stock_code, stock_name in stock_list:
            print(f"\n分析 {stock_name}({stock_code})...")

            # 创建配置
            config = ReportConfig(stock_code, stock_name)

            # 获取数据
            processed_data = get_data_func(stock_code, stock_name)
            if processed_data is None:
                print(f"  无法获取 {stock_name} 数据，跳过")
                continue

            # 获取模型
            models, best_model_name, simulated_data = get_models_func(processed_data)
            if models is None:
                print(f"  {stock_name} 模型生成失败，跳过")
                continue

            # 生成报告
            generator = ReportGenerator(config)
            result = generator.generate_report(processed_data, simulated_data, models, best_model_name)

            if result:
                all_results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'result': result
                })
                print(f"  {stock_name} 分析完成")
            else:
                print(f"  {stock_name} 分析失败")

        # 打印汇总信息
        print("\n" + "=" * 48)
        print(f"批量分析完成！")
        print(f"成功分析: {len(all_results)} 只股票")
        print(f"失败: {len(stock_list) - len(all_results)} 只股票")
        print(f"风险汇总表: {self.risk_summary_file}")
        print("=" * 48)

        return all_results


# 使用示例 - 批量分析模式
if __name__ == "__main__":
    # 示例股票列表
    stock_list = [
        ("000001", "平安银行"),
        ("000002", "万科A"),
        ("000063", "中兴通讯"),
        # 可以添加更多股票
    ]


    def example_get_data(stock_code, stock_name):
        """示例数据获取函数"""
        # 这里应该从数据库或API获取数据
        # 返回格式: {'processed': pd.DataFrame}
        return {
            'processed': pd.DataFrame({
                'open': [10, 11, 12, 13, 14, 13.5, 14.2, 13.8, 14.5, 15],
                'high': [12, 13, 14, 15, 16, 14.5, 15.2, 14.8, 15.5, 16],
                'low': [9, 10, 11, 12, 13, 12.5, 13.2, 12.8, 13.5, 14],
                'close': [11, 12, 13, 14, 15, 14.2, 14.8, 14.5, 15.2, 15.8],
                'volume': [100000, 110000, 120000, 130000, 140000, 135000, 142000, 138000, 145000, 150000]
            }, index=pd.date_range('2024-01-01', periods=10))
        }


    def example_get_models(processed_data):
        """示例模型获取函数"""
        # 这里应该生成模型
        simulated_data = pd.DataFrame({
            'open': [15.5, 15.8, 16.2, 15.8, 16.2, 16.5, 16.8, 16.3, 16.6, 16.9],
            'high': [15.8, 16.2, 16.5, 16.3, 16.8, 16.9, 17.2, 16.7, 17.0, 17.3],
            'low': [15.2, 15.5, 15.8, 15.3, 15.8, 16.1, 16.4, 15.9, 16.2, 16.5],
            'close': [15.8, 16.3, 16.1, 16.6, 16.9, 16.7, 17.1, 16.8, 17.2, 17.5],
            'volume': [150000, 160000, 170000, 165000, 180000, 175000, 185000, 180000, 190000, 195000]
        }, index=pd.date_range('2024-01-11', periods=10))

        return {'neural_network': '神经网络模型'}, "神经网络", simulated_data


    # 创建批量分析管理器
    batch_manager = BatchAnalysisManager()

    # 自动设置回答（模拟用户输入）
    print("修改批量导入股票模式下，分析完成后自动在浏览器中打开所有报告？(y/N): N")
    print("是否开始批量分析？(Y/n): Y")

    # 运行批量分析
    results = batch_manager.run_batch_analysis(stock_list, example_get_data, example_get_models)

    # 询问是否继续分析其他股票
    print("\n是否继续分析其他股票？(y/N): N")
    print("批量分析程序结束")