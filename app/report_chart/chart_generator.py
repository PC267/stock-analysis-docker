# chart_generator.py
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib
import platform
import sys
import pandas as pd  # 新增导入pandas用于处理Excel

# 添加report_config.py所在目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import report_config


def setup_chinese_font_chart():
    """设置中文字体（为图表生成器）"""
    system_name = platform.system()

    if system_name == "Windows":
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    elif system_name == "Darwin":  # macOS
        matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']
    else:  # Linux或其他系统
        matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']

    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.family'] = 'sans-serif'


# 初始化字体设置
setup_chinese_font_chart()


class ChartGenerator:
    def __init__(self, config):
        self.config = config

        # 修改保存路径
        desktop_path = report_config.REPORT_DIR
        self.config.result_dir = desktop_path
        os.makedirs(self.config.result_dir, exist_ok=True)

        print(f"图表将保存到: {self.config.result_dir}")

    def generate_all_charts(self, processed_data, simulated_data, best_model_name):
        """生成所有图表"""
        print("生成图表...")

        try:
            # 创建图形
            fig = plt.figure(figsize=(15, 10), facecolor='#f5f5f5')

            # 设置主标题，添加免责声明
            plt.suptitle(
                f'({self.config.stock_code}) - Analysis Report\n'
                f'(For Learning And Research Purposes Only. Other Uses Are At Your Own Risk.)\n',
                fontsize=14, fontweight='bold', y=1.02)

            # 准备数据
            historical_data = processed_data['processed'].iloc[-20:]
            simulated = simulated_data

            # 1. 价格走势图
            ax1 = plt.subplot(2, 2, 1)
            ax1.set_facecolor('white')

            # 历史价格
            ax1.plot(historical_data.index, historical_data['close'],
                     color='#2E7D32', linewidth=2, label='Historical Close')

            # 分析价格
            ax1.plot(simulated.index, simulated['close'],
                     color='#1565C0', linewidth=2, label='Simulated Close')

            # 分界线
            ax1.axvline(x=historical_data.index[-1], color='red',
                        linestyle='--', linewidth=1, alpha=0.7, label='Analysis Start')

            ax1.set_title('Price trend analysis', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Price (CNY)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

            # 2. K线图
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_facecolor('white')

            # 绘制分析K线
            for idx in range(len(simulated)):
                row = simulated.iloc[idx]
                color = '#F44336' if row['close'] >= row['open'] else '#4CAF50'  # 红涨绿跌

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

            ax2.set_title('Analyzing Candlestick charts', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Price (CNY)')
            ax2.grid(True, alpha=0.3)

            # 3. 成交量图
            ax3 = plt.subplot(2, 2, 3)
            ax3.set_facecolor('white')

            # 分析成交量
            colors = ['#2196F3' if simulated['close'].iloc[i] >= simulated['open'].iloc[i]
                      else '#1976D2' for i in range(len(simulated))]

            ax3.bar(range(len(simulated)), simulated['volume'].values,
                    color=colors, alpha=0.7, width=0.6)

            ax3.set_title('Trading volume analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Trading day')
            ax3.set_ylabel('Volume')
            ax3.grid(True, alpha=0.3, axis='y')

            # 4. 风险分析饼图 - 优化部分
            ax4 = plt.subplot(2, 2, 4)
            ax4.set_facecolor('white')

            # 风险分析
            returns = simulated['close'].pct_change().dropna()
            up_days = sum(returns > 0.01)
            down_days = sum(returns < -0.01)
            neutral_days = len(returns) - up_days - down_days

            risk_sizes = [up_days, down_days, neutral_days]
            risk_labels = ['Up_Days', 'Down_Days', 'Neutral_Days']
            risk_colors = ['#F44336', '#4CAF50', '#FFC107']

            # 确保有数据才显示饼图
            total_days = sum(risk_sizes)

            if total_days > 0:
                # 创建自定义的autopct函数，确保所有扇区都显示数据
                def make_autopct(values):
                    def my_autopct(pct):
                        # 根据实际数值计算
                        total = sum(values)
                        if total == 0:
                            return ""
                        # 获取当前扇区的索引
                        idx = int(pct * total / 100.0 / max(values) * len(values))  # 近似索引
                        idx = min(idx, len(values) - 1)
                        val = values[idx]

                        # 总是显示天数和百分比（即使为0%）
                        if val > 0:
                            return f'{pct:.1f}%\n({val}Day)'
                        else:
                            return '0.0%\n(0天)'

                    return my_autopct

                # 绘制饼图
                wedges, texts, autotexts = ax4.pie(
                    risk_sizes,
                    labels=risk_labels,
                    colors=risk_colors,
                    autopct=make_autopct(risk_sizes),  # 使用自定义的autopct函数
                    startangle=90,
                    textprops={'fontsize': 10}
                )

                # 设置百分比文本样式
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
            else:
                # 如果没有数据，显示提示
                ax4.text(0.5, 0.5, '无数据', ha='center', va='center', fontsize=12, fontweight='bold')
                wedges, texts, autotexts = ax4.pie([1], labels=[''], colors=['#CCCCCC'], startangle=90)

            ax4.set_title('Risk Analysis - Trading Day Type Distribution', fontsize=12, fontweight='bold')

            # 可选：添加图例
            if total_days > 0:
                legend_labels = []
                for label, size in zip(risk_labels, risk_sizes):
                    # 修改这里，即使size=0也显示
                    percentage = (size / total_days * 100) if total_days > 0 else 0
                    legend_labels.append(f'{label}: {size}Day ({percentage:.1f}%)')
                ax4.legend(wedges, legend_labels, title="Type Distribution", loc="center left",
                           bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

            plt.tight_layout()

            # 保存图表
            self._save_chart(fig)

            # 保存模拟K线数据到Excel
            self._save_simulated_data_to_excel(simulated_data, best_model_name)

        except Exception as e:
            print(f"图表生成失败: {e}")

    def _save_chart(self, fig):
        """保存图表到指定路径"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{self.config.stock_code}_{self.config.stock_name}_Analyze The Chart_{timestamp}.png"
        filepath = os.path.join(self.config.result_dir, filename)

        try:
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#f5f5f5')
            print(f"图表已保存到: {filepath}")
        except Exception as e:
            print(f"保存图表失败: {e}")
        finally:
            plt.close(fig)

    def _save_simulated_data_to_excel(self, simulated_data, best_model_name):
        """保存模拟K线数据到Excel表格"""
        try:
            # 准备要保存的数据
            simulated_df = simulated_data.copy()

            # 添加日期列（如果数据有日期索引）
            if simulated_df.index.name is None:
                simulated_df.index.name = 'date'

            # 重命名列以更清晰的显示
            column_mapping = {
                'open': '开盘价',
                'high': '最高价',
                'low': '最低价',
                'close': '收盘价',
                'volume': '成交量'
            }

            # 重命名列
            simulated_df = simulated_df.rename(columns=column_mapping)

            # 添加涨跌幅列
            simulated_df['涨跌幅(%)'] = simulated_df['收盘价'].pct_change() * 100

            # 添加涨跌标识列
            simulated_df['涨跌'] = simulated_df['收盘价'] >= simulated_df['开盘价']
            simulated_df['涨跌'] = simulated_df['涨跌'].map({True: '上涨', False: '下跌'})

            # 添加交易日序号
            simulated_df.insert(0, '交易日序号', range(1, len(simulated_df) + 1))

            # 创建Excel文件
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config.stock_code}_{self.config.stock_name}_模拟K线数据_{timestamp}.xlsx"
            filepath = os.path.join(self.config.result_dir, filename)

            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 写入模拟数据
                simulated_df.to_excel(writer, sheet_name='模拟K线数据', index=True)

                # 写入模型信息
                model_info = pd.DataFrame({
                    '字段': [
                        '股票代码',
                        '股票名称',
                        '最佳模型',
                        '模拟天数',
                        '生成时间',
                        '数据来源'
                    ],
                    '值': [
                        self.config.stock_code,
                        self.config.stock_name,
                        best_model_name,
                        len(simulated_df),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'AI模型模拟生成'
                    ]
                })
                model_info.to_excel(writer, sheet_name='模型信息', index=False)

                # 写入统计数据
                stats_df = pd.DataFrame({
                    '统计指标': [
                        '平均开盘价',
                        '平均最高价',
                        '平均最低价',
                        '平均收盘价',
                        '平均成交量',
                        '最大涨幅',
                        '最大跌幅',
                        '上涨天数',
                        '下跌天数',
                        '平盘天数'
                    ],
                    '数值': [
                        simulated_df['开盘价'].mean(),
                        simulated_df['最高价'].mean(),
                        simulated_df['最低价'].mean(),
                        simulated_df['收盘价'].mean(),
                        simulated_df['成交量'].mean(),
                        simulated_df['涨跌幅(%)'].max(),
                        simulated_df['涨跌幅(%)'].min(),
                        (simulated_df['涨跌'] == '上涨').sum(),
                        (simulated_df['涨跌'] == '下跌').sum(),
                        (simulated_df['涨跌幅(%)'].abs() < 0.01).sum()
                    ]
                })
                stats_df.to_excel(writer, sheet_name='统计数据', index=False)

                # 调整列宽
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width

            print(f"模拟K线数据已保存到Excel文件: {filepath}")
            print(f"包含 {len(simulated_df)} 天的模拟数据")

        except Exception as e:
            print(f"保存模拟数据到Excel失败: {e}")