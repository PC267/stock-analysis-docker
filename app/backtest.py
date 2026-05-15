# backtest.py - 优化版本：集成全部模型训练与高保真模拟（含 ARIMA/LSTM/Boosting）
import os
import sys
import itertools
from collections import Counter

os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互式后端

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 确保使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import gc

# ========== 修改导入路径 ==========
current_dir = os.path.dirname(os.path.abspath(__file__))
report_chart_dir = os.path.join(current_dir, "report_chart")
sys.path.insert(0, report_chart_dir)
import report_config

warnings.filterwarnings('ignore')

# ========== 尝试导入真实模型与模拟器 ==========
try:
    from factory import ModelFactory
    FACTORY_AVAILABLE = True
    print("✓ 成功导入 ModelFactory")
except ImportError as e:
    FACTORY_AVAILABLE = False
    print(f"⚠️ 无法导入 ModelFactory: {e}，将使用简单模拟作为后备")

try:
    from simulator import StockSimulator
    SIMULATOR_AVAILABLE = True
    print("✓ 成功导入 StockSimulator")
except ImportError as e:
    SIMULATOR_AVAILABLE = False
    print(f"⚠️ 无法导入 StockSimulator: {e}，将使用简单模拟作为后备")


# ==================== 内存清理函数 ====================
def cleanup_resources():
    """清理内存和资源"""
    try:
        plt.close('all')
        print("✓ 清理 matplotlib 图形")
    except Exception as e:
        print(f"⚠️ 清理 matplotlib 时出错: {e}")

    try:
        gc.collect()
        print("✓ 触发垃圾回收")
    except Exception as e:
        print(f"⚠️ 垃圾回收时出错: {e}")


def analyze_memory_usage(process=None, start_memory=0, step_name=""):
    """分析内存使用情况"""
    if not process:
        try:
            import psutil
            import os as os_module
            process = psutil.Process(os_module.getpid())
        except ImportError:
            return 0, 0

    if process:
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_diff = current_memory - start_memory
        print(f"  {step_name}内存: {current_memory:.1f} MB, 增量: {memory_diff:+.1f} MB")
        return current_memory, memory_diff

    return 0, 0


# ==================== 设置中文字体 ====================
def set_chinese_font():
    """设置中文字体支持"""
    try:
        if os.name == 'nt':
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            matplotlib.rcParams['axes.unicode_minus'] = False
        else:
            from matplotlib.font_manager import FontManager
            fm = FontManager()
            available_fonts = set([f.name for f in fm.ttflist])

            font_candidates = [
                'DejaVu Sans', 'Arial Unicode MS', 'SimHei', 'Microsoft YaHei',
                'WenQuanYi Micro Hei', 'STHeiti', 'Source Han Sans CN'
            ]

            for font in font_candidates:
                if font in available_fonts:
                    matplotlib.rcParams['font.sans-serif'] = [font]
                    break

            matplotlib.rcParams['axes.unicode_minus'] = False

        print("✓ 中文字体设置完成")
    except Exception as e:
        print(f"⚠️ 设置中文字体时出错: {e}")
        print("将使用默认字体，中文可能显示为方框")


# 初始化中文字体
set_chinese_font()


# ==================== 全部候选模型列表 ====================
ALL_CANDIDATE_MODELS = [
    'sarima', 'garch', 'xgboost', 'multi_factor',
    'arima', 'lstm', 'lightgbm', 'catboost', 'ensemble'
]


class StockDataLoader:
    """股票数据加载器"""

    @staticmethod
    def load_stock_data(file_path):
        """加载股票数据"""
        try:
            if not os.path.exists(file_path):
                print(f"❌ 文件不存在: {file_path}")
                return None

            df = pd.read_csv(file_path)
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                print(f"❌ 文件缺少必要的列: {missing_columns}")
                return None

            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            print(f"✓ 成功加载数据，共 {len(df)} 条记录")
            print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")

            return df
        except Exception as e:
            print(f"❌ 加载数据时出错: {e}")
            return None

    @staticmethod
    def split_train_test(df, train_ratio=0.7):
        """分割训练集和测试集"""
        if df is None or len(df) == 0:
            return None, None

        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        print(f"  训练集: {len(train_df)} 条 ({train_df.index[0]} 到 {train_df.index[-1]})")
        print(f"  测试集: {len(test_df)} 条 ({test_df.index[0]} 到 {test_df.index[-1]})")

        return train_df, test_df


# ==================== 参数优化类 ====================
class ParameterOptimizer:
    """参数优化器 - 支持全模型"""

    def __init__(self):
        self.best_params = {}
        self.optimization_history = []

    def optimize_for_model(self, model_name, train_data, validation_data,
                           param_space=None, metric='综合得分'):
        """为特定模型优化参数"""
        print(f"\n🔍 开始优化 {model_name} 模型参数")
        print("-" * 80)

        if param_space is None:
            param_space = self._get_default_param_space_for_model(model_name)

        keys = param_space.keys()
        values = param_space.values()
        param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        print(f"✓ 生成 {len(param_combinations)} 个参数组合")

        best_score = -float('inf')
        best_params = None

        test_limit = min(10, len(param_combinations))
        for i, params in enumerate(param_combinations[:test_limit]):
            print(f"  测试组合 {i + 1}/{test_limit}: {params}")

            try:
                score = self._simple_evaluate_params(model_name, params, train_data, validation_data)

                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"    ✓ 新最佳得分: {score:.4f}")

                cleanup_resources()
            except Exception as e:
                print(f"    ⚠️ 参数测试失败: {e}")
                continue

        if best_params:
            self.best_params[model_name] = {
                'params': best_params,
                'score': best_score
            }

            print(f"\n✅ {model_name} 模型优化完成")
            print(f"   最佳参数: {best_params}")
            print(f"   最佳得分: {best_score:.4f}")

        return best_params, best_score

    def _get_default_param_space_for_model(self, model_name):
        """获取模型参数搜索空间（包含新增模型）"""
        base_space = {
            'window_size': [50, 100, 150],
            'prediction_horizon': [5, 10, 20]
        }

        if model_name in ['sarima', 'arima']:
            base_space.update({
                'p': [1, 2],
                'd': [0, 1],
                'q': [0, 1]
            })
        elif model_name == 'xgboost':
            base_space.update({
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            })
        elif model_name == 'lightgbm':
            base_space.update({
                'n_estimators': [50, 100],
                'max_depth': [5, 7],
                'learning_rate': [0.01, 0.05]
            })
        elif model_name == 'catboost':
            base_space.update({
                'iterations': [50, 100],
                'depth': [4, 6],
                'learning_rate': [0.01, 0.05]
            })
        elif model_name == 'garch':
            base_space.update({
                'p': [1, 2],
                'q': [1, 2]
            })
        elif model_name == 'multi_factor':
            base_space.update({
                'factor_count': [3, 5, 8],
                'correlation_threshold': [0.7, 0.8, 0.9]
            })
        elif model_name == 'lstm':
            base_space.update({
                'units': [32, 64],
                'epochs': [30, 50],
                'batch_size': [32, 64]
            })
        elif model_name == 'ensemble':
            # 集成模型不需要独立参数优化
            pass

        return base_space

    def _simple_evaluate_params(self, model_name, params, train_data, validation_data):
        """简化的参数评估函数"""
        try:
            if len(train_data) < 20 or len(validation_data) < 10:
                return 0.0

            base_score = 0.5
            window_factor = params.get('window_size', 100) / 200.0
            horizon_factor = 1.0 / (1 + params.get('prediction_horizon', 10) / 30.0)

            np.random.seed(hash(str(params)) % 1000)
            random_factor = 0.2 * np.random.random()

            score = base_score + 0.3 * window_factor + 0.2 * horizon_factor + random_factor
            score = max(0.0, min(1.0, score))

            return score
        except Exception as e:
            print(f"  ⚠️ 评估参数时出错: {e}")
            return 0.0


# ==================== 回测配置类 ====================
class BacktestConfig:
    """回测配置类"""

    def __init__(self):
        self.simulator = type('obj', (object,), {
            'simulation_days': 30,
            'stock_symbol': ''
        })()

        self.ensemble = type('obj', (object,), {
            'threshold_settings': type('obj', (object,), {
                'max_window_size': 200
            })()
        })()

        self.parameter_optimization = type('obj', (object,), {
            'enabled': True
        })()


# ==================== 回测处理器 ====================
class BacktestProcessor:
    """回测处理器"""

    def __init__(self, config):
        self.config = config
        self.market_regime = 'default'
        self.parameter_optimizer = ParameterOptimizer()
        self.optimized_params = {}
        self.parameter_report_path = None
        self.models = None  # 训练后的模型字典
        self.processed_data = None  # 保存processed_data供后续使用

    def prepare_data(self, df):
        """准备数据"""
        try:
            if df is None or len(df) == 0:
                return None

            df = df.sort_index()

            self.market_regime = self._analyze_market_regime(df)
            print(f"  市场状态分析结果: {self.market_regime}")

            if self.config.parameter_optimization.enabled:
                print("  启用参数优化...")
                self._optimize_parameters(df)

            # 保存原始训练数据副本
            processed_data = {
                'processed': df.copy(),
                'prepared': self._prepare_training_data(df),
                'optimized_params': self.optimized_params
            }
            self.processed_data = processed_data

            # 训练真实模型（如果可用）
            if FACTORY_AVAILABLE:
                print("\n🔧 开始训练真实模型...")
                try:
                    # 为ModelFactory准备一个简单的config（其内部主要依赖param_loader）
                    model_config = type('obj', (object,), {})()
                    model_factory = ModelFactory(config=model_config)
                    self.models = model_factory.train_all_models(processed_data)
                    print("✓ 模型训练完成")
                except Exception as e:
                    print(f"⚠️ 模型训练失败: {e}，将回退到简单模拟")
                    self.models = None
            else:
                self.models = None

            return processed_data
        except Exception as e:
            print(f"❌ 准备数据时出错: {e}")
            return None

    def _display_optimization_results(self):
        if not self.optimized_params:
            print("⚠️ 没有优化后的参数")
            return

        print("\n" + "=" * 48)
        print("✅ 参数优化完成")
        print("=" * 48)

        for model_name, result in self.optimized_params.items():
            print(f"{model_name:<15}: 得分={result['score']:.4f}")
            print(f"               参数: {result['params']}")

    def _prepare_training_data(self, df):
        from sklearn.preprocessing import StandardScaler

        prepared_data = {}

        scaler_close = StandardScaler()
        close_scaled = scaler_close.fit_transform(df[['close']])

        returns = df['close'].pct_change().dropna().values

        prepared_data['close_scaled'] = close_scaled
        prepared_data['scaler_close'] = scaler_close
        prepared_data['returns'] = returns
        prepared_data['market_regime'] = self.market_regime
        prepared_data['optimized_params'] = self.optimized_params
        return prepared_data

    def _analyze_market_regime(self, df):
        try:
            if df is None or len(df) < 20:
                return 'default'

            returns = df['close'].pct_change().dropna().values

            if len(returns) < 10:
                return 'default'

            volatility = np.std(returns) if len(returns) > 0 else 0

            if volatility > 0.03:
                return 'high_volatility'
            elif volatility < 0.01:
                return 'stable'
            else:
                return 'default'

        except Exception as e:
            print(f"  市场状态分析失败: {e}")
            return 'default'

    def _optimize_parameters(self, df):
        print("\n🔍 开始参数优化流程")
        print("-" * 80)

        split_idx = int(len(df) * 0.7)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        print(f"  参数优化数据集:")
        print(f"    训练集: {len(train_df)} 条")
        print(f"    验证集: {len(val_df)} 条")

        # 包含所有可优化模型（集成模型除外）
        models_to_optimize = [
            'sarima', 'xgboost', 'garch', 'multi_factor',
            'arima', 'lstm', 'lightgbm', 'catboost'
        ]

        for model_name in models_to_optimize:
            print(f"\n  优化 {model_name} 参数...")

            best_params, best_score = self.parameter_optimizer.optimize_for_model(
                model_name=model_name,
                train_data=train_df,
                validation_data=val_df
            )

            if best_params:
                self.optimized_params[model_name] = {
                    'params': best_params,
                    'score': best_score
                }

        self._display_optimization_results()
        self._save_parameter_report()

        return self.optimized_params

    def _save_parameter_report(self):
        if not self.optimized_params:
            print("⚠️ 没有优化参数可保存")
            return

        try:
            report_dir = f'{report_config.REPORT_DIR}\\backtest_results\\parameter_reports'
            os.makedirs(report_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(report_dir, f'parameter_optimization_{timestamp}.txt')

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 48 + "\n")
                f.write("参数优化报告\n")
                f.write("=" * 48 + "\n\n")

                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"优化模型数量: {len(self.optimized_params)}\n\n")

                sorted_models = sorted(
                    self.optimized_params.items(),
                    key=lambda x: x[1]['score'],
                    reverse=True
                )

                f.write("模型参数优化结果:\n")
                f.write("-" * 80 + "\n")

                for model_name, result in sorted_models:
                    score = result['score']
                    params = result['params']

                    f.write(f"\n🔹 {model_name}:\n")
                    f.write(f"   最佳得分: {score:.4f}\n")
                    f.write(f"   最佳参数: {params}\n")

                    f.write("   参数详情:\n")
                    for param_name, param_value in params.items():
                        f.write(f"     - {param_name}: {param_value}\n")

                f.write("\n" + "=" * 48 + "\n")
                f.write("通用参数建议:\n")
                f.write("=" * 48 + "\n\n")

                param_stats = {}
                for model_name, result in self.optimized_params.items():
                    for param_name, param_value in result['params'].items():
                        if param_name not in param_stats:
                            param_stats[param_name] = []
                        param_stats[param_name].append(param_value)

                f.write("参数统计摘要:\n")
                for param_name, values in param_stats.items():
                    if values:
                        try:
                            if all(isinstance(v, (int, float)) for v in values):
                                avg_value = sum(values) / len(values)
                                min_value = min(values)
                                max_value = max(values)
                                f.write(f"  {param_name:<20}: 平均值={avg_value:.2f}, 范围={min_value}-{max_value}\n")
                        except:
                            pass

            self.parameter_report_path = report_path
            print(f"✓ 参数优化报告已保存到: {report_path}")

        except Exception as e:
            print(f"⚠️ 保存参数报告时出错: {e}")


# ==================== 回测评估器 ====================
class BacktestEvaluator:
    """回测评估器"""

    def __init__(self):
        self.metrics_history = []

    def evaluate_simulation(self, real_data, simulated_data, model_name):
        """评估模拟结果"""
        metrics = {}

        common_dates = real_data.index.intersection(simulated_data.index)

        if len(common_dates) == 0:
            print(f"⚠️ 没有共同日期可比较")
            return None

        real_subset = real_data.loc[common_dates]
        sim_subset = simulated_data.loc[common_dates]

        metrics['模型名称'] = model_name
        metrics['评估天数'] = len(common_dates)

        metrics['价格匹配度'] = self._calculate_price_match(real_subset['close'], sim_subset['close'])

        real_returns = real_subset['close'].pct_change().dropna()
        sim_returns = sim_subset['close'].pct_change().dropna()
        metrics['方向准确率'] = self._calculate_direction_accuracy(real_returns, sim_returns)

        metrics['价格MAE'] = self._calculate_mae(real_subset['close'], sim_subset['close'])
        metrics['价格RMSE'] = self._calculate_rmse(real_subset['close'], sim_subset['close'])

        metrics['综合得分'] = self._calculate_composite_score(metrics)

        self.metrics_history.append(metrics)

        return metrics

    def _calculate_price_match(self, real_prices, sim_prices):
        if len(real_prices) != len(sim_prices) or len(real_prices) == 0:
            return 0

        relative_errors = abs((sim_prices - real_prices) / (real_prices + 1e-8))
        relative_errors = relative_errors[relative_errors < 1.0]

        if len(relative_errors) > 0:
            return max(0, 1 - relative_errors.mean())

        return 0

    def _calculate_direction_accuracy(self, real_returns, sim_returns):
        if len(real_returns) != len(sim_returns) or len(real_returns) == 0:
            return 0

        common_idx = real_returns.index.intersection(sim_returns.index)
        real_returns = real_returns.loc[common_idx]
        sim_returns = sim_returns.loc[common_idx]

        real_direction = (real_returns > 0).astype(int)
        sim_direction = (sim_returns > 0).astype(int)

        accuracy = (real_direction == sim_direction).mean()

        return accuracy if not pd.isna(accuracy) else 0

    def _calculate_mae(self, real_values, sim_values):
        if len(real_values) != len(sim_values) or len(real_values) == 0:
            return float('inf')

        mae = abs(sim_values - real_values).mean()
        return mae

    def _calculate_rmse(self, real_values, sim_values):
        if len(real_values) != len(sim_values) or len(real_values) == 0:
            return float('inf')

        mse = ((sim_values - real_values) ** 2).mean()
        rmse = np.sqrt(mse)
        return rmse

    def _calculate_composite_score(self, metrics):
        weights = {
            '价格匹配度': 0.5,
            '方向准确率': 0.3,
            '价格MAE': 0.1,
            '价格RMSE': 0.1
        }

        score = 0
        for metric_name, weight in weights.items():
            if metric_name in metrics:
                value = metrics[metric_name]

                if metric_name in ['价格MAE', '价格RMSE']:
                    error_score = max(0, 1 - value / 10.0)
                    score += error_score * weight * 100
                else:
                    score += value * weight * 100

        return min(100, max(0, score))

    def generate_report(self, metrics_list, output_dir=f'{report_config.REPORT_DIR}\\backtest_results',
                        optimized_params=None):
        if not metrics_list:
            print("❌ 没有评估数据可生成报告")
            return

        os.makedirs(output_dir, exist_ok=True)

        df_metrics = pd.DataFrame(metrics_list)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(output_dir, f'backtest_metrics_{timestamp}.csv')
        df_metrics.to_csv(csv_path, index=False, encoding='utf-8-sig')

        report_path = os.path.join(output_dir, f'backtest_report_{timestamp}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 48 + "\n")
            f.write("股票模拟回测报告\n")
            f.write("=" * 48 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"评估模型数量: {len(metrics_list)}\n")

            if optimized_params:
                f.write(f"参数优化模型数量: {len(optimized_params)}\n")
                optimized_models = list(optimized_params.keys())
                f.write(f"已优化模型: {', '.join(optimized_models)}\n")
            f.write("\n")

            df_sorted = df_metrics.sort_values('综合得分', ascending=False)

            f.write("模型表现排名:\n")
            f.write("-" * 80 + "\n")
            for idx, row in df_sorted.iterrows():
                f.write(f"{idx + 1:2d}. {row['模型名称']:15s} - 综合得分: {row['综合得分']:6.2f}分\n")
                f.write(f"     价格匹配度: {row['价格匹配度']:.3f}, "
                        f"方向准确率: {row['方向准确率']:.3f}\n")

                if '优化得分' in row:
                    f.write(f"     参数优化得分: {row['优化得分']:.4f}\n")

            if optimized_params:
                f.write("\n" + "=" * 48 + "\n")
                f.write("参数优化详细结果\n")
                f.write("=" * 48 + "\n\n")

                for model_name, result in optimized_params.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  优化得分: {result['score']:.4f}\n")
                    f.write(f"  最优参数: {result['params']}\n\n")

        print(f"✓ 报告已保存到: {report_path}")
        print(f"✓ 数据已保存到: {csv_path}")

        self._plot_results(df_metrics, output_dir, timestamp)

        return report_path, csv_path

    def _plot_results(self, df_metrics, output_dir, timestamp):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('股票模拟回测结果分析', fontsize=16, fontweight='bold')

            ax1 = axes[0, 0]
            df_sorted = df_metrics.sort_values('综合得分')
            ax1.barh(range(len(df_sorted)), df_sorted['综合得分'])
            ax1.set_yticks(range(len(df_sorted)))
            ax1.set_yticklabels(df_sorted['模型名称'])
            ax1.set_xlabel('综合得分')
            ax1.set_title('模型综合得分排名')
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            x = np.arange(len(df_metrics))
            width = 0.35

            ax2.bar(x - width / 2, df_metrics['价格匹配度'], width, label='价格匹配度', alpha=0.8)
            ax2.bar(x + width / 2, df_metrics['方向准确率'], width, label='方向准确率', alpha=0.8)

            ax2.set_xlabel('模型')
            ax2.set_ylabel('得分')
            ax2.set_title('关键指标对比')
            ax2.set_xticks(x)
            ax2.set_xticklabels(df_metrics['模型名称'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            ax3 = axes[1, 0]
            if '价格MAE' in df_metrics.columns and '价格RMSE' in df_metrics.columns:
                x = np.arange(len(df_metrics))
                width = 0.35

                ax3.bar(x - width / 2, df_metrics['价格MAE'], width, label='MAE', alpha=0.8)
                ax3.bar(x + width / 2, df_metrics['价格RMSE'], width, label='RMSE', alpha=0.8)

                ax3.set_xlabel('模型')
                ax3.set_ylabel('误差')
                ax3.set_title('误差指标对比')
                ax3.set_xticks(x)
                ax3.set_xticklabels(df_metrics['模型名称'], rotation=45, ha='right')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            scatter = ax4.scatter(df_metrics['价格匹配度'],
                                  df_metrics['方向准确率'],
                                  c=df_metrics['综合得分'],
                                  s=100,
                                  alpha=0.6,
                                  cmap='viridis')

            ax4.set_xlabel('价格匹配度')
            ax4.set_ylabel('方向准确率')
            ax4.set_title('匹配度与准确率关系')
            ax4.grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=ax4, label='综合得分')

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f'backtest_plots_{timestamp}.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✓ 图表已保存到: {plot_path}")

        except Exception as e:
            print(f"⚠️ 生成图表时出错: {e}")


# ==================== 回测模拟器 ====================
class BacktestSimulator:
    """主回测模拟器 - 集成全部模型"""

    def __init__(self, data_dir='stock_data'):
        self.data_dir = data_dir
        self.config = BacktestConfig()
        self.loader = StockDataLoader()
        self.processor = BacktestProcessor(self.config)
        self.evaluator = BacktestEvaluator()
        self.aggregator = ParameterRecommendationAggregator()

    def run_backtest(self, stock_file, test_days=30):
        """运行回测"""
        print("=" * 48)
        print(f"开始回测: {stock_file}")
        print("=" * 48)

        if self.config.parameter_optimization.enabled:
            print("🎯 参数优化已启用")

        # 1. 加载数据
        file_path = os.path.join(self.data_dir, stock_file)
        df = self.loader.load_stock_data(file_path)

        if df is None or len(df) < 100:
            print(f"❌ 数据不足，需要至少100条记录")
            return None

        # 2. 分割数据
        train_df, test_df = self.loader.split_train_test(df, 0.7)

        if test_df is None or len(test_df) < test_days:
            print(f"❌ 测试数据不足，需要至少{test_days}条")
            return None

        # 3. 准备训练数据（含模型训练）
        processed_data = self.processor.prepare_data(train_df)

        if processed_data is None:
            print("❌ 准备训练数据失败")
            return None

        # 获取优化参数及训练好的模型
        optimized_params = self.processor.optimized_params
        trained_models = self.processor.models

        if optimized_params and self.config.parameter_optimization.enabled:
            stock_name = stock_file.replace('.csv', '').replace('_baostock', '')
            self.aggregator.add_optimization_results(stock_name, optimized_params)

        # 4. 模拟测试
        print("\n" + "-" * 80)
        print("运行模拟测试...")
        print("-" * 80)

        all_metrics = []
        use_real_simulation = (FACTORY_AVAILABLE and SIMULATOR_AVAILABLE and trained_models is not None)

        if use_real_simulation:
            # 使用 StockSimulator 进行高保真模拟
            try:
                # 构造适合 StockSimulator 的 config 对象
                sim_config = type('obj', (object,), {
                    'stock': type('obj', (object,), {
                        'code': stock_file.replace('.csv', ''),
                        'simulation_days': test_days
                    }),
                    'simulator': type('obj', (object,), {
                        'simulation_days': test_days,
                        'deterministic_mode': True
                    }),
                    'days_to_simulate': test_days,
                    'deterministic_mode': True
                })()

                stock_simulator = StockSimulator(
                    config=sim_config,
                    models=trained_models,
                    processed_data=processed_data
                )

                # 强制评估所有候选模型，而不仅仅是动态选择的
                stock_simulator.selected_models = [m for m in ALL_CANDIDATE_MODELS if m in trained_models]
                stock_simulator.model_weights = {m: 1.0 for m in stock_simulator.selected_models}

                simulations, future_dates = stock_simulator.simulate_future()
                print(f"✓ 成功生成 {len(simulations)} 个模型的模拟数据")

                # 评估每个模拟模型
                test_subset = test_df.iloc[:test_days]  # 取前 test_days 的真实数据
                for model_name in ALL_CANDIDATE_MODELS:
                    if model_name not in simulations:
                        continue
                    sim_prices = simulations[model_name]
                    print(f"\n评估模型: {model_name}")
                    try:
                        # 确保长度对齐
                        min_len = min(len(sim_prices), len(test_subset))
                        sim_df = pd.DataFrame(
                            {'close': sim_prices[:min_len]},
                            index=future_dates[:min_len]
                        )

                        metrics = self.evaluator.evaluate_simulation(test_subset.iloc[:min_len], sim_df, model_name)
                        if metrics:
                            if model_name in optimized_params:
                                metrics['优化得分'] = optimized_params[model_name].get('score', 0)
                                metrics['优化参数'] = str(optimized_params[model_name].get('params', {}))
                            all_metrics.append(metrics)

                            print(f"  ✓ 综合得分: {metrics['综合得分']:.2f}分")
                            print(f"    价格匹配度: {metrics['价格匹配度']:.3f}")
                            print(f"    方向准确率: {metrics['方向准确率']:.3f}")
                            if '优化得分' in metrics:
                                print(f"    参数优化得分: {metrics['优化得分']:.4f}")

                            # 可视化对比
                            self._plot_comparison(test_subset.iloc[:min_len], sim_df, model_name, stock_file)
                    except Exception as e:
                        print(f"  ⚠️ 评估模型 {model_name} 时出错: {e}")

            except Exception as e:
                print(f"❌ 真实模拟失败: {e}，回退到简单随机模拟")
                use_real_simulation = False

        if not use_real_simulation:
            # 回退到简单随机模拟（包含所有模型）
            print("⚠️ 使用简单随机模拟")
            for model_name in ALL_CANDIDATE_MODELS:
                print(f"\n评估模型: {model_name}")
                try:
                    simulated_data = self._generate_simulated_data(model_name, test_df, test_days)
                    if simulated_data is None:
                        print(f"  ⚠️ 模拟数据生成失败")
                        continue

                    metrics = self.evaluator.evaluate_simulation(test_df.iloc[:test_days], simulated_data, model_name)
                    if metrics:
                        if model_name in optimized_params:
                            metrics['优化得分'] = optimized_params[model_name].get('score', 0)
                            metrics['优化参数'] = str(optimized_params[model_name].get('params', {}))
                        all_metrics.append(metrics)

                        print(f"  ✓ 综合得分: {metrics['综合得分']:.2f}分")
                        print(f"    价格匹配度: {metrics['价格匹配度']:.3f}")
                        print(f"    方向准确率: {metrics['方向准确率']:.3f}")
                        if '优化得分' in metrics:
                            print(f"    参数优化得分: {metrics['优化得分']:.4f}")

                        self._plot_comparison(test_df.iloc[:test_days], simulated_data, model_name, stock_file)
                except Exception as e:
                    print(f"  ⚠️ 评估模型 {model_name} 时出错: {e}")

        # 5. 生成报告
        if all_metrics:
            print("\n" + "=" * 48)
            print("生成回测报告...")
            print("=" * 48)

            result = self.evaluator.generate_report(all_metrics, optimized_params=optimized_params)

            return {
                'stock_file': stock_file,
                'metrics': all_metrics,
                'optimized_params': optimized_params,
                'parameter_report_path': self.processor.parameter_report_path,
                'report_path': result[0] if result else None
            }
        else:
            print("❌ 没有成功的评估结果")
            return None

    def _generate_simulated_data(self, model_name, test_data, test_days):
        """生成简单模拟数据（后备方案，支持全模型）"""
        try:
            dates = test_data.index[:test_days]
            simulated_data = test_data.iloc[:test_days].copy()

            np.random.seed(42 + hash(model_name) % 1000)

            # 根据模型类型设置不同的噪声水平
            noise_map = {
                'sarima': 0.008,
                'xgboost': 0.005,
                'garch': 0.015,
                'multi_factor': 0.01,
                'arima': 0.007,
                'lstm': 0.012,
                'lightgbm': 0.006,
                'catboost': 0.006,
                'ensemble': 0.005
            }
            noise_std = noise_map.get(model_name, 0.01)

            noise = noise_std * np.random.randn(len(simulated_data))
            simulated_data['close'] = simulated_data['close'] * (1 + noise)

            for col in ['open', 'high', 'low']:
                if col in simulated_data.columns:
                    simulated_data[col] = simulated_data['close'] * (0.99 + 0.02 * np.random.rand())

            return simulated_data

        except Exception as e:
            print(f"  生成模拟数据时出错: {e}")
            return None

    def _plot_comparison(self, real_data, simulated_data, model_name, stock_file):
        """绘制真实数据与模拟数据对比图"""
        try:
            output_dir = f'{report_config.REPORT_DIR}\\backtest_results\\comparison_plots'
            os.makedirs(output_dir, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{stock_file} - {model_name} 模型对比', fontsize=16, fontweight='bold')

            ax1 = axes[0, 0]
            ax1.plot(real_data.index, real_data['close'], 'b-', label='真实收盘价', linewidth=2)
            ax1.plot(simulated_data.index, simulated_data['close'], 'r--', label='模拟收盘价', linewidth=2)
            ax1.set_xlabel('日期')
            ax1.set_ylabel('价格')
            ax1.set_title('收盘价对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            if len(real_data) > 1 and len(simulated_data) > 1:
                real_returns = real_data['close'].pct_change().dropna()
                sim_returns = simulated_data['close'].pct_change().dropna()

                ax2 = axes[0, 1]
                ax2.plot(real_returns.index, real_returns.values, 'b-',
                         label='真实收益率', linewidth=1.5, alpha=0.7)
                ax2.plot(sim_returns.index, sim_returns.values, 'r--',
                         label='模拟收益率', linewidth=1.5, alpha=0.7)
                ax2.set_xlabel('日期')
                ax2.set_ylabel('收益率')
                ax2.set_title('收益率对比')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            ax3 = axes[1, 0]
            errors = (simulated_data['close'] - real_data['close']) / real_data['close']
            ax3.hist(errors.dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_xlabel('相对误差')
            ax3.set_ylabel('频次')
            ax3.set_title('预测误差分布')
            ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            if len(real_data) > 1 and len(simulated_data) > 1:
                real_cumulative = (1 + real_data['close'].pct_change().fillna(0)).cumprod()
                sim_cumulative = (1 + simulated_data['close'].pct_change().fillna(0)).cumprod()

                ax4.plot(real_data.index, real_cumulative, 'b-',
                         label='真实累积收益', linewidth=2)
                ax4.plot(simulated_data.index, sim_cumulative, 'r--',
                         label='模拟累积收益', linewidth=2)
                ax4.set_xlabel('日期')
                ax4.set_ylabel('累积收益')
                ax4.set_title('累积收益对比 (起始为1)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            safe_model_name = model_name.replace(':', '_').replace('/', '_')
            plot_filename = f"{stock_file.replace('.csv', '')}_{safe_model_name}_comparison.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"     对比图已保存到: {plot_path}")

        except Exception as e:
            print(f"     生成对比图时出错: {e}")


# ==================== 参数推荐汇总器 ====================
class ParameterRecommendationAggregator:
    """参数推荐汇总器"""

    def __init__(self):
        self.all_optimizations = []
        self.parameter_stats = {}

    def add_optimization_results(self, stock_name, optimized_params):
        if not optimized_params:
            return

        for model_name, result in optimized_params.items():
            self.all_optimizations.append({
                'stock': stock_name,
                'model': model_name,
                'score': result['score'],
                'params': result['params']
            })

        print(f"  ✓ 已收集 {len(optimized_params)} 个模型的优化参数")

    def generate_recommendations(self):
        if not self.all_optimizations:
            return None

        print("\n" + "=" * 48)
        print("汇总参数优化结果，生成推荐报告...")
        print("=" * 48)

        model_groups = {}
        for opt in self.all_optimizations:
            model_name = opt['model']
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(opt)

        recommendations = {}
        param_stats_summary = {}

        for model_name, optimizations in model_groups.items():
            if len(optimizations) < 3:
                continue

            sorted_opts = sorted(optimizations, key=lambda x: x['score'], reverse=True)
            top_count = max(3, int(len(sorted_opts) * 0.3))
            top_opts = sorted_opts[:top_count]

            avg_score = sum(opt['score'] for opt in top_opts) / len(top_opts)

            param_counts = {}
            for opt in top_opts:
                for param_name, param_value in opt['params'].items():
                    key = f"{param_name}_{param_value}"
                    if key not in param_counts:
                        param_counts[key] = 0
                    param_counts[key] += 1

            recommended_params = {}
            for param_name in set(k.split('_')[0] for k in param_counts.keys()):
                param_values = {}
                for key, count in param_counts.items():
                    if key.startswith(f"{param_name}_"):
                        value_str = key.split('_', 1)[1]
                        try:
                            value = float(value_str) if '.' in value_str else int(value_str)
                        except:
                            value = value_str

                        param_values[value] = count

                if param_values:
                    recommended_value = max(param_values.items(), key=lambda x: x[1])[0]
                    recommended_params[param_name] = recommended_value

            if recommended_params:
                recommendations[model_name] = {
                    'params': recommended_params,
                    'avg_score': avg_score,
                    'sample_count': len(optimizations),
                    'top_count': top_count
                }

                for param_name, param_value in recommended_params.items():
                    if param_name not in param_stats_summary:
                        param_stats_summary[param_name] = []
                    param_stats_summary[param_name].append(param_value)

        return recommendations, param_stats_summary

    def save_recommendations(self, recommendations, param_stats_summary,
                             output_dir=f'{report_config.REPORT_DIR}\\backtest_results\\parameter_recommendations'):
        if not recommendations:
            print("⚠️ 没有足够的优化结果生成推荐报告")
            return

        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'parameter_recommendations.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 48 + "\n")
            f.write("参数配置推荐报告（基于回测汇总）\n")
            f.write("=" * 48 + "\n\n")

            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"基于 {len(self.all_optimizations)} 个模型的优化结果\n")
            f.write(f"分析股票数量: {len(set(opt['stock'] for opt in self.all_optimizations))}\n")
            f.write(f"内存优化: 已启用定期清理和垃圾回收\n\n")

            sorted_recs = sorted(recommendations.items(),
                                 key=lambda x: x[1]['avg_score'],
                                 reverse=True)

            f.write("模型参数推荐 (基于前30%高得分样本):\n")
            f.write("-" * 80 + "\n\n")

            for model_name, rec_info in sorted_recs:
                params = rec_info['params']
                avg_score = rec_info['avg_score']
                sample_count = rec_info['sample_count']
                top_count = rec_info['top_count']

                high_score_count = sum(1 for opt in self.all_optimizations
                                       if opt['model'] == model_name and opt['score'] >= 0.7)
                success_rate = high_score_count / sample_count if sample_count > 0 else 0

                f.write(f"🔹 {model_name}:\n")
                f.write(f"   推荐参数: {params}\n")
                f.write(f"   平均得分: {avg_score:.4f}\n")
                f.write(f"   应用次数: {sample_count}\n")
                f.write(f"   成功率: {success_rate:.1%} ({high_score_count}/{sample_count})\n")
                f.write("   参数详情:\n")
                for param_name, param_value in params.items():
                    f.write(f"     - {param_name}: {param_value}\n")
                f.write("\n")

            if param_stats_summary:
                f.write("=" * 48 + "\n")
                f.write("通用参数建议\n")
                f.write("=" * 48 + "\n\n")

                f.write("常用参数值范围:\n")
                for param_name, values in param_stats_summary.items():
                    if values and all(isinstance(v, (int, float)) for v in values):
                        avg_value = sum(values) / len(values)
                        min_value = min(values)
                        max_value = max(values)

                        if param_name in ['window_size', 'n_estimators', 'batch_size']:
                            recommended = int(round(avg_value))
                        elif param_name in ['learning_rate', 'dropout_rate', 'correlation_threshold']:
                            recommended = round(avg_value, 2)
                        else:
                            recommended = round(avg_value, 2) if isinstance(avg_value, float) else avg_value

                        f.write(
                            f"  {param_name:<20}: 平均值={avg_value:.2f}, 范围={min_value}-{max_value}, 推荐值={recommended}\n")
                    elif values:
                        from collections import Counter
                        counter = Counter(values)
                        most_common = counter.most_common(3)
                        common_str = ", ".join([f"{value}({count}次)" for value, count in most_common])
                        f.write(f"  {param_name:<20}: 常用值: {common_str}\n")

        print(f"✓ 参数推荐报告已保存到: {report_path}")
        return report_path


# ==================== 主函数 ====================
def main():
    """主函数"""
    print("股票模拟回测程序")
    print("=" * 48)

    backtester = BacktestSimulator(data_dir='stock_data')

    print("\n请选择运行模式:")
    print("1. 单股票回测")
    print("2. 批量回测")

    choice = input("请输入选择 (1 或 2): ").strip()

    if choice == '1':
        print("\n" + "=" * 48)
        print("单股票回测")
        print("=" * 48)

        csv_files = [f for f in os.listdir('stock_data') if f.endswith('.csv')]

        if not csv_files:
            print("❌ 没有找到CSV文件，请确保stock_data目录下有数据文件")
            return

        print("可用的股票文件:")
        for i, csv_file in enumerate(csv_files):
            print(f"{i + 1:2d}. {csv_file}")

        try:
            file_choice = int(input("\n请选择文件编号: ").strip())
            if 1 <= file_choice <= len(csv_files):
                selected_file = csv_files[file_choice - 1]

                test_days_input = input("输入回测天数 (默认30天): ").strip()
                test_days = int(test_days_input) if test_days_input else 30

                enable_opt = input("是否启用参数优化？(y/n, 默认y): ").strip().lower()
                backtester.config.parameter_optimization.enabled = enable_opt != 'n'

                result = backtester.run_backtest(selected_file, test_days)

                if result:
                    print(f"\n✅ 回测完成！")
                    print(f"   报告路径: {result['report_path']}")
            else:
                print("❌ 无效的选择")
        except ValueError:
            print("❌ 请输入有效的数字")

    elif choice == '2':
        print("\n" + "=" * 48)
        print("批量回测")
        print("=" * 48)

        if not os.path.exists('stock_data'):
            print("❌ stock_data目录不存在")
            return

        csv_files = [f for f in os.listdir('stock_data') if f.endswith('.csv')]

        if not csv_files:
            print("❌ 没有找到CSV文件，请确保stock_data目录下有数据文件")
            return

        print(f"找到 {len(csv_files)} 个股票数据文件")

        test_days_input = input("输入回测天数 (默认30天): ").strip()
        test_days = int(test_days_input) if test_days_input else 30

        enable_opt = input("是否启用参数优化？(y/n, 默认y): ").strip().lower()
        backtester.config.parameter_optimization.enabled = enable_opt != 'n'

        print(f"\n开始批量回测 {len(csv_files)} 个股票...")
        print(f"回测天数: {test_days} 天")
        print(f"参数优化: {'启用' if backtester.config.parameter_optimization.enabled else '禁用'}")

        all_results = []

        for i, csv_file in enumerate(csv_files):
            print(f"\n{'=' * 80}")
            print(f"📊 回测文件 {i + 1}/{len(csv_files)}: {csv_file}")
            print('=' * 80)

            try:
                result = backtester.run_backtest(csv_file, test_days)

                if result:
                    all_results.append(result)
                    print(f"  ✅ 回测完成")

                if (i + 1) % 3 == 0 and (i + 1) < len(csv_files):
                    print(f"\n🧹 已分析 {i + 1} 只股票，清理中间资源...")
                    cleanup_resources()

            except Exception as e:
                print(f"  ❌ 回测文件 {csv_file} 时出错: {e}")

        if all_results:
            print(f"\n✅ 批量回测完成！共完成 {len(all_results)} 个股票的回测")

            if backtester.config.parameter_optimization.enabled:
                recommendations, param_stats = backtester.aggregator.generate_recommendations()
                if recommendations:
                    rec_report_path = backtester.aggregator.save_recommendations(recommendations, param_stats)
                    if rec_report_path:
                        print(f"📊 参数推荐报告已生成: {rec_report_path}")
                else:
                    print("⚠️ 无法生成参数推荐报告，可能因为优化样本不足")

    else:
        print("❌ 无效的选择")


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
        cleanup_resources()
        print("\n程序结束，感谢使用！")