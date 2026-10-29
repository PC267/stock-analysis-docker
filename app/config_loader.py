# config_loader.py - 修复YAML加载器问题
import os
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict, is_dataclass, fields
import re
import json
import types
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np


# ==================== YAML 兼容性处理 ====================

class SafeTupleLoader(yaml.SafeLoader):
    """安全的YAML加载器，支持python/tuple标记"""

    def __init__(self, stream):
        super(SafeTupleLoader, self).__init__(stream)
        # 添加对python/tuple的处理
        self.add_constructor(
            'tag:yaml.org,2002:python/tuple',
            self.construct_python_tuple
        )
        # 添加对其他python特殊标记的处理
        self.add_constructor(
            'tag:yaml.org,2002:python/name:__main__.MyClass',
            self.construct_python_object
        )

    def construct_python_tuple(self, node):
        """将序列节点转换为元组 - 修复参数问题"""
        try:
            # 不同的PyYAML版本可能有不同的参数传递方式
            # 尝试处理节点
            if hasattr(node, 'value'):
                # 处理标量节点
                return node.value
            else:
                # 处理序列节点
                sequence = self.construct_sequence(node)
                return tuple(sequence)
        except Exception as e:
            print(f"警告: 元组解析失败: {e}")
            return ()

    def construct_python_object(self, node):
        """处理其他python对象（安全版本）"""
        # 对于不安全的对象，返回None
        return None


def safe_yaml_load(stream):
    """安全的YAML加载，支持python/tuple特殊标记"""
    try:
        if hasattr(stream, 'read'):
            content = stream.read()
        else:
            content = stream

        # 方法1：直接使用安全加载器，但处理特殊标记
        # 预处理：将!!python/tuple转换为普通列表
        content = content.replace('!!python/tuple', '')

        # 处理日期时间格式
        # 将 "2026-01-28 03:25:31.029546" 这样的字符串用引号包起来
        date_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+'

        def quote_date(match):
            return f"'{match.group(0)}'"

        content = re.sub(date_pattern, quote_date, content)

        # 使用FullLoader（支持更多特性）
        try:
            data = yaml.load(content, Loader=yaml.FullLoader)
            if data is None:
                print("⚠️ YAML文件内容为空，使用默认配置")
                return {}
            return data
        except Exception as e:
            print(f"FullLoader失败: {e}")

            # 方法2：使用SafeLoader
            try:
                data = yaml.safe_load(content)
                if data is None:
                    print("⚠️ YAML文件内容为空，使用默认配置")
                    return {}
                return data
            except Exception as e2:
                print(f"SafeLoader失败: {e2}")
                return {}
    except Exception as e:
        print(f"YAML加载失败: {e}")
        return {}


def safe_yaml_dump(data, stream=None, **kwargs):
    """安全的YAML导出，避免生成python/tuple标记"""
    kwargs.setdefault('default_flow_style', False)
    kwargs.setdefault('allow_unicode', True)
    kwargs.setdefault('sort_keys', False)
    kwargs.setdefault('indent', 2)

    # 创建自定义的SafeDumper，确保元组被序列化为列表
    class SafeTupleDumper(yaml.SafeDumper):
        def represent_tuple(self, data):
            return self.represent_sequence('tag:yaml.org,2002:seq', list(data))

        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', '')

    SafeTupleDumper.add_representer(tuple, SafeTupleDumper.represent_tuple)
    SafeTupleDumper.add_representer(type(None), SafeTupleDumper.represent_none)

    try:
        return yaml.dump(data, stream, Dumper=SafeTupleDumper, **kwargs)
    except Exception as e:
        print(f"YAML导出失败: {e}")
        # 回退到普通SafeDumper
        return yaml.dump(data, stream, **kwargs)


def safe_yaml_dump_with_comments(data, stream=None, **kwargs):
    """安全的YAML导出，为每个参数添加注释 - 简化版本"""
    kwargs.setdefault('default_flow_style', False)
    kwargs.setdefault('allow_unicode', True)
    kwargs.setdefault('sort_keys', False)
    kwargs.setdefault('indent', 2)

    class SimpleCommentedDumper(yaml.SafeDumper):
        """简化的带注释Dumper，不直接操作comment属性"""

        # 添加元组表示方法
        def represent_tuple(self, data):
            return self.represent_sequence('tag:yaml.org,2002:seq', list(data))

        # 添加None表示方法
        def represent_none(self, _):
            return self.represent_scalar('tag:yaml.org,2002:null', '')

    SimpleCommentedDumper.add_representer(tuple, SimpleCommentedDumper.represent_tuple)
    SimpleCommentedDumper.add_representer(type(None), SimpleCommentedDumper.represent_none)

    try:
        # 先导出YAML
        yaml_output = yaml.dump(data, Dumper=SimpleCommentedDumper, **kwargs)

        # 然后在输出中添加注释
        commented_output = _add_comments_to_yaml(yaml_output)

        if stream:
            stream.write(commented_output)
        return commented_output

    except Exception as e:
        print(f"带注释的YAML导出失败: {e}")
        # 回退到普通导出
        return safe_yaml_dump(data, stream, **kwargs)


def _add_comments_to_yaml(yaml_content):
    """在YAML内容中添加注释"""
    lines = yaml_content.split('\n')
    commented_lines = []

    # 配置部分的注释映射
    section_comments = {
        'stock:': '# ==================== 股票配置 ====================',
        'data:': '# ==================== 数据配置 ====================',
        'features:': '# ==================== 特征配置 ====================',
        'ensemble:': '# ==================== 集成模型配置 ====================',
        'correction:': '# ==================== 误差校正配置 ====================',
        'probabilistic:': '# ==================== 概率预测配置 ====================',
        'monitoring:': '# ==================== 监控配置 ====================',
        'simulator:': '# ==================== 模拟器配置 ====================',
        'reporting:': '# ==================== 报告配置 ====================',
        'sarima:': '# ==================== SARIMA模型配置 ====================',
        'garch:': '# ==================== GARCH模型配置 ====================',
        'xgboost:': '# ==================== XGBoost模型配置 ====================',
        'multi_factor:': '# ==================== 多因子模型配置 ====================',
        'technical_features': '# ==================== 技术特征配置 ===================='
    }

    # 参数注释映射
    param_comments = {
        # StockConfig
        'code:': '# 股票代码，如：000001',
        'name:': '# 股票名称，如：平安银行',
        'simulation_days:': '# 模拟交易的天数',

        # DataConfig
        'start_date_offset_days:': '# 起始日期偏移天数，如：3650表示10年前',
        'end_date:': '# 结束日期，可以是具体日期或"today"',
        'cache_dir:': '# 数据缓存目录',
        'results_dir:': '# 结果保存目录',

        # 基础模型参数
        'base_score:': '# 模型基础评分，用于模型权重分配',
        'enabled:': '# 是否启用该模型',

        # 模拟器主要参数
        'deterministic_mode:': '# 是否使用确定性模式（True为可重复结果）',
        'random_seed:': '# 随机种子，确保结果可重复',
        'simulation_days:': '# 模拟天数',

        # TechnicalFeaturesConfig 参数注释
        'enabled:': '# 是否启用技术特征提取',
        'include_indicators:': '# 是否包含技术指标',
        'include_golden_triangle:': '# 是否包含金三角指标',
        'include_kdj:': '# 是否包含KDJ指标',
        'include_normality:': '# 是否包含正态分布分析',
        'include_volume_analysis:': '# 是否包含成交量分析',
        'return_dataframe:': '# 是否返回DataFrame格式',

        # GoldenTriangleConfig 参数注释
        'short_period:': '# 金三角短期均线周期',
        'medium_period:': '# 金三角中期均线周期',
        'long_period:': '# 金三角长期均线周期',
        'window:': '# 金三角模式识别窗口',
        'overbought_threshold:': '# 金三角超买阈值',
        'oversold_threshold:': '# 金三角超卖阈值',
        'extreme_overbought_threshold:': '# 金三角极端超买阈值',
        'extreme_oversold_threshold:': '# 金三角极端超卖阈值',
        'lookback_period:': '# 金三角背离检测回溯周期',
        'strength_window:': '# 金三角趋势强度计算窗口',
        'divergence_window:': '# 金三角发散度计算窗口',

        # KDJConfig 参数注释
        'n:': '# KDJ的RSV计算周期',
        'm1:': '# KDJ的K值平滑周期',
        'm2:': '# KDJ的D值平滑周期',

        # NormalityConfig 参数注释
        'jarque_bera_min_samples:': '# Jarque-Bera正态性检验最小样本数',
        'shapiro_wilk_min_samples:': '# Shapiro-Wilk正态性检验最小样本数',
        'shapiro_wilk_max_samples:': '# Shapiro-Wilk正态性检验最大样本数',
        'normal_skewness_threshold:': '# 正常偏度阈值',
        'normal_kurtosis_threshold:': '# 正常峰度阈值',
        'qq_quantiles:': '# Q-Q图分位数数量',
        'fat_tail_threshold:': '# 厚尾判断阈值',
        'thin_tail_threshold:': '# 薄尾判断阈值',
        'normality_window:': '# 滚动正态性分析窗口',

        # VolumeAnalysisConfig 参数注释
        'high_volume_period:': '# 高量柱检测周期',
        'volume_ma_period:': '# 量比计算均线周期',
        'winner_window:': '# 获利盘比例计算窗口',
        'volume_double_threshold:': '# 成交量翻倍阈值',
        'volume_shrink_threshold:': '# 成交量萎缩阈值',
        'volume_min_ratio:': '# 成交量最小比例阈值',

        # TechnicalIndicatorsConfig 参数注释
        'rsi_period:': '# RSI计算周期',
        'macd_fast:': '# MACD快速移动平均周期',
        'macd_slow:': '# MACD慢速移动平均周期',
        'macd_signal:': '# MACD信号线周期',
        'bollinger_window:': '# 布林带窗口大小',
        'bollinger_num_std:': '# 布林带标准差倍数',
        'volume_ma_4:': '# 4日成交量均线',
        'volume_ma_11:': '# 11日成交量均线',
        'volume_ma_117:': '# 117日成交量均线',
        # 更多参数注释可以按需添加...
    }

    # 添加文件头注释
    commented_lines.append('# ========================================')
    commented_lines.append('# 股票预测系统配置')
    commented_lines.append('# 自动生成的配置文件，包含所有参数说明')
    commented_lines.append('# ========================================')
    commented_lines.append('')

    for line in lines:
        stripped = line.strip()

        # 检查是否是配置部分
        section_added = False
        for section_key, comment in section_comments.items():
            if stripped.startswith(section_key):
                commented_lines.append('')
                commented_lines.append(comment)
                section_added = True
                break

        # 检查是否是参数
        param_added = False
        for param_key, comment in param_comments.items():
            if stripped.startswith(param_key):
                # 添加缩进以匹配原始行的缩进
                indent = len(line) - len(line.lstrip())
                commented_lines.append(' ' * indent + comment)
                param_added = True
                break

        # 添加原始行
        if line:
            commented_lines.append(line)
        elif not section_added and not param_added:
            # 保留空行，但避免在添加注释后重复空行
            if commented_lines and commented_lines[-1] != '':
                commented_lines.append('')

    return '\n'.join(commented_lines)


def generate_comprehensive_commented_config():
    """生成带完整注释的配置文件（所有参数都有注释）"""
    try:
        print("=" * 60)
        print("生成带完整注释的配置文件（所有参数）")
        print("=" * 60)

        # 获取默认配置
        config_manager = ConfigManager()

        # 生成默认配置
        config = config_manager.load_config(
            config_path="config.yaml",
            stock_code="000001",
            stock_name="平安银行",
            days_to_simulate=30
        )

        # 导出带注释的配置
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"config_comprehensive_commented_{timestamp}.yaml"

        # 使用简化版本
        success = export_config_to_yaml_with_comments(config, output_file)

        if success:
            print(f"\n✅ 带完整注释的配置文件已生成:")
            print(f"   文件: {output_file}")

            # 显示文件大小
            file_size = os.path.getsize(output_file)
            print(f"\n📊 文件大小: {file_size} 字节 ({file_size / 1024:.1f} KB)")

            # 显示注释统计
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                comment_count = content.count('# =')
                param_comment_count = content.count('# ') - comment_count

            print(f"📝 注释统计:")
            print(f"   - 配置部分分隔符: {comment_count} 个")
            print(f"   - 参数说明注释: {param_comment_count} 个")

        return success

    except Exception as e:
        print(f"❌ 生成带注释的配置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== 配置类定义 ====================

@dataclass
class StockConfig:
    """股票配置"""
    code: str = "000001"  # 股票代码
    name: str = "平安银行"  # 股票名称
    simulation_days: int = 30  # 模拟天数

    def __post_init__(self):
        """初始化后处理"""
        self.simulation_days = int(self.simulation_days)


@dataclass
class DataConfig:
    """数据配置"""
    start_date_offset_days: int = 3650  # 10年
    end_date: str = "today"  # 结束日期
    cache_dir: str = "stock_data"  # 缓存目录
    results_dir: str = "results"  # 结果目录

    def __post_init__(self):
        """计算实际日期"""
        if self.end_date.lower() == "today":
            self.end_date = datetime.now()
        else:
            try:
                self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
            except:
                self.end_date = datetime.now()

        self.start_date = self.end_date - timedelta(days=self.start_date_offset_days)


@dataclass
class ModelConfig:
    """模型配置基类"""
    base_score: float = 6.0  # 基础评分
    enabled: bool = True  # 是否启用

@dataclass
class SARIMAConfig(ModelConfig):
    """SARIMA模型配置"""
    order: Tuple[int, int, int] = (1, 1, 1)  # ARIMA阶数
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)  # 季节阶数
    window_size: int = 50  # 窗口大小
    prediction_horizon: int = 5  # 预测周期
    weight_strategy: str = "equal"  # 权重策略
    p: int = 1  # 自回归阶数
    d: int = 1  # 差分阶数
    q: int = 1  # 移动平均阶数


@dataclass
class GARCHConfig(ModelConfig):
    """GARCH模型配置"""
    order: Tuple[int, int] = (1, 1)  # GARCH阶数
    p: int = 1  # GARCH项阶数
    q: int = 1  # ARCH项阶数
    window_size: int = 50  # 窗口大小
    prediction_horizon: int = 5  # 预测周期
    weight_strategy: str = "recent_weighted"  # 权重策略
    distribution: str = "t"  # 分布类型


@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost模型配置"""
    n_estimators: int = 50  # 树的数量
    max_depth: int = 4  # 最大深度
    learning_rate: float = 0.1  # 学习率
    n_features: int = 13  # 特征数量
    window_size: int = 50  # 窗口大小
    prediction_horizon: int = 5  # 预测周期
    weight_strategy: str = "equal"  # 权重策略


@dataclass
class MultiFactorConfig(ModelConfig):
    """多因子模型配置"""
    n_features: int = 15  # 特征数量
    window_size: int = 150  # 窗口大小
    prediction_horizon: int = 10  # 预测周期
    weight_strategy: str = "equal"  # 权重策略


@dataclass
class FeaturesConfig:
    """特征工程配置"""
    technical_indicators: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # 是否启用技术指标
        'rsi_period': 14,  # RSI计算周期
        'macd_fast': 12,  # MACD快速移动平均周期
        'macd_slow': 26,  # MACD慢速移动平均周期
        'bollinger_window': 20,  # 布林带窗口大小
        'kdj_period': 9,  # KDJ计算周期
        'kdj_smooth1': 3,  # KDJ平滑参数1
        'kdj_smooth2': 3  # KDJ平滑参数2
    })

    # 技术特征提取器配置
    technical_features: Dict[str, Any] = field(default_factory=lambda: {
        'include_indicators': True,  # 是否包含技术指标
        'return_dataframe': True,  # 是否返回DataFrame格式
        'normality_window': 20,  # 正态性检验窗口大小
        'atr_period': 14,  # ATR计算周期
        'high_volume_period': 1500  # 高成交量检测周期
    })


@dataclass
class EnsembleConfig:
    """集成模型配置 - 基于631支股票回测结果优化"""
    dynamic_weights: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # 是否启用动态权重
        'lookback_period': 30,  # 回看周期
        'min_data_points': 10,  # 最小数据点数
        'strategy': 'equal'  # 权重策略
    })

    # 模型评分配置 - 基于631支股票回测结果优化
    model_scoring: Dict[str, Any] = field(default_factory=lambda: {
        'market_regime_adjustments': {
            # 组合1: 平衡型 (推荐)
            'stable_trend': {'sarima': 0.8},  # 稳定趋势市场
            'high_volatility': {'garch': 1.0},  # 高波动市场
            'high_volatility_nonlinear': {'garch': 1.0, 'xgboost': 0.7, 'multi_factor': 0.5},
            # 高波动非线性市场
            'trending': {'sarima': 0.8},  # 趋势市场
            'default': {'sarima': 0.6, 'garch': 0.5, 'multi_factor': 0.3}  # 默认市场
        },
        # 其他可选组合
        'alternative_combinations': {
            # 组合2: 保守型 (侧重稳定模型)
            'conservative': {
                'stable_trend': {'sarima': 0.9},
                'high_volatility': {'garch': 1.0},
                'high_volatility_nonlinear': {'garch': 1.0, 'xgboost': 0.5},
                'trending': {'sarima': 0.9},
                'default': {'sarima': 0.7, 'garch': 0.6}
            },
            # 组合3: 激进型 (侧重预测能力)
            'aggressive': {
                'stable_trend': {'sarima': 0.8},
                'high_volatility': {'garch': 1.0, 'xgboost': 0.2},
                'high_volatility_nonlinear': {'garch': 1.0, 'xgboost': 0.9, 'multi_factor': 0.7},
                'trending': {'sarima': 0.2},
                'default': {'garch': 0.6, 'xgboost': 0.3}
            }
        },
        'selected_model_count': 3,  # 选择模型数量
        'confidence_threshold': 0.6,  # 置信度阈值

        # 模型统计信息 (用于参考)
        'model_statistics': {
            'usage_rate': {
                'garch': '36.8%',  # 主力模型，表现稳定
                'multi_factor': '15.2%',  # 多因子表现良好
                'xgboost': '10.9%',  # 非线性市场有效
                'sarima': '2.7%',  # 大幅降低权重

            },
            'removed_models': [
                {
                    'model': 'SARIMA',
                    'reason': '被ARIMA全面超越，相同状态ARIMA入选率31.9% vs SARIMA 7.2%',
                    'original_weight': 1.6,
                    'optimized_weight': '0.0-0.7'
                },
                {
                    'model': 'Monte Carlo',
                    'reason': '使用率低(2.2%)，仅14次入选，表现不稳定',
                    'original_weight': 0.4,
                    'optimized_weight': '0.0-0.5'
                }
            ]
        }
    })


@dataclass
class CorrectionConfig:
    """误差校正配置"""
    error_correction: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # 是否启用误差校正
        'window_size': 50,  # 校正窗口大小
        'correction_strength': 0.7  # 校正强度
    })


@dataclass
class ProbabilisticConfig:
    """概率预测配置"""
    uncertainty_quantification: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # 是否启用不确定性量化
        'n_samples': 1000,  # 采样次数
        'confidence_level': 0.95,  # 置信水平
        'distribution': 'normal'  # 分布类型
    })


@dataclass
class MonitoringConfig:
    """监控配置"""
    accuracy_tracking: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # 是否启用精度跟踪
        'evaluation_frequency': 'daily'  # 评估频率
    })
    ab_testing: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,  # 是否启用A/B测试
        'test_period': 30  # 测试周期
    })


@dataclass
class SimulatorConfig:
    """模拟器配置"""
    deterministic_mode: bool = True  # 确定性模式
    random_seed: int = 42  # 随机种子
    simulation_days: int = 10  # 模拟天数

    # ==================== 模型配置参数 ====================
    # GARCH模型配置
    garch_volatility_persistence: float = 0.9  # GARCH波动率持续性

    # 多因子模型配置
    multi_factor_n_features: int = 10  # 多因子特征数

    # XGBoost模型配置
    xgboost_n_features: int = 13  # XGBoost特征数


    # ==================== 市场状态参数 ====================
    high_volatility_threshold: float = 0.04  # 高波动率阈值
    low_volatility_threshold: float = 0.015  # 低波动率阈值
    strong_trend_threshold: float = 0.3  # 强趋势阈值
    stable_trend_volatility_threshold: float = 0.02  # 稳定趋势波动率阈值

    # ==================== 技术特征参数 ====================
    # RSI参数
    rsi_oversold_threshold: float = 30.0  # RSI超卖阈值
    rsi_overbought_threshold: float = 70.0  # RSI超买阈值
    rsi_normal_min: float = 30.0  # RSI正常最小值
    rsi_normal_max: float = 70.0  # RSI正常最大值

    # 技术分数调整参数
    technical_score_high_threshold: float = 0.7  # 技术分数高阈值
    technical_score_low_threshold: float = 0.3  # 技术分数低阈值
    technical_score_neutral: float = 0.5  # 技术分数中性值

    # 权重调整参数
    market_adaptation_boost: float = 1.2  # 市场适应提升因子
    high_vol_garch_boost: float = 1.3  # 高波动GARCH提升因子
    low_vol_sarima_boost: float = 1.2  # 低波动SARIMA提升因子
    high_tech_xgboost_boost: float = 1.1  # 高技术XGBoost提升因子

    # ==================== 模拟控制参数 ====================
    max_daily_return_limit: float = 0.08  # 最大日收益率限制
    volatility_multiplier: float = 2.5  # 波动率乘数
    deterministic_noise_scale_low: float = 0.3  # 低波动噪声规模
    deterministic_noise_scale_medium: float = 0.5  # 中波动噪声规模
    deterministic_noise_scale_high: float = 0.7  # 高波动噪声规模

    # 趋势调整参数
    trend_adjustment_factor: float = 0.1  # 趋势调整因子

    # ==================== K线生成参数 ====================
    # 日收益率范围
    daily_return_min: float = -0.1  # 最小日收益率
    daily_return_max: float = 0.08  # 最大日收益率

    # 价格范围参数
    price_range_min: float = 0.3  # 价格范围最小值比例
    price_range_max: float = 3.0  # 价格范围最大值比例

    # K线波动参数
    bullish_candle_range_multiplier: float = 1.5  # 看涨K线范围乘数
    bearish_candle_range_multiplier: float = 2.5  # 看跌K线范围乘数
    min_candle_range_ratio: float = 0.01  # 最小K线范围比例
    bullish_high_ratio: float = 0.10  # 看涨K线上影线比例
    bullish_low_ratio: float = 0.15  # 看涨K线下影线比例
    bearish_high_ratio: float = 0.15  # 看跌K线上影线比例
    bearish_low_ratio: float = 0.10  # 看跌K线下影线比例

    # 成交量参数
    base_volume_multiplier_high: float = 8.0  # 高波动成交量乘数
    base_volume_multiplier_medium: float = 5.0  # 中波动成交量乘数
    base_volume_multiplier_low: float = 3.0  # 低波动成交量乘数
    volume_noise_mean: float = 1.0  # 成交量噪声均值
    volume_noise_std: float = 0.2  # 成交量噪声标准差
    volume_min_ratio: float = 0.3  # 成交量最小比例
    volume_max_ratio: float = 3.0  # 成交量最大比例

    # 技术特征权重参数
    tech_weight_max: float = 0.8  # 技术权重最大值
    random_weight_min: float = 0.2  # 随机权重最小值

    # ==================== 验证参数 ====================
    max_allowed_daily_return: float = 0.15  # 最大允许日收益率
    price_change_sanity_check_min: float = 0.3  # 价格变化合理性检查最小值
    price_change_sanity_check_max: float = 3.0  # 价格变化合理性检查最大值

    # ==================== 回测结果集成参数 ====================
    confidence_cap: float = 0.95  # 置信度上限
    direction_accuracy_boost: float = 1.1  # 方向准确性提升因子
    technical_score_boost_threshold: float = 0.6  # 技术分数提升阈值
    trend_strength_boost_threshold: float = 0.3  # 趋势强度提升阈值
    trend_strength_boost_factor: float = 1.08  # 趋势强度提升因子
    technical_score_boost_factor: float = 1.05  # 技术分数提升因子

    # ==================== 市场状态映射参数 ====================
    regime_high_volatility_nonlinear_vol_threshold: float = 0.04  # 高波动非线性市场波动率阈值
    regime_high_volatility_vol_threshold: float = 0.04  # 高波动市场波动率阈值
    regime_trending_trend_threshold: float = 0.7  # 趋势市场趋势强度阈值
    regime_stable_trend_vol_threshold: float = 0.02  # 稳定趋势市场波动率阈值
    regime_low_volatility_vol_threshold: float = 0.015  # 低波动市场波动率阈值
    regime_low_volatility_return_threshold: float = 0.001  # 低波动市场收益率阈值

    # ==================== 原始simulator.py中的参数 ====================
    normal_std_factor: float = 1.0  # 正态分布标准差因子
    beta_alpha: int = 3  # Beta分布alpha参数
    beta_beta: int = 2  # Beta分布beta参数
    open_gap_multiplier: float = 0.3  # 开盘缺口乘数
    max_daily_change: float = 0.1  # 最大日变化率
    atr_period: int = 14  # ATR计算周期
    kline_body_ratio: float = 0.6  # K线实体比例
    kline_upper_ratio: float = 0.2  # K线上影线比例
    kline_lower_ratio: float = 0.2  # K线下影线比例
    volume_noise_factor: float = 0.2  # 成交量噪声因子
    market_vol_multiplier_high: float = 1.5  # 高波动市场乘数
    market_vol_multiplier_stable: float = 0.8  # 稳定市场乘数
    min_volatility: float = 0.005  # 最小波动率
    reverse_kline: bool = True  # 是否允许反转K线
    reverse_strength: float = 0.3  # 反转强度
    reverse_probability: float = 0.4  # 反转概率
    max_reverse_days: int = 15  # 最大连续反转天数
    max_allowed_volatility: float = 0.5  # 最大允许波动率
    min_price_range: float = 0.8  # 最小价格范围
    return_correction_factor: float = 0.7  # 收益率校正因子
    min_return_threshold: float = 0.001  # 最小收益率阈值
    price_match_adjustment: float = 0.05  # 价格匹配调整因子
    xgboost_feature_correction: bool = True  # XGBoost特征校正
    xgboost_return_boost: float = 1.2  # XGBoost收益率提升因子
    multi_factor_vol_scale: float = 0.8  # 多因子模型波动率缩放因子
    asymmetry_factor: float = 0.3  # 不对称因子


@dataclass
class ReportingConfig:
    """报告配置"""
    output_format: str = "html"  # 输出格式
    auto_open_html: bool = True  # 自动打开HTML
    save_path: str = "results"  # 保存路径
    timestamp_format: str = "%Y%m%d_%H%M%S"  # 时间戳格式


# ==================== 技术特征配置类 ====================

@dataclass
class GoldenTriangleConfig:
    """金三角指标配置"""
    short_period: int = 5  # 短期均线周期
    medium_period: int = 10  # 中期均线周期
    long_period: int = 20  # 长期均线周期
    window: int = 10  # 模式识别窗口
    overbought_threshold: int = 80  # 超买阈值
    oversold_threshold: int = 20  # 超卖阈值
    extreme_overbought_threshold: int = 90  # 极端超买阈值
    extreme_oversold_threshold: int = 10  # 极端超卖阈值
    lookback_period: int = 20  # 背离检测回溯周期
    strength_window: int = 20  # 趋势强度计算窗口
    divergence_window: int = 10  # 发散度计算窗口


@dataclass
class KDJConfig:
    """KDJ指标配置"""
    n: int = 9  # RSV计算周期
    m1: int = 3  # K值平滑周期
    m2: int = 3  # D值平滑周期
    overbought_threshold: int = 80  # 超买阈值
    oversold_threshold: int = 20  # 超卖阈值
    extreme_overbought_threshold: int = 90  # 极端超买阈值
    extreme_oversold_threshold: int = 10  # 极端超卖阈值
    lookback_period: int = 20  # 背离检测回溯周期
    volatility_window: int = 20  # 波动率计算窗口
    position_window: int = 50  # 位置分析窗口
    trend_window: int = 20  # 趋势分析窗口
    pattern_window: int = 10  # 模式识别窗口


@dataclass
class NormalityConfig:
    """正态分布分析配置"""
    jarque_bera_min_samples: int = 4  # Jarque-Bera检验最小样本数
    shapiro_wilk_min_samples: int = 3  # Shapiro-Wilk检验最小样本数
    shapiro_wilk_max_samples: int = 5000  # Shapiro-Wilk检验最大样本数
    normal_skewness_threshold: float = 0.5  # 正常偏度阈值
    normal_kurtosis_threshold: float = 1.0  # 正常峰度阈值
    qq_quantiles: int = 100  # Q-Q图分位数数量
    deviation_z_scores: List[int] = field(default_factory=lambda: [1, 2, 3])  # 偏离分析Z分数
    fat_tail_threshold: float = -0.01  # 厚尾判断阈值
    thin_tail_threshold: float = 0.01  # 薄尾判断阈值
    normality_window: int = 20  # 滚动正态性分析窗口


@dataclass
class VolumeAnalysisConfig:
    """成交量分析配置"""
    high_volume_period: int = 1500  # 高量柱检测周期
    volume_ma_period: int = 5  # 量比计算均线周期
    winner_window: int = 20  # 获利盘比例计算窗口
    ma_periods: List[int] = field(default_factory=lambda: [20, 60, 120])  # 均线位置判断周期
    volume_double_threshold: float = 2.0  # 成交量翻倍阈值
    volume_shrink_threshold: float = 0.5  # 成交量萎缩阈值
    volume_min_ratio: float = 0.02  # 成交量最小比例阈值

    # 量比量化分类阈值
    volume_category_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        '严重缩量': (0.001, 0.5),
        '缩量': (0.5, 0.8),
        '正常量比': (0.8, 1.5),
        '温和放量': (1.5, 2.5),
        '明显放量': (2.5, 5.0),
        '剧烈放量': (5.0, 10.0),
        '反向操作': (10.0, float('inf'))
    })


@dataclass
class TechnicalIndicatorsConfig:
    """技术指标基础配置"""
    rsi_period: int = 14  # RSI计算周期
    macd_fast: int = 12  # MACD快速移动平均周期
    macd_slow: int = 26  # MACD慢速移动平均周期
    macd_signal: int = 9  # MACD信号线周期
    bollinger_window: int = 20  # 布林带窗口大小
    bollinger_num_std: int = 2  # 布林带标准差倍数

    # 成交量均线周期
    volume_ma_4: int = 4  # 4日成交量均线
    volume_ma_11: int = 11  # 11日成交量均线
    volume_ma_117: int = 117  # 117日成交量均线


@dataclass
class TechnicalFeaturesConfig:
    """技术特征综合配置"""
    enabled: bool = True  # 是否启用技术特征提取
    include_indicators: bool = True  # 是否包含技术指标
    include_golden_triangle: bool = True  # 是否包含金三角指标
    include_kdj: bool = True  # 是否包含KDJ指标
    include_normality: bool = True  # 是否包含正态分布分析
    include_volume_analysis: bool = True  # 是否包含成交量分析
    return_dataframe: bool = True  # 是否返回DataFrame格式

    # 子配置
    golden_triangle: GoldenTriangleConfig = field(default_factory=GoldenTriangleConfig)
    kdj: KDJConfig = field(default_factory=KDJConfig)
    normality: NormalityConfig = field(default_factory=NormalityConfig)
    volume_analysis: VolumeAnalysisConfig = field(default_factory=VolumeAnalysisConfig)
    indicators: TechnicalIndicatorsConfig = field(default_factory=TechnicalIndicatorsConfig)

    def __post_init__(self):
        """确保子配置正确初始化"""
        if isinstance(self.golden_triangle, dict):
            self.golden_triangle = GoldenTriangleConfig(**self.golden_triangle)
        if isinstance(self.kdj, dict):
            self.kdj = KDJConfig(**self.kdj)
        if isinstance(self.normality, dict):
            self.normality = NormalityConfig(**self.normality)
        if isinstance(self.volume_analysis, dict):
            self.volume_analysis = VolumeAnalysisConfig(**self.volume_analysis)
        if isinstance(self.indicators, dict):
            self.indicators = TechnicalIndicatorsConfig(**self.indicators)


# ==================== 完整配置类 ====================

class FullConfig:
    """完整配置（支持动态参数推荐）"""

    def __init__(
            self,
            stock: Union[StockConfig, Dict[str, Any]],
            data: Union[DataConfig, Dict[str, Any]],
            features: Union[FeaturesConfig, Dict[str, Any]],
            ensemble: Union[EnsembleConfig, Dict[str, Any]],
            correction: Union[CorrectionConfig, Dict[str, Any]],
            probabilistic: Union[ProbabilisticConfig, Dict[str, Any]],
            monitoring: Union[MonitoringConfig, Dict[str, Any]],
            simulator: Union[SimulatorConfig, Dict[str, Any]],
            reporting: Union[ReportingConfig, Dict[str, Any]],

            # 各模型配置
            sarima: Union[SARIMAConfig, Dict[str, Any]] = None,
            garch: Union[GARCHConfig, Dict[str, Any]] = None,
            xgboost: Union[XGBoostConfig, Dict[str, Any]] = None,
            multi_factor: Union[MultiFactorConfig, Dict[str, Any]] = None,
            technical_features: Union[TechnicalFeaturesConfig, Dict[str, Any]] = None,

            # 参数推荐文件路径
            recommendation_file_path: str = "backtest_results/parameter_recommendations/parameter_recommendations.txt",

            # 运行时属性
            base_dir: str = None,
            current_timestamp: str = None
    ):
        """初始化配置，自动从推荐文件加载参数"""
        # 设置运行时属性
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.current_timestamp = current_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recommendation_file_path = recommendation_file_path

        # 初始化基础配置
        self.stock = self._ensure_config_type(stock, StockConfig)
        self.data = self._ensure_config_type(data, DataConfig)
        self.features = self._ensure_config_type(features, FeaturesConfig)
        self.ensemble = self._ensure_config_type(ensemble, EnsembleConfig)
        self.correction = self._ensure_config_type(correction, CorrectionConfig)
        self.probabilistic = self._ensure_config_type(probabilistic, ProbabilisticConfig)
        self.monitoring = self._ensure_config_type(monitoring, MonitoringConfig)
        self.simulator = self._ensure_config_type(simulator, SimulatorConfig)
        self.reporting = self._ensure_config_type(reporting, ReportingConfig)
        # 技术特征配置
        if technical_features is None:
            self.technical_features = TechnicalFeaturesConfig()
        else:
            self.technical_features = self._ensure_config_type(technical_features, TechnicalFeaturesConfig)
        # 加载参数推荐
        self.recommendation_loader = ParameterConfigLoader(self.recommendation_file_path)

        # 初始化模型配置，优先使用推荐参数
        self._init_model_configs(
            sarima=sarima,
            garch=garch,
            xgboost=xgboost,
            multi_factor=multi_factor,
        )

        # 创建目录
        self._create_directories()

        # 确保模拟天数同步
        self.simulator.simulation_days = self.stock.simulation_days

    def _ensure_config_type(self, config_data, config_class):
        """确保配置数据转换为正确的类型"""
        if isinstance(config_data, config_class):
            return config_data
        elif isinstance(config_data, dict):
            # 处理元组字段
            processed_data = {}

            # 获取配置类的所有字段
            try:
                import inspect
                sig = inspect.signature(config_class.__init__)
                valid_params = list(sig.parameters.keys())[1:]  # 排除 self

                for key, value in config_data.items():
                    # 只保留有效的参数
                    if key not in valid_params:
                        print(f"⚠️ 警告: 配置类 {config_class.__name__} 忽略未知字段: {key}")
                        continue

                    # 如果是列表且字段类型提示是元组，转换为元组
                    if isinstance(value, list):
                        # 检查字段的类型提示
                        try:
                            annotations = inspect.get_annotations(config_class)
                            if key in annotations and annotations[key] == Tuple:
                                value = tuple(value)
                        except:
                            pass
                    processed_data[key] = value
            except:
                # 如果无法获取签名，使用原始方法
                for key, value in config_data.items():
                    # 如果是列表且字段类型提示是元组，转换为元组
                    if isinstance(value, list):
                        try:
                            import inspect
                            annotations = inspect.get_annotations(config_class)
                            if key in annotations and annotations[key] == Tuple:
                                value = tuple(value)
                        except:
                            pass
                    processed_data[key] = value

            return config_class(**processed_data)
        else:
            return config_class()

    def _init_model_configs(self, **kwargs):
        """初始化模型配置，应用推荐参数"""
        # 模型配置映射
        model_config_classes = {
            'sarima': SARIMAConfig,
            'garch': GARCHConfig,
            'xgboost': XGBoostConfig,
            'multi_factor': MultiFactorConfig,
        }

        # 为每个模型创建配置
        for model_name, config_class in model_config_classes.items():
            # 获取传入的配置字典
            config_dict = kwargs.get(model_name, {})

            # 获取推荐参数
            recommendation = self.recommendation_loader.get_model_recommendation(model_name)

            # 构建最终参数字典
            final_params = {}

            # 1. 先使用传入的参数（来自配置文件）
            if config_dict:
                if isinstance(config_dict, config_class):
                    # 如果已经是配置对象，直接使用
                    setattr(self, model_name, config_dict)
                    continue
                else:
                    final_params.update(config_dict)

            # 2. 然后使用推荐参数覆盖
            # 推荐参数应该覆盖配置文件参数
            if recommendation and hasattr(recommendation, 'parameters'):
                for key, value in recommendation.parameters.items():
                    final_params[key] = value

            # 3. 处理特殊配置
            if model_name == 'sarima' and 'seasonal_order' not in final_params:
                # SARIMA的特殊季节参数
                final_params['seasonal_order'] = (1, 1, 1, 12)

            # 4. 确保元组类型正确
            final_params = self._handle_field_dependencies(model_name, final_params)
            final_params = self._ensure_tuple_types(final_params, config_class)

            # 5. 创建配置对象
            try:
                # 获取dataclass的所有字段
                import inspect
                sig = inspect.signature(config_class.__init__)
                fields_names = list(sig.parameters.keys())[1:]  # 排除self

                # 过滤出有效的字段
                valid_params = {}
                for field_name in fields_names:
                    if field_name in final_params:
                        valid_params[field_name] = final_params[field_name]

                # 创建配置对象
                config_obj = config_class(**valid_params)
                setattr(self, model_name, config_obj)

            except Exception as e:
                print(f"⚠️ 创建 {model_name} 配置失败: {e}")
                # 使用默认配置
                setattr(self, model_name, config_class())

    def _handle_field_dependencies(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """处理字段间的依赖关系"""
        if model_name == 'sarima':
            # 如果提供了p、d、q，但order还是默认值，则更新order
            if 'p' in params or 'd' in params or 'q' in params:
                p = params.get('p', 1)
                d = params.get('d', 1)
                q = params.get('q', 1)
                params['order'] = (p, d, q)

        elif model_name == 'garch':
            # 如果提供了p、q，但order还是默认值，则更新order
            if 'p' in params or 'q' in params:
                p = params.get('p', 1)
                q = params.get('q', 1)
                params['order'] = (p, q)

        return params

    def _ensure_tuple_types(self, params: Dict[str, Any], config_class) -> Dict[str, Any]:
        """确保参数中的元组类型正确"""
        try:
            import inspect
            annotations = inspect.get_annotations(config_class)

            for key, value in params.items():
                if key in annotations and annotations[key] == Tuple:
                    if isinstance(value, list):
                        params[key] = tuple(value)
                    elif isinstance(value, str):
                        # 尝试从字符串解析
                        if value.startswith('(') and value.endswith(')'):
                            try:
                                # 解析如 "(1, 1, 1)" 的字符串
                                value = value.strip('()')
                                parts = [int(x.strip()) for x in value.split(',')]
                                params[key] = tuple(parts)
                            except:
                                pass
        except:
            pass

        return params

    def _create_directories(self):
        """创建必要的目录"""
        data_dir = os.path.join(self.base_dir, self.data.cache_dir)
        results_dir = os.path.join(self.base_dir, self.data.results_dir)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

    @property
    def cache_file_path(self) -> str:
        """缓存文件路径"""
        data_dir = os.path.join(self.base_dir, self.data.cache_dir)
        return os.path.join(data_dir, f"{self.stock.code}_{self.stock.name}_baostock.csv")

    @property
    def result_file_prefix(self) -> str:
        """结果文件前缀"""
        results_dir = os.path.join(self.base_dir, self.data.results_dir)
        timestamp = datetime.now().strftime(self.reporting.timestamp_format)
        return os.path.join(results_dir, f"{self.stock.code}_{timestamp}")

    def get_model_configs(self) -> Dict[str, Any]:
        """获取所有模型配置"""
        return {
            'sarima': self.safe_asdict(self.sarima),
            'garch': self.safe_asdict(self.garch),
            'xgboost': self.safe_asdict(self.xgboost),
            'multi_factor': self.safe_asdict(self.multi_factor),
        }

    def safe_asdict(self, obj):
        """安全的dataclass转字典，处理非dataclass字段"""
        if not is_dataclass(obj):
            return obj

        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)

            # 递归处理嵌套的dataclass
            if is_dataclass(value):
                result[field.name] = self.safe_asdict(value)
            elif isinstance(value, (list, tuple)):
                # 处理列表和元组中的dataclass
                result[field.name] = type(value)(
                    self.safe_asdict(item) if is_dataclass(item) else item for item in value)
            elif isinstance(value, dict):
                # 处理字典中的dataclass
                result[field.name] = {k: self.safe_asdict(v) if is_dataclass(v) else v for k, v in value.items()}
            else:
                result[field.name] = value

        return result

    def get_model_names(self) -> List[str]:
        """获取所有模型名称"""
        return ['sarima', 'garch', 'xgboost','multi_factor']

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，使用安全的dataclass转字典方法"""
        return {
            'stock': self.safe_asdict(self.stock),
            'data': self.safe_asdict(self.data),
            'features': self.safe_asdict(self.features),
            'ensemble': self.safe_asdict(self.ensemble),
            'correction': self.safe_asdict(self.correction),
            'probabilistic': self.safe_asdict(self.probabilistic),
            'monitoring': self.safe_asdict(self.monitoring),
            'simulator': self.safe_asdict(self.simulator),
            'reporting': self.safe_asdict(self.reporting),
            'sarima': self.safe_asdict(self.sarima),
            'garch': self.safe_asdict(self.garch),
            'xgboost': self.safe_asdict(self.xgboost),
            'multi_factor': self.safe_asdict(self.multi_factor),
            'technical_features': self.safe_asdict(self.technical_features)
        }

    def apply_latest_recommendations(self):
        """应用最新的推荐参数（热更新）"""
        # 重新初始化模型配置
        self._init_model_configs(
            sarima=self.safe_asdict(self.sarima),
            garch=self.safe_asdict(self.garch),
            xgboost=self.safe_asdict(self.xgboost),
            multi_factor=self.safe_asdict(self.multi_factor),
        )

        print("✅ 推荐参数已更新")


# ==================== 参数推荐系统 ====================

@dataclass
class ModelRecommendation:
    """单个模型的参数推荐"""
    model_name: str  # 模型名称
    parameters: Dict[str, Any]  # 参数
    average_score: float  # 平均得分
    application_count: int  # 应用次数
    success_rate: float  # 成功率

    def __str__(self):
        return f"{self.model_name}: {self.parameters} (得分: {self.average_score:.4f})"


@dataclass
class GeneralParameterRecommendation:
    """通用参数推荐"""
    parameter_name: str  # 参数名称
    average_value: float  # 平均值
    value_range: str  # 值范围
    recommended_value: Any  # 推荐值
    usage_count: Optional[int] = None  # 使用次数

    def __str__(self):
        return f"{self.parameter_name}: 推荐值={self.recommended_value}, 范围={self.value_range}, 平均值={self.average_value:.2f}"

class ParameterConfigLoader:
    """参数配置加载器"""

    def __init__(self, config_file_path: str = None):
        self.config_file_path = config_file_path
        self.model_recommendations: Dict[str, ModelRecommendation] = {}
        self.general_parameters: Dict[str, GeneralParameterRecommendation] = {}
        self.weight_strategy_stats: Dict[str, int] = {}
        self.distribution_stats: Dict[str, int] = {}
        self.loaded = False

        # 如果提供了文件路径，立即加载
        if config_file_path:
            self.load_config_file(config_file_path)

    def load_config_file(self, config_file_path: str = None) -> bool:
        """加载配置文件"""
        if config_file_path:
            self.config_file_path = config_file_path

        if not self.config_file_path:
            print("❌ 未指定配置文件路径")
            return False

        try:
            # 检查文件是否存在
            if not os.path.exists(self.config_file_path):
                print(f"⚠️ 推荐文件不存在: {self.config_file_path}")
                print("⚠️ 使用默认模型参数")
                return True

            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # 简化解析：直接查找每个模型
            self._simple_parse_models(content)
            self._parse_general_parameters(content)
            self.loaded = True
        except Exception as e:
            print(f"❌ 加载参数推荐文件失败: {e}")
            import traceback
            traceback.print_exc()
            print("⚠️ 使用默认模型参数")
            return False

    def _simple_parse_models(self, content: str):
        """简单解析模型参数"""
        # 查找所有模型块
        models = ['sarima', 'garch', 'xgboost','multi_factor']

        for model_name in models:
            # 查找模型推荐部分
            model_pattern = rf'{model_name}:\s*\n\s*推荐参数:\s*({{.*?}})'

            matches = list(re.finditer(model_pattern, content, re.DOTALL | re.IGNORECASE))

            if matches:
                # 取第一个匹配
                match = matches[0]
                param_str = match.group(1).strip()

                # 查找得分
                score_pattern = r'平均得分:\s*([\d\.]+)'
                score_matches = list(re.finditer(score_pattern, content[match.start():match.start() + 500]))
                average_score = float(score_matches[0].group(1)) if score_matches else 0.0

                # 查找应用次数
                app_pattern = r'应用次数:\s*(\d+)/(\d+)'
                app_matches = list(re.finditer(app_pattern, content[match.start():match.start() + 500]))
                application_count = int(app_matches[0].group(1)) if app_matches else 0

                # 查找成功率
                success_pattern = r'成功率:\s*([\d\.]+)%'
                success_matches = list(re.finditer(success_pattern, content[match.start():match.start() + 500]))
                success_rate = float(success_matches[0].group(1)) if success_matches else 0.0

                # 解析参数
                parameters = self._parse_parameter_string(param_str)

                recommendation = ModelRecommendation(
                    model_name=model_name,
                    parameters=parameters,
                    average_score=average_score,
                    application_count=application_count,
                    success_rate=success_rate
                )

                self.model_recommendations[model_name] = recommendation

    def _parse_parameter_string(self, param_str: str) -> Dict[str, Any]:
        """解析参数字符串为字典"""
        # 清理字符串
        param_str = param_str.strip()

        # 如果是空的
        if not param_str or param_str == '{}':
            return {}

        # 尝试直接解析为JSON
        if param_str.startswith('{') and param_str.endswith('}'):
            try:
                # 预处理：替换单引号为双引号
                json_str = param_str.replace("'", '"')

                # 修复常见的JSON格式问题
                # 1. 修复没有引号的键
                json_str = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_str)

                # 2. 修复尾随逗号
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)

                # 3. 处理元组格式
                json_str = re.sub(r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', r'[\1, \2, \3]', json_str)
                json_str = re.sub(r'\(\s*(\d+)\s*,\s*(\d+)\s*\)', r'[\1, \2]', json_str)

                parsed = json.loads(json_str)
                return parsed
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}, 使用自定义解析")
                return self._manual_parse_dict(param_str)
        else:
            return self._manual_parse_dict(param_str)

    def _manual_parse_dict(self, param_str: str) -> Dict[str, Any]:
        """手动解析字典字符串"""
        params = {}

        # 移除大括号
        if param_str.startswith('{'):
            param_str = param_str[1:]
        if param_str.endswith('}'):
            param_str = param_str[:-1]

        param_str = param_str.strip()

        if not param_str:
            return params

        # 分割键值对（处理嵌套）
        parts = []
        current = ""
        depth = 0

        for char in param_str:
            if char == '{' or char == '(':
                depth += 1
            elif char == '}' or char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                parts.append(current.strip())
                current = ""
                continue

            current += char

        if current:
            parts.append(current.strip())

        # 解析每个键值对
        for part in parts:
            if ':' in part:
                key_value = part.split(':', 1)
                key = key_value[0].strip().strip("'\"")
                value_str = key_value[1].strip()

                # 解析值
                if value_str.startswith('{'):
                    # 嵌套字典
                    value = self._manual_parse_dict(value_str)
                elif value_str.startswith('['):
                    # 列表
                    value = self._parse_list_string(value_str)
                elif value_str.startswith('('):
                    # 元组
                    value = self._parse_tuple_string(value_str)
                else:
                    # 简单值
                    value = self._parse_simple_value(value_str.strip("'\" "))

                params[key] = value

        return params

    def _parse_list_string(self, list_str: str) -> List[Any]:
        """解析列表字符串"""
        if list_str.startswith('[') and list_str.endswith(']'):
            list_str = list_str[1:-1].strip()
            if not list_str:
                return []

            items = []
            current = ""
            depth = 0

            for char in list_str:
                if char == '[' or char == '{' or char == '(':
                    depth += 1
                elif char == ']' or char == '}' or char == ')':
                    depth -= 1
                elif char == ',' and depth == 0:
                    if current:
                        items.append(self._parse_simple_value(current.strip().strip("'\" ")))
                    current = ""
                    continue

                current += char

            if current:
                items.append(self._parse_simple_value(current.strip().strip("'\" ")))

            return items
        return []

    def _parse_tuple_string(self, tuple_str: str) -> Tuple:
        """解析元组字符串"""
        if tuple_str.startswith('(') and tuple_str.endswith(')'):
            tuple_str = tuple_str[1:-1].strip()
            if not tuple_str:
                return ()

            items = []
            current = ""
            depth = 0

            for char in tuple_str:
                if char == '(' or char == '[' or char == '{':
                    depth += 1
                elif char == ')' or char == ']' or char == '}':
                    depth -= 1
                elif char == ',' and depth == 0:
                    if current:
                        items.append(self._parse_simple_value(current.strip().strip("'\" ")))
                    current = ""
                    continue

                current += char

            if current:
                items.append(self._parse_simple_value(current.strip().strip("'\" ")))

            return tuple(items)
        return ()

    def _parse_simple_value(self, value_str: str) -> Any:
        """解析简单值"""
        if not value_str:
            return None

        # 布尔值
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'

        # 数字值
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # 字符串
            return value_str

    def _parse_general_parameters(self, content: str):
        """解析通用参数建议部分"""
        # 查找通用参数部分
        if '常用参数值范围:' in content:
            start_idx = content.find('常用参数值范围:')
            section = content[start_idx:start_idx + 1000]  # 查看后面1000个字符

            # 解析参数行
            lines = section.split('\n')
            for line in lines:
                line = line.strip()

                # 匹配模式: parameter_name: 平均值=X, 范围=Y, 推荐值=Z
                pattern = r'(\w+)\s*:\s*平均值=([\d\.]+),\s*范围=([\d\-\.]+),\s*推荐值=([\d\.]+)'
                match = re.search(pattern, line)

                if match:
                    param_name = match.group(1)
                    avg_value = float(match.group(2))
                    value_range = match.group(3)
                    recommended_value = float(match.group(4))

                    recommendation = GeneralParameterRecommendation(
                        parameter_name=param_name,
                        average_value=avg_value,
                        value_range=value_range,
                        recommended_value=recommended_value
                    )

                    self.general_parameters[param_name] = recommendation

    def get_model_recommendation(self, model_name: str) -> Optional[ModelRecommendation]:
        """获取指定模型的推荐参数"""
        return self.model_recommendations.get(model_name)

    def get_general_parameter(self, param_name: str) -> Optional[GeneralParameterRecommendation]:
        """获取通用参数推荐"""
        return self.general_parameters.get(param_name)

    def get_optimal_weight_strategy(self) -> str:
        """获取最优权重策略"""
        if not self.weight_strategy_stats:
            return "equal"
        return max(self.weight_strategy_stats.items(), key=lambda x: x[1])[0]

    def get_optimal_distribution(self) -> str:
        """获取最优分布类型"""
        if not self.distribution_stats:
            return "normal"
        return max(self.distribution_stats.items(), key=lambda x: x[1])[0]

    def print_recommendations(self):
        """打印推荐参数摘要"""
        if not self.loaded:
            print("参数推荐未加载")
            return

        print("\n" + "=" * 48)
        print("参数推荐摘要")
        print("=" * 48)

        print("\n📊 模型参数推荐:")
        for model_name, rec in self.model_recommendations.items():
            print(f"  {model_name:15s} | 得分: {rec.average_score:.4f} | 成功率: {rec.success_rate}%")
            print(f"    参数: {rec.parameters}")

        print("\n📊 通用参数推荐:")
        for param_name, rec in self.general_parameters.items():
            print(f"  {param_name:20s} | 推荐值: {rec.recommended_value} | 范围: {rec.value_range}")

        print("=" * 48)

    def get_recommended_value(self, model_name, param_name, default_value):
        """
        获取推荐参数值

        Args:
            model_name: 模型名称
            param_name: 参数名称
            default_value: 默认值

        Returns:
            推荐参数值或默认值
        """
        # 首先尝试从推荐配置中获取
        if hasattr(self, 'recommendations') and self.recommendations:
            if model_name in self.recommendations and param_name in self.recommendations[model_name]:
                return self.recommendations[model_name][param_name]

        # 尝试从参数配置中获取
        if hasattr(self, 'parameters') and self.parameters:
            if model_name in self.parameters and param_name in self.parameters[model_name]:
                return self.parameters[model_name][param_name]

        # 返回默认值
        return default_value


# ==================== 配置管理器 ====================

_config_loader = None

def get_config_loader(config_file_path: str = None) -> ParameterConfigLoader:
    """获取配置加载器单例"""
    global _config_loader

    if _config_loader is None:
        _config_loader = ParameterConfigLoader(config_file_path)

    return _config_loader

class ConfigManager:
    """配置管理器"""

    _instance = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = None
            self.config_path = None
            self.default_config_path = "config.yaml"
            self.initialized = True

    def load_config(
            self,
            config_path: Optional[str] = None,
            stock_code: Optional[str] = None,
            stock_name: Optional[str] = None,
            days_to_simulate: Optional[int] = None,
            recommendation_file_path: str = "backtest_results/parameter_recommendations/parameter_recommendations.txt"
    ) -> FullConfig:
        """加载配置"""
        self.config_path = config_path or self.default_config_path

        # 加载YAML配置
        config_dict = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = safe_yaml_load(f)
                    if config_dict is None:
                        config_dict = {}
                        print("⚠️ YAML文件内容为空，使用默认配置")
            except Exception as e:
                print(f"❌ 加载YAML配置文件失败: {e}")
                print("⚠️ 使用默认配置")
                config_dict = self._get_default_config()
        else:
            print(f"⚠️ 配置文件不存在: {self.config_path}")
            print("⚠️ 使用默认配置")
            config_dict = self._get_default_config()

        # 提取主要配置
        system_config = config_dict.get('stock_system', {})

        # 覆盖命令行参数
        if stock_code:
            system_config['stock'] = system_config.get('stock', {})
            system_config['stock']['code'] = stock_code
        if stock_name:
            system_config['stock'] = system_config.get('stock', {})
            system_config['stock']['name'] = stock_name
        if days_to_simulate:
            system_config['stock'] = system_config.get('stock', {})
            system_config['stock']['simulation_days'] = days_to_simulate

        # 创建完整配置对象，传递推荐文件路径
        self.config = FullConfig(
            stock=system_config.get('stock', {}),
            data=system_config.get('data', {}),
            features=system_config.get('features', {}),
            ensemble=system_config.get('ensemble', {}),
            correction=system_config.get('correction', {}),
            probabilistic=system_config.get('probabilistic', {}),
            monitoring=system_config.get('monitoring', {}),
            simulator=system_config.get('simulator', {}),
            reporting=system_config.get('reporting', {}),
            sarima=system_config.get('sarima', {}),
            garch=system_config.get('garch', {}),
            xgboost=system_config.get('xgboost', {}),
            multi_factor=system_config.get('multi_factor', {}),
            recommendation_file_path=recommendation_file_path
        )

        return self.config

    def save_config(self, file_path: Optional[str] = None):
        """保存配置到文件"""
        if not self.config:
            return

        file_path = file_path or self.config_path or self.default_config_path

        # 构建配置字典
        config_dict = {
            'stock_system': self.config.to_dict()
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                safe_yaml_dump(config_dict, f)

            print(f"✅ 配置已保存到 {file_path}")
        except Exception as e:
            print(f"❌ 保存配置失败: {e}")

    def save_config_with_comments(self, file_path: Optional[str] = None):
        """保存配置到文件，并添加注释说明"""
        if not self.config:
            return

        file_path = file_path or self.config_path or self.default_config_path

        # 构建配置字典
        config_dict = {
            'stock_system': self.config.to_dict()
        }

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                safe_yaml_dump_with_comments(config_dict, f)

            print(f"✅ 带注释的配置已保存到 {file_path}")
            print(f"📝 所有参数均已添加详细说明")
        except Exception as e:
            print(f"❌ 保存带注释的配置失败: {e}")
            # 回退到普通保存
            self.save_config(file_path)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        # 直接返回默认配置字典
        return {
            'stock_system': {
                'stock': {
                    'code': '000001',  # 股票代码
                    'name': '平安银行',  # 股票名称
                    'simulation_days': 30  # 模拟天数
                },
                'data': {
                    'start_date_offset_days': 3650,  # 起始日期偏移天数
                    'end_date': 'today',  # 结束日期
                    'cache_dir': 'stock_data',  # 缓存目录
                    'results_dir': 'results'  # 结果目录
                },
                'features': {
                    'technical_indicators': {
                        'enabled': True,  # 是否启用技术指标
                        'rsi_period': 14,  # RSI计算周期
                        'macd_fast': 12,  # MACD快速移动平均周期
                        'macd_slow': 26,  # MACD慢速移动平均周期
                        'bollinger_window': 20,  # 布林带窗口大小
                        'kdj_period': 9,  # KDJ计算周期
                        'kdj_smooth1': 3,  # KDJ平滑参数1
                        'kdj_smooth2': 3  # KDJ平滑参数2
                    },
                    'technical_features': {
                        'include_indicators': True,  # 是否包含技术指标
                        'return_dataframe': True,  # 是否返回DataFrame格式
                        'normality_window': 20,  # 正态性检验窗口大小
                        'atr_period': 14,  # ATR计算周期
                        'high_volume_period': 1500  # 高成交量检测周期
                    }
                },
                'ensemble': {
                    'dynamic_weights': {
                        'enabled': True,  # 是否启用动态权重
                        'lookback_period': 30,  # 回看周期
                        'min_data_points': 10,  # 最小数据点数
                        'strategy': 'equal'  # 权重策略
                    },
                    'model_scoring': {
                        'market_regime_adjustments': {
                            # 组合1: 平衡型 (推荐)
                            'stable_trend': {'sarima': 0.8},  # 稳定趋势市场
                            'high_volatility': {'garch': 1.0},  # 高波动市场
                            'high_volatility_nonlinear': {'garch': 1.0, 'xgboost': 0.7, 'multi_factor': 0.5},
                            # 高波动非线性市场
                            'trending': {'sarima': 0.8},  # 趋势市场
                            'default': {'sarima': 0.6, 'garch': 0.5, 'multi_factor': 0.3}  # 默认市场
                        },
                        # 其他可选组合
                        'alternative_combinations': {
                            # 组合2: 保守型 (侧重稳定模型)
                            'conservative': {
                                'stable_trend': {'sarima': 0.9},
                                'high_volatility': {'garch': 1.0},
                                'high_volatility_nonlinear': {'garch': 1.0, 'xgboost': 0.5},
                                'trending': {'sarima': 0.9},
                                'default': {'sarima': 0.7, 'garch': 0.6}
                            },
                            # 组合3: 激进型 (侧重预测能力)
                            'aggressive': {
                                'stable_trend': {'sarima': 0.8},
                                'high_volatility': {'garch': 1.0, 'xgboost': 0.2},
                                'high_volatility_nonlinear': {'garch': 1.0, 'xgboost': 0.9, 'multi_factor': 0.7},
                                'trending': {'sarima': 0.2},
                                'default': {'garch': 0.6, 'xgboost': 0.3}
                            }
                        },
                        'selected_model_count': 3,  # 选择模型数量
                        'confidence_threshold': 0.6,  # 置信度阈值

                        # 模型统计信息 (用于参考)
                        'model_statistics': {
                            'usage_rate': {
                                'garch': '36.8%',  # 主力模型，表现稳定
                                'multi_factor': '15.2%',  # 多因子表现良好
                                'xgboost': '10.9%',  # 非线性市场有效
                                'sarima': '2.7%',  # 大幅降低权重
                            },
                            'removed_models': [
                                {
                                    'model': 'SARIMA',
                                    'reason': '被ARIMA全面超越，相同状态ARIMA入选率31.9% vs SARIMA 7.2%',
                                    'original_weight': 1.6,
                                    'optimized_weight': '0.0-0.7'
                                }
                            ]
                        }
                    }
                },
                'correction': {
                    'error_correction': {
                        'enabled': True,  # 是否启用误差校正
                        'window_size': 50,  # 校正窗口大小
                        'correction_strength': 0.7  # 校正强度
                    }
                },
                'probabilistic': {
                    'uncertainty_quantification': {
                        'enabled': True,  # 是否启用不确定性量化
                        'n_samples': 1000,  # 采样次数
                        'confidence_level': 0.95,  # 置信水平
                        'distribution': 'normal'  # 分布类型
                    }
                },
                'monitoring': {
                    'accuracy_tracking': {
                        'enabled': True,  # 是否启用精度跟踪
                        'evaluation_frequency': 'daily'  # 评估频率
                    },
                    'ab_testing': {
                        'enabled': True,  # 是否启用A/B测试
                        'test_period': 30  # 测试周期
                    }
                },
                'simulator': {
                    'deterministic_mode': True,  # 确定性模式
                    'random_seed': 42,  # 随机种子
                    'simulation_days': 10,  # 模拟天数
                    # ==================== 新增的模拟器参数 ====================
                    'garch_volatility_persistence': 0.9,  # GARCH波动率持续性
                    'multi_factor_n_features': 10,  # 多因子特征数
                    'xgboost_n_features': 13,  # XGBoost特征数

                    'high_volatility_threshold': 0.04,  # 高波动率阈值
                    'low_volatility_threshold': 0.015,  # 低波动率阈值
                    'strong_trend_threshold': 0.3,  # 强趋势阈值
                    'stable_trend_volatility_threshold': 0.02,  # 稳定趋势波动率阈值

                    'rsi_oversold_threshold': 30.0,  # RSI超卖阈值
                    'rsi_overbought_threshold': 70.0,  # RSI超买阈值
                    'rsi_normal_min': 30.0,  # RSI正常最小值
                    'rsi_normal_max': 70.0,  # RSI正常最大值

                    'technical_score_high_threshold': 0.7,  # 技术分数高阈值
                    'technical_score_low_threshold': 0.3,  # 技术分数低阈值
                    'technical_score_neutral': 0.5,  # 技术分数中性值

                    'market_adaptation_boost': 1.2,  # 市场适应提升因子
                    'high_vol_garch_boost': 1.3,  # 高波动GARCH提升因子
                    'low_vol_sarima_boost': 1.2,  # 低波动SARIMA提升因子
                    'high_tech_xgboost_boost': 1.1,  # 高技术XGBoost提升因子
                },
                'reporting': {
                    'output_format': 'html',  # 输出格式
                    'auto_open_html': True,  # 自动打开HTML
                    'save_path': 'results',  # 保存路径
                    'timestamp_format': '%Y%m%d_%H%M%S'  # 时间戳格式
                },
                # 模型配置
                'sarima': {
                    'base_score': 6.0,  # 基础评分
                    'enabled': True,  # 是否启用
                    'order': [1, 1, 1],  # ARIMA阶数
                    'seasonal_order': [1, 1, 1, 12],  # 季节阶数
                    'window_size': 50,  # 窗口大小
                    'prediction_horizon': 5,  # 预测周期
                    'weight_strategy': 'equal',  # 权重策略
                    'p': 1,  # 自回归阶数
                    'd': 1,  # 差分阶数
                    'q': 1  # 移动平均阶数
                },
                'garch': {
                    'base_score': 6.0,  # 基础评分
                    'enabled': True,  # 是否启用
                    'order': [1, 1],  # GARCH阶数
                    'p': 1,  # GARCH项阶数
                    'q': 1,  # ARCH项阶数
                    'window_size': 50,  # 窗口大小
                    'prediction_horizon': 5,  # 预测周期
                    'weight_strategy': 'recent_weighted',  # 权重策略
                    'distribution': 't'  # 分布类型
                },
                'xgboost': {
                    'base_score': 6.0,  # 基础评分
                    'enabled': True,  # 是否启用
                    'n_estimators': 50,  # 树的数量
                    'max_depth': 4,  # 最大深度
                    'learning_rate': 0.1,  # 学习率
                    'n_features': 13,  # 特征数量
                    'window_size': 50,  # 窗口大小
                    'prediction_horizon': 5,  # 预测周期
                    'weight_strategy': 'equal'  # 权重策略
                },
                'multi_factor': {
                    'base_score': 6.0,  # 基础评分
                    'enabled': True,  # 是否启用
                    'n_features': 15,  # 特征数量
                    'window_size': 150,  # 窗口大小
                    'prediction_horizon': 10,  # 预测周期
                    'weight_strategy': 'equal'  # 权重策略
                },
                'technical_features': {
                    'enabled': True,  # 是否启用技术特征提取
                    'include_indicators': True,  # 是否包含技术指标
                    'include_golden_triangle': True,  # 是否包含金三角指标
                    'include_kdj': True,  # 是否包含KDJ指标
                    'include_normality': True,  # 是否包含正态分布分析
                    'include_volume_analysis': True,  # 是否包含成交量分析
                    'return_dataframe': True,  # 是否返回DataFrame格式
                },
                'golden_triangle': {
                        'short_period': 5,  # 短期均线周期
                        'medium_period': 10,  # 中期均线周期
                        'long_period': 20,  # 长期均线周期
                        'window': 10,  # 模式识别窗口
                        'overbought_threshold': 80,  # 超买阈值
                        'oversold_threshold': 20,  # 超卖阈值
                        'extreme_overbought_threshold': 90,  # 极端超买阈值
                        'extreme_oversold_threshold': 10,  # 极端超卖阈值
                        'lookback_period': 20,  # 背离检测回溯周期
                        'strength_window': 20,  # 趋势强度计算窗口
                        'divergence_window': 10,  # 发散度计算窗口
                    },

                    'kdj': {
                        'n': 9,  # RSV计算周期
                        'm1': 3,  # K值平滑周期
                        'm2': 3,  # D值平滑周期
                        'overbought_threshold': 80,  # 超买阈值
                        'oversold_threshold': 20,  # 超卖阈值
                        'extreme_overbought_threshold': 90,  # 极端超买阈值
                        'extreme_oversold_threshold': 10,  # 极端超卖阈值
                        'lookback_period': 20,  # 背离检测回溯周期
                        'volatility_window': 20,  # 波动率计算窗口
                        'position_window': 50,  # 位置分析窗口
                        'trend_window': 20,  # 趋势分析窗口
                        'pattern_window': 10,  # 模式识别窗口
                    },

                    'normality': {
                        'jarque_bera_min_samples': 4,  # Jarque-Bera检验最小样本数
                        'shapiro_wilk_min_samples': 3,  # Shapiro-Wilk检验最小样本数
                        'shapiro_wilk_max_samples': 5000,  # Shapiro-Wilk检验最大样本数
                        'normal_skewness_threshold': 0.5,  # 正常偏度阈值
                        'normal_kurtosis_threshold': 1.0,  # 正常峰度阈值
                        'qq_quantiles': 100,  # Q-Q图分位数数量
                        'deviation_z_scores': [1, 2, 3],  # 偏离分析Z分数
                        'fat_tail_threshold': -0.01,  # 厚尾判断阈值
                        'thin_tail_threshold': 0.01,  # 薄尾判断阈值
                        'normality_window': 20,  # 滚动正态性分析窗口
                    },

                    'volume_analysis': {
                        'high_volume_period': 1500,  # 高量柱检测周期
                        'volume_ma_period': 5,  # 量比计算均线周期
                        'winner_window': 20,  # 获利盘比例计算窗口
                        'ma_periods': [20, 60, 120],  # 均线位置判断周期
                        'volume_double_threshold': 2.0,  # 成交量翻倍阈值
                        'volume_shrink_threshold': 0.5,  # 成交量萎缩阈值
                        'volume_min_ratio': 0.02,  # 成交量最小比例阈值

                        'volume_category_thresholds': {
                            '严重缩量': [0.001, 0.5],  # 严重缩量范围
                            '缩量': [0.5, 0.8],  # 缩量范围
                            '正常量比': [0.8, 1.5],  # 正常量比范围
                            '温和放量': [1.5, 2.5],  # 温和放量范围
                            '明显放量': [2.5, 5.0],  # 明显放量范围
                            '剧烈放量': [5.0, 10.0],  # 剧烈放量范围
                            '反向操作': [10.0, 999999.0]  # 反向操作范围
                        },


                    'indicators': {
                        'rsi_period': 14,  # RSI计算周期
                        'macd_fast': 12,  # MACD快速移动平均周期
                        'macd_slow': 26,  # MACD慢速移动平均周期
                        'macd_signal': 9,  # MACD信号线周期
                        'bollinger_window': 20,  # 布林带窗口大小
                        'bollinger_num_std': 2,  # 布林带标准差倍数

                        'volume_ma_4': 4,  # 4日成交量均线
                        'volume_ma_11': 11,  # 11日成交量均线
                        'volume_ma_117': 117,  # 117日成交量均线
                    }
                }
            }
        }

class Config:
    """兼容旧版本的Config类"""

    def __init__(
            self,
            stock_code: str,
            stock_name: str,
            days_to_simulate: int = 10,
            recommendation_file_path: str = "backtest_results/parameter_recommendations/parameter_recommendations.txt"
    ):
        """初始化配置（保持与旧版本兼容）"""
        self.config_manager = ConfigManager()
        self.full_config = self.config_manager.load_config(
            stock_code=stock_code,
            stock_name=stock_name,
            days_to_simulate=days_to_simulate,
            recommendation_file_path=recommendation_file_path
        )

        # 设置兼容属性
        self.stock_code = self.full_config.stock.code
        self.stock_name = self.full_config.stock.name
        self.days_to_simulate = self.full_config.stock.simulation_days

        self.base_dir = self.full_config.base_dir
        self.data_dir = os.path.join(self.base_dir, self.full_config.data.cache_dir)
        self.result_dir = os.path.join(self.base_dir, self.full_config.data.results_dir)

        self.end_date = self.full_config.data.end_date.strftime('%Y-%m-%d')
        self.start_date = self.full_config.data.start_date.strftime('%Y-%m-%d')
        # 打印模型参数摘要
        self._print_model_config_summary()

    def _print_model_config_summary(self):
        """打印模型配置摘要"""
        for model_name in self.full_config.get_model_names():
            model_config = getattr(self.full_config, model_name)

            # 获取关键参数
            key_params = []
            if hasattr(model_config, 'window_size'):
                key_params.append(f"window_size={model_config.window_size}")
            if hasattr(model_config, 'prediction_horizon'):
                key_params.append(f"prediction_horizon={model_config.prediction_horizon}")
            if hasattr(model_config, 'weight_strategy'):
                key_params.append(f"weight_strategy={model_config.weight_strategy}")

    @property
    def cache_file_path(self):
        """缓存文件路径"""
        return self.full_config.cache_file_path

    @property
    def result_file_prefix(self):
        """结果文件前缀"""
        return self.full_config.result_file_prefix

    def get_full_config(self) -> FullConfig:
        """获取完整配置对象"""
        return self.full_config

    def update_recommendations(self):
        """更新推荐参数（热更新）"""
        self.full_config.apply_latest_recommendations()
        self._print_model_config_summary()

    def to_dict(self):
        """转换为字典"""
        return self.full_config.to_dict()

#==================== 导出配置功能 ====================

def export_config_to_yaml_with_comments(config_obj, export_path="config_export_with_comments.yaml"):
    """
    将配置对象导出为YAML文件，并添加详细的参数说明

    Args:
        config_obj: 配置对象（FullConfig或Config实例）
        export_path: 导出文件路径
    """
    try:
        # 获取配置字典
        if hasattr(config_obj, 'to_dict'):
            config_dict = config_obj.to_dict()
        elif hasattr(config_obj, 'get_full_config'):
            config_dict = config_obj.get_full_config().to_dict()
        else:
            print("❌ 不支持的配置对象类型")
            return False

        # 构建完整的配置结构
        full_config = {'stock_system': config_dict}

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(export_path)), exist_ok=True)

        # 使用带注释的YAML导出
        with open(export_path, 'w', encoding='utf-8') as f:
            safe_yaml_dump_with_comments(full_config, f)

        print(f"✅ 带详细注释的配置已成功导出到: {export_path}")
        print(f"   文件大小: {os.path.getsize(export_path)} 字节")
        return True

    except Exception as e:
        print(f"❌ 导出带注释的配置失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_current_config_with_comments():
    """导出当前加载的配置到YAML文件，并添加详细的参数说明"""
    try:
        # 获取配置管理器实例
        config_manager = ConfigManager()

        if config_manager.config is None:
            print("❌ 当前没有加载的配置，请先加载配置")
            return False

        # 生成带时间戳的导出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"config_export_with_comments_{timestamp}.yaml"

        return export_config_to_yaml_with_comments(config_manager.config, export_path)

    except Exception as e:
        print(f"❌ 导出当前带注释的配置失败: {e}")
        return False

def get_param_helper():
    """
    参数助手函数（兼容性函数）
    返回一个参数助手对象，用于获取特征参数
    """

    class ParamHelper:
        def __init__(self):
            self.config_loader = get_config_loader()

        def get_feature_param(self, param_loader, feature_name, param_name, default_value, config=None):
            """
            获取特征参数
            参数优先级：配置文件 > 推荐参数 > 默认值
            """
            # 1. 首先从传入的配置中获取
            if config and hasattr(config, 'features'):
                # 这里需要根据你的配置结构来获取参数
                # 示例：从config.features.technical_indicators中获取
                if hasattr(config.features, 'technical_indicators'):
                    tech_indicators = config.features.technical_indicators
                    param_key = f"{feature_name.lower()}_{param_name}"
                    if param_key in tech_indicators:
                        return tech_indicators[param_key]

            # 2. 从配置文件获取
            if param_loader:
                try:
                    # 假设param_loader有get_recommended_value方法
                    return param_loader.get_recommended_value(
                        f'features.{feature_name}', param_name, default_value
                    )
                except:
                    pass

            # 3. 返回默认值
            return default_value

    return ParamHelper()

# ==================== 测试函数 ====================
def test_param_helper():
    """测试参数助手函数"""
    helper = get_param_helper()
    print("✅ 参数助手函数创建成功")

    # 测试获取参数
    rsi_period = helper.get_feature_param(None, 'rsi', 'period', 14, None)
    print(f"RSI周期默认值: {rsi_period}")

    return True

def get_simulator_param(config, param_name, default_value):
    """
    获取模拟器参数

    Args:
        config: 配置对象
        param_name: 参数名称
        default_value: 默认值

    Returns:
        参数值
    """
    if hasattr(config, 'simulator'):
        return getattr(config.simulator, param_name, default_value)
    elif isinstance(config, dict):
        if 'simulator' in config:
            return config['simulator'].get(param_name, default_value)
        else:
            return config.get(param_name, default_value)
    else:
        return default_value

if __name__ == "__main__":
    test_param_helper()
    print("测试动态参数推荐配置系统")
    print("=" * 48)

    # 首先测试YAML加载
    print("测试加载config.yaml...")
    with open("config.yaml", 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用新的安全加载器
    try:
        config_data = safe_yaml_load(content)
        if config_data:
            print("✅ YAML加载成功")

            # 提取主要配置
            system_config = config_data.get('stock_system', {})

            # 打印配置摘要
            print("\n📊 配置摘要:")
            if 'stock' in system_config:
                stock = system_config['stock']
                print(f"  股票: {stock.get('name', '未知')}({stock.get('code', '未知')})")
                print(f"  模拟天数: {stock.get('simulation_days', '未知')}")

            # 测试模型配置
            models = ['sarima', 'xgboost', 'garch']
            print("\n🎯 模型配置状态:")
            for model in models:
                if model in system_config:
                    cfg = system_config[model]
                    enabled = cfg.get('enabled', False)
                    status = "✅ 启用" if enabled else "❌ 禁用"
                    print(f"  {model.upper():10s}: {status}")

            # 测试模拟器配置
            if 'simulator' in system_config:
                sim = system_config['simulator']
                print(f"\n🎮 模拟器:")
                print(f"  随机种子: {sim.get('random_seed', '未知')}")
                print(f"  确定性模式: {sim.get('deterministic_mode', '未知')}")

    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 48)

    # 测试完整的配置加载
    print("测试完整配置加载...")
    try:
        config_manager = ConfigManager()

        # 指定推荐文件路径
        recommendation_path = "backtest_results/parameter_recommendations/parameter_recommendations.txt"

        config = config_manager.load_config(
            config_path="config.yaml",
            stock_code="000001",
            stock_name="平安银行",
            days_to_simulate=20,
            recommendation_file_path=recommendation_path
        )

        print("✅ 完整配置加载成功")

        # 验证配置
        print(f"\n📊 股票配置:")
        print(f"  代码: {config.stock.code}")
        print(f"  名称: {config.stock.name}")
        print(f"  模拟天数: {config.stock.simulation_days}")

        print(f"\n📅 数据配置:")
        print(f"  开始日期: {config.data.start_date}")
        print(f"  结束日期: {config.data.end_date}")

        print(f"\n🎯 启用的模型:")
        for model_name in ['sarima', 'xgboost', 'garch','multi_factor']:
            model_config = getattr(config, model_name)
            if model_config.enabled:
                print(f"  {model_name.upper()}")

        # ==================== 新增：测试导出带注释的配置 ====================
        print("\n" + "=" * 48)
        print("测试导出带注释的配置功能...")
        # 方法2：导出带注释的配置
        if export_current_config_with_comments():
            print("✅ 当前配置导出成功（带注释版本）")
    except Exception as e:
        print(f"❌ 完整配置加载失败: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 48)