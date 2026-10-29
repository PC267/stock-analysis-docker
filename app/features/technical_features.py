# 技术指标特征提取
import numpy as np
import pandas as pd
from scipy import stats  # 新增导入
from scipy.stats import jarque_bera, shapiro  # 新增导入

try:
    from config_loader import get_param_helper
except ImportError:
    get_param_helper = None


class TechnicalFeatureExtractor:
    """技术指标特征提取器"""

    def __init__(self, param_loader=None, config=None):
        self.param_loader = param_loader
        self.config = config

        # 修复这里的逻辑
        if param_loader and get_param_helper is not None:
            try:
                self.param_helper = get_param_helper()
            except:
                self.param_helper = None
        else:
            self.param_helper = None

    def _validate_numeric_param(self, value, param_name, default_value, min_val=None, max_val=None):
        """验证数值参数"""
        try:
            # 如果是Mock对象
            if hasattr(value, 'return_value'):
                value = value.return_value
            elif callable(value):
                value = value()

            # 转换为浮点数
            num_value = float(value)

            # 范围检查
            if min_val is not None and num_value < min_val:
                print(f"警告：参数{param_name}={num_value}小于最小值{min_val}，使用默认值{default_value}")
                return default_value
            if max_val is not None and num_value > max_val:
                print(f"警告：参数{param_name}={num_value}大于最大值{max_val}，使用默认值{default_value}")
                return default_value

            return num_value
        except Exception as e:
            print(f"验证参数{param_name}时出错: {e}，使用默认值{default_value}")
            return default_value

    def _get_param_with_fallback(self, indicator, param_name, default_value):
        """带多层回退机制的参数获取"""
        sources = [
            self._try_get_from_param_helper,
            self._try_get_from_param_loader,
            self._try_get_from_config,
            lambda i, p, d: default_value
        ]

        for source in sources:
            try:
                value = source(indicator, param_name, default_value)
                if value is not None and value != default_value:
                    return value
            except:
                continue

        return default_value

    def _try_get_from_param_helper(self, indicator, param_name, default_value):
        """尝试从param_helper获取参数"""
        if self.param_helper:
            try:
                return self.param_helper.get_feature_param(
                    self.param_loader, indicator, param_name, default_value, self.config
                )
            except:
                pass
        return default_value

    def _try_get_from_param_loader(self, indicator, param_name, default_value):
        """尝试从param_loader获取参数"""
        if self.param_loader:
            try:
                # 获取指标配置
                if hasattr(self.param_loader, 'get'):
                    indicator_config = self.param_loader.get(indicator, {})
                else:
                    # 尝试作为字典访问
                    indicator_config = getattr(self.param_loader, indicator, {})

                if param_name in indicator_config:
                    return indicator_config[param_name]
            except:
                pass
        return default_value

    def _try_get_from_config(self, indicator, param_name, default_value):
        """尝试从config获取参数"""
        if self.config:
            try:
                key = f"{indicator}.{param_name}"
                if key in self.config:
                    return self.config[key]
            except:
                pass
        return default_value

    def _get_param(self, indicator, param_name, default_value):
        """获取配置参数"""
        try:
            # 使用回退机制获取参数值
            raw_value = self._get_param_with_fallback(indicator, param_name, default_value)

            # 确保数值参数是合适的类型
            if param_name in ['period', 'window', 'fast', 'slow', 'signal', 'n', 'm1', 'm2',
                              'short_period', 'medium_period', 'long_period', 'short', 'medium', 'long']:
                # 验证并转换为整数
                validated_value = self._validate_numeric_param(
                    raw_value, f'{indicator}.{param_name}', default_value, min_val=1
                )
                return int(validated_value)

            # 处理列表类型的参数（如ma_position的periods）
            elif param_name == 'periods':
                try:
                    if isinstance(raw_value, (list, tuple)):
                        # 确保列表中的值都是整数
                        return [int(self._validate_numeric_param(v, f'{indicator}.{param_name}[{i}]', v))
                                for i, v in enumerate(raw_value)]
                    elif isinstance(raw_value, str):
                        # 尝试解析字符串列表
                        import ast
                        parsed = ast.literal_eval(raw_value)
                        if isinstance(parsed, (list, tuple)):
                            return [int(self._validate_numeric_param(v, f'{indicator}.{param_name}[{i}]', v))
                                    for i, v in enumerate(parsed)]
                except:
                    print(f"警告：无法解析参数{indicator}.{param_name}={raw_value}，使用默认值{default_value}")
                return default_value

            # 处理浮点数参数（如bollinger_bands的num_std）
            elif param_name in ['num_std', 'overbought_threshold', 'oversold_threshold',
                                'confidence_level', 'alpha', 'beta']:
                return self._validate_numeric_param(raw_value, f'{indicator}.{param_name}', default_value)

            return raw_value

        except Exception as e:
            print(f"获取参数{indicator}.{param_name}时出错: {e}")

        return default_value

    # ========== 实例方法（支持配置参数）==========

    def calculate_rsi(self, prices):
        """计算RSI，使用配置参数"""
        period = self._get_param('rsi', 'period', 14)
        return self.calculate_rsi_static(prices, period)

    # ========== 静态方法（无需实例化）==========

    @staticmethod
    def calculate_rsi_static(prices, period=14):
        """计算相对强弱指数（静态方法版本）"""
        # 确保周期是正整数
        period = max(1, int(period))

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices, fast=14, slow=53, signal=5):
        """计算MACD指标"""
        # 确保参数是正整数
        fast = max(1, int(fast))
        slow = max(2, int(slow))
        signal = max(1, int(signal))

        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """计算布林带"""
        # 确保参数有效
        window = max(1, int(window))
        num_std = max(0.5, float(num_std))

        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    # ========== 新增：均线金三角指标 ==========

    @staticmethod
    def calculate_golden_triangle(close, short_period=5, medium_period=10, long_period=20):
        """
        计算金三角指标（5日、10日、20日均线）
        """
        # 确保参数是正整数
        short_period = max(1, int(short_period))
        medium_period = max(2, int(medium_period))
        long_period = max(3, int(long_period))

        # 计算各周期均线
        ma_short = close.rolling(window=short_period).mean()
        ma_medium = close.rolling(window=medium_period).mean()
        ma_long = close.rolling(window=long_period).mean()

        # 初始化信号序列 - 使用float类型
        signal = pd.Series(0.0, index=close.index, dtype=float)
        triangle_type = pd.Series('无信号', index=close.index)
        ma_relation = pd.Series('未知', index=close.index)

        # 计算金三角条件
        start_idx = max(short_period, medium_period, long_period)
        for i in range(start_idx, len(close)):
            # 金三角条件：短期均线 > 中期均线 > 长期均线，且短期均线上穿中期均线或长期均线
            golden_condition = (
                    ma_short.iloc[i] > ma_medium.iloc[i] and
                    ma_medium.iloc[i] > ma_long.iloc[i] and
                    (
                        # 短期上穿中期
                            (ma_short.iloc[i] > ma_medium.iloc[i] and ma_short.iloc[i - 1] <= ma_medium.iloc[i - 1]) or
                            # 短期上穿长期
                            (ma_short.iloc[i] > ma_long.iloc[i] and ma_short.iloc[i - 1] <= ma_long.iloc[i - 1]) or
                            # 中期上穿长期
                            (ma_medium.iloc[i] > ma_long.iloc[i] and ma_medium.iloc[i - 1] <= ma_long.iloc[i - 1])
                    )
            )

            # 死三角条件：短期均线 < 中期均线 < 长期均线，且短期均线下穿中期均线或长期均线
            death_condition = (
                    ma_short.iloc[i] < ma_medium.iloc[i] and
                    ma_medium.iloc[i] < ma_long.iloc[i] and
                    (
                        # 短期下穿中期
                            (ma_short.iloc[i] < ma_medium.iloc[i] and ma_short.iloc[i - 1] >= ma_medium.iloc[i - 1]) or
                            # 短期下穿长期
                            (ma_short.iloc[i] < ma_long.iloc[i] and ma_short.iloc[i - 1] >= ma_long.iloc[i - 1]) or
                            # 中期下穿长期
                            (ma_medium.iloc[i] < ma_long.iloc[i] and ma_medium.iloc[i - 1] >= ma_long.iloc[i - 1])
                    )
            )

            # 设置信号
            if golden_condition:
                signal.iloc[i] = 1.0
                triangle_type.iloc[i] = '金三角'
            elif death_condition:
                signal.iloc[i] = -1.0
                triangle_type.iloc[i] = '死三角'
            else:
                signal.iloc[i] = 0.0
                triangle_type.iloc[i] = '无信号'

            # 设置均线关系状态
            if ma_short.iloc[i] > ma_medium.iloc[i] > ma_long.iloc[i]:
                ma_relation.iloc[i] = '多头排列'
            elif ma_short.iloc[i] < ma_medium.iloc[i] < ma_long.iloc[i]:
                ma_relation.iloc[i] = '空头排列'
            elif ma_short.iloc[i] > ma_long.iloc[i] > ma_medium.iloc[i]:
                ma_relation.iloc[i] = '短期强势'
            elif ma_medium.iloc[i] > ma_long.iloc[i] > ma_short.iloc[i]:
                ma_relation.iloc[i] = '中期强势'
            elif ma_long.iloc[i] > ma_medium.iloc[i] > ma_short.iloc[i]:
                ma_relation.iloc[i] = '长期弱势'
            else:
                ma_relation.iloc[i] = '震荡整理'

        return {
            'ma_short': ma_short,
            'ma_medium': ma_medium,
            'ma_long': ma_long,
            'golden_triangle_signal': signal,
            'triangle_type': triangle_type,
            'ma_relation': ma_relation,
            'ma_spread_short_medium': ma_short - ma_medium,
            'ma_spread_medium_long': ma_medium - ma_long,
            'ma_spread_short_long': ma_short - ma_long,
        }

    @staticmethod
    def calculate_golden_triangle_strength(ma_short, ma_medium, ma_long):
        """
        计算金三角强度指标
        """
        # 均线发散度（衡量金三角的强弱）
        divergence = (ma_short - ma_medium).abs() + (ma_medium - ma_long).abs()

        # 均线角度（近似计算斜率）- 使用float类型
        short_angle = ma_short.diff().rolling(window=5).mean().astype(float)
        medium_angle = ma_medium.diff().rolling(window=10).mean().astype(float)
        long_angle = ma_long.diff().rolling(window=20).mean().astype(float)

        # 角度一致性 - 使用float类型
        angle_alignment = ((short_angle > 0) & (medium_angle > 0) & (long_angle > 0)).astype(float)

        # 强度评分（0-100）- 使用float类型
        strength_score = pd.Series(50.0, index=ma_short.index, dtype=float)

        start_idx = max(20, len(ma_short) - 100)
        for i in range(start_idx, len(ma_short)):
            # 基础分
            score = 50.0

            # 多头排列加分
            if ma_short.iloc[i] > ma_medium.iloc[i] > ma_long.iloc[i]:
                score += 20.0

                # 发散度越大，分数越高
                if ma_long.iloc[i] != 0:
                    divergence_ratio = divergence.iloc[i] / abs(ma_long.iloc[i]) * 100.0
                    score += min(divergence_ratio, 15.0)

                # 角度向上加分
                if short_angle.iloc[i] > 0 and medium_angle.iloc[i] > 0 and long_angle.iloc[i] > 0:
                    score += 15.0
                elif short_angle.iloc[i] > 0 and medium_angle.iloc[i] > 0:
                    score += 10.0
                elif short_angle.iloc[i] > 0:
                    score += 5.0

            # 空头排列减分
            elif ma_short.iloc[i] < ma_medium.iloc[i] < ma_long.iloc[i]:
                score -= 20.0

                # 角度向下减分
                if short_angle.iloc[i] < 0 and medium_angle.iloc[i] < 0 and long_angle.iloc[i] < 0:
                    score -= 15.0
                elif short_angle.iloc[i] < 0 and medium_angle.iloc[i] < 0:
                    score -= 10.0
                elif short_angle.iloc[i] < 0:
                    score -= 5.0

            # 限制分数在0-100之间
            strength_score.iloc[i] = max(0.0, min(100.0, score))

        # 强度等级分类
        strength_level = pd.Series('中等', index=strength_score.index)
        strength_level[strength_score < 30] = '弱'
        strength_level[(strength_score >= 30) & (strength_score < 50)] = '中等偏弱'
        strength_level[(strength_score >= 50) & (strength_score < 70)] = '中等偏强'
        strength_level[strength_score >= 70] = '强'

        return {
            'triangle_divergence': divergence.astype(float),
            'short_ma_angle': short_angle,
            'medium_ma_angle': medium_angle,
            'long_ma_angle': long_angle,
            'angle_alignment': angle_alignment,
            'triangle_strength_score': strength_score,
            'triangle_strength_level': strength_level
        }

    @staticmethod
    def calculate_golden_triangle_pattern(ma_short, ma_medium, ma_long, window=10):
        """
        识别金三角模式
        """
        window = max(1, int(window))
        patterns = pd.Series('无模式', index=ma_short.index)

        for i in range(window, len(ma_short)):
            # 避免除以零
            if ma_medium.iloc[i] == 0 or ma_long.iloc[i] == 0:
                continue

            # 检查是否形成标准金三角
            if (ma_short.iloc[i] > ma_medium.iloc[i] > ma_long.iloc[i] and
                    ma_short.iloc[i - 5] <= ma_medium.iloc[i - 5] and
                    ma_medium.iloc[i - 5] <= ma_long.iloc[i - 5]):
                patterns.iloc[i] = '标准金三角'

            # 检查是否形成强势金三角（均线快速发散）
            elif (ma_short.iloc[i] > ma_medium.iloc[i] > ma_long.iloc[i] and
                  abs(ma_medium.iloc[i]) > 0.0001 and
                  (ma_short.iloc[i] - ma_medium.iloc[i]) > (ma_medium.iloc[i] - ma_long.iloc[i]) * 1.5):
                patterns.iloc[i] = '强势金三角'

            # 检查是否形成弱势金三角（均线收敛）
            elif (ma_short.iloc[i] > ma_medium.iloc[i] > ma_long.iloc[i] and
                  abs(ma_medium.iloc[i]) > 0.0001 and
                  (ma_short.iloc[i] - ma_medium.iloc[i]) < (ma_medium.iloc[i] - ma_long.iloc[i]) * 0.5):
                patterns.iloc[i] = '弱势金三角'

            # 检查是否形成死三角
            elif (ma_short.iloc[i] < ma_medium.iloc[i] < ma_long.iloc[i] and
                  ma_short.iloc[i - 5] >= ma_medium.iloc[i - 5] and
                  ma_medium.iloc[i - 5] >= ma_long.iloc[i - 5]):
                patterns.iloc[i] = '标准死三角'

            # 检查均线粘合后发散（使用百分比判断）
            elif (abs(ma_short.iloc[i] - ma_medium.iloc[i]) / abs(ma_medium.iloc[i]) < 0.01 and
                  abs(ma_medium.iloc[i] - ma_long.iloc[i]) / abs(ma_long.iloc[i]) < 0.01 and
                  ma_short.iloc[i - 3] - ma_medium.iloc[i - 3] < 0 and
                  ma_medium.iloc[i - 3] - ma_long.iloc[i - 3] < 0):
                patterns.iloc[i] = '粘合向上突破'

            elif (abs(ma_short.iloc[i] - ma_medium.iloc[i]) / abs(ma_medium.iloc[i]) < 0.01 and
                  abs(ma_medium.iloc[i] - ma_long.iloc[i]) / abs(ma_long.iloc[i]) < 0.01 and
                  ma_short.iloc[i - 3] - ma_medium.iloc[i - 3] > 0 and
                  ma_medium.iloc[i - 3] - ma_long.iloc[i - 3] > 0):
                patterns.iloc[i] = '粘合向下突破'

        return {
            'triangle_pattern': patterns,
            'pattern_changed': (patterns != patterns.shift(1)).astype(float)
        }

    @staticmethod
    def extract_golden_triangle_features(close, short_period=5, medium_period=10, long_period=20):
        """
        提取所有金三角相关特征
        """
        try:
            # 确保参数有效
            short_period = max(1, int(short_period))
            medium_period = max(2, int(medium_period))
            long_period = max(3, int(long_period))

            # 计算基础金三角指标
            triangle_base = TechnicalFeatureExtractor.calculate_golden_triangle(
                close, short_period, medium_period, long_period
            )

            features = {}
            features.update({f'triangle_{k}': v for k, v in triangle_base.items()})

            # 计算金三角强度
            strength_features = TechnicalFeatureExtractor.calculate_golden_triangle_strength(
                triangle_base['ma_short'],
                triangle_base['ma_medium'],
                triangle_base['ma_long']
            )
            features.update({f'triangle_{k}': v for k, v in strength_features.items()})

            # 识别金三角模式
            pattern_features = TechnicalFeatureExtractor.calculate_golden_triangle_pattern(
                triangle_base['ma_short'],
                triangle_base['ma_medium'],
                triangle_base['ma_long']
            )
            features.update({f'triangle_{k}': v for k, v in pattern_features.items()})

            # 计算金三角持续时间
            signal = triangle_base['golden_triangle_signal']
            duration = pd.Series(0.0, index=signal.index, dtype=float)
            current_duration = 0

            for i in range(1, len(signal)):
                if signal.iloc[i] == 1:
                    if signal.iloc[i - 1] == 1:
                        current_duration += 1
                    else:
                        current_duration = 1
                elif signal.iloc[i] == -1:
                    if signal.iloc[i - 1] == -1:
                        current_duration += 1
                    else:
                        current_duration = 1
                else:
                    current_duration = 0

                duration.iloc[i] = float(current_duration)

            features['triangle_duration'] = duration

            # 金三角综合信号
            composite_signal = pd.Series(0.0, index=signal.index, dtype=float)
            for i in range(len(signal)):
                if not pd.isna(strength_features['triangle_strength_score'].iloc[i]):
                    composite_signal.iloc[i] = (
                            signal.iloc[i] * 0.6 +
                            (strength_features['triangle_strength_score'].iloc[i] / 100.0 - 0.5) * 2.0
                    )

            features['triangle_composite_signal'] = composite_signal

            return features

        except Exception as e:
            print(f"金三角特征提取失败: {e}")
            # 返回空特征
            return {
                'triangle_ma_short': pd.Series(np.nan, index=close.index),
                'triangle_ma_medium': pd.Series(np.nan, index=close.index),
                'triangle_ma_long': pd.Series(np.nan, index=close.index),
                'triangle_golden_triangle_signal': pd.Series(0.0, index=close.index, dtype=float),
                'triangle_triangle_type': pd.Series('无信号', index=close.index),
                'triangle_ma_relation': pd.Series('未知', index=close.index),
            }

    # ========== 新增：KDJ分析指标 ==========

    @staticmethod
    def calculate_kdj(high, low, close, n=20, m1=3, m2=3):
        """
        计算KDJ指标
        """
        # 确保参数是正整数
        n = max(1, int(n))
        m1 = max(1, int(m1))
        m2 = max(1, int(m2))

        # 计算最低价的最低值和最高价的最高值
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()

        # 计算RSV（未成熟随机值）
        rsv = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        rsv = rsv.replace([np.inf, -np.inf], 0).fillna(50)

        # 计算K值：RSV的m1日指数移动平均
        K = rsv.ewm(alpha=1 / m1, adjust=False).mean()

        # 计算D值：K值的m2日指数移动平均
        D = K.ewm(alpha=1 / m2, adjust=False).mean()

        # 计算J值：3K - 2D
        J = 3 * K - 2 * D

        return K, D, J

    @staticmethod
    def kdj_cross_signal(K, D):
        """
        生成KDJ金叉死叉信号
        """
        signal = pd.Series(0, index=K.index)
        cross_type = pd.Series('无信号', index=K.index)

        # 金叉：K上穿D
        golden_cross = (K > D) & (K.shift(1) <= D.shift(1))
        # 死叉：K下穿D
        death_cross = (K < D) & (K.shift(1) >= D.shift(1))

        signal[golden_cross] = 1
        cross_type[golden_cross] = '金叉'

        signal[death_cross] = -1
        cross_type[death_cross] = '死叉'

        return signal, cross_type

    @staticmethod
    def kdj_overbought_oversold(K, D, J,
                                overbought_threshold=80,
                                oversold_threshold=20):
        """
        判断KDJ超买超卖状态
        """
        # 确保阈值有效
        overbought_threshold = max(50, min(100, float(overbought_threshold)))
        oversold_threshold = max(0, min(50, float(oversold_threshold)))

        # 超买状态
        overbought_K = K > overbought_threshold
        overbought_D = D > overbought_threshold
        overbought_J = J > overbought_threshold

        # 超卖状态
        oversold_K = K < oversold_threshold
        oversold_D = D < oversold_threshold
        oversold_J = J < oversold_threshold

        # 综合状态
        overbought = overbought_K & overbought_D
        oversold = oversold_K & oversold_D

        # J值的极端状态
        extreme_overbought = J > 90
        extreme_oversold = J < 10

        return {
            'overbought_K': overbought_K.astype(int),
            'overbought_D': overbought_D.astype(int),
            'overbought_J': overbought_J.astype(int),
            'oversold_K': oversold_K.astype(int),
            'oversold_D': oversold_D.astype(int),
            'oversold_J': oversold_J.astype(int),
            'overbought': overbought.astype(int),
            'oversold': oversold.astype(int),
            'extreme_overbought': extreme_overbought.astype(int),
            'extreme_oversold': extreme_oversold.astype(int),
            'overbought_percentage': overbought_K.mean(),
            'oversold_percentage': oversold_K.mean()
        }

    @staticmethod
    def kdj_divergence(high, low, close, K, D, lookback_period=20):
        """
        检测KDJ背离
        """
        lookback_period = max(5, int(lookback_period))

        # 顶背离：价格创新高，但KDJ未创新高
        price_peak = high.rolling(window=lookback_period, center=True).apply(
            lambda x: x.argmax(), raw=True
        )
        kdj_peak = K.rolling(window=lookback_period, center=True).apply(
            lambda x: x.argmax(), raw=True
        )

        # 底背离：价格创新低，但KDJ未创新低
        price_trough = low.rolling(window=lookback_period, center=True).apply(
            lambda x: x.argmin(), raw=True
        )
        kdj_trough = K.rolling(window=lookback_period, center=True).apply(
            lambda x: x.argmin(), raw=True
        )

        # 顶背离信号
        top_divergence = (price_peak == lookback_period // 2) & (kdj_peak != lookback_period // 2)

        # 底背离信号
        bottom_divergence = (price_trough == lookback_period // 2) & (kdj_trough != lookback_period // 2)

        return {
            'top_divergence': top_divergence.astype(int),
            'bottom_divergence': bottom_divergence.astype(int),
            'divergence_strength': abs(K.diff()).rolling(window=5).mean()
        }

    @staticmethod
    def kdj_momentum(K, D, J, periods=[5, 10, 20]):
        """
        计算KDJ动量
        """
        momentum_features = {}

        for period in periods:
            period = max(1, int(period))
            # K值动量
            momentum_features[f'K_momentum_{period}'] = K.pct_change(periods=period)
            # D值动量
            momentum_features[f'D_momentum_{period}'] = D.pct_change(periods=period)
            # J值动量
            momentum_features[f'J_momentum_{period}'] = J.pct_change(periods=period)
            # K-D差值动量
            kd_diff = K - D
            momentum_features[f'KD_diff_momentum_{period}'] = kd_diff.pct_change(periods=period)

        # 动量方向
        momentum_features['K_momentum_direction'] = np.sign(K.diff())
        momentum_features['D_momentum_direction'] = np.sign(D.diff())
        momentum_features['J_momentum_direction'] = np.sign(J.diff())

        # 动量加速
        momentum_features['K_momentum_acceleration'] = K.diff().diff()
        momentum_features['D_momentum_acceleration'] = D.diff().diff()
        momentum_features['J_momentum_acceleration'] = J.diff().diff()

        return momentum_features

    @staticmethod
    def kdj_volatility(K, D, J, window=20):
        """
        计算KDJ波动率
        """
        window = max(5, int(window))

        return {
            'K_volatility': K.rolling(window=window).std(),
            'D_volatility': D.rolling(window=window).std(),
            'J_volatility': J.rolling(window=window).std(),
            'KD_ratio_volatility': (K / D).rolling(window=window).std(),
            'K_smoothness': K.rolling(window=window).apply(
                lambda x: np.abs(np.diff(x)).mean(), raw=False
            ),
            'D_smoothness': D.rolling(window=window).apply(
                lambda x: np.abs(np.diff(x)).mean(), raw=False
            )
        }

    @staticmethod
    def kdj_position_analysis(K, D, J):
        """
        KDJ位置分析
        """
        # 相对位置
        k_position = (K - K.rolling(50).min()) / (K.rolling(50).max() - K.rolling(50).min())
        d_position = (D - D.rolling(50).min()) / (D.rolling(50).max() - D.rolling(50).min())
        j_position = (J - J.rolling(50).min()) / (J.rolling(50).max() - J.rolling(50).min())

        # 分位数位置
        k_quantile = K.rolling(100).apply(lambda x: (x.iloc[-1] > x).mean(), raw=False)
        d_quantile = D.rolling(100).apply(lambda x: (x.iloc[-1] > x).mean(), raw=False)
        j_quantile = J.rolling(100).apply(lambda x: (x.iloc[-1] > x).mean(), raw=False)

        return {
            'K_position': k_position,
            'D_position': d_position,
            'J_position': j_position,
            'K_quantile': k_quantile,
            'D_quantile': d_quantile,
            'J_quantile': j_quantile,
            'KD_position_diff': k_position - d_position,
            'KJ_position_diff': k_position - j_position
        }

    @staticmethod
    def kdj_trend_strength(K, D, window=20):
        """
        KDJ趋势强度分析
        """
        window = max(5, int(window))

        # K值趋势
        k_trend = K.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )

        # D值趋势
        d_trend = D.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )

        # K-D差值趋势
        kd_diff = K - D
        kd_trend = kd_diff.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )

        # 趋势一致性
        trend_alignment = (np.sign(k_trend) == np.sign(d_trend)).astype(int)

        return {
            'K_trend_slope': k_trend,
            'D_trend_slope': d_trend,
            'KD_trend_slope': kd_trend,
            'trend_alignment': trend_alignment,
            'trend_strength': np.abs(k_trend),
            'trend_direction': np.sign(k_trend)
        }

    @staticmethod
    def kdj_pattern_recognition(K, D, J, window=10):
        """
        KDJ模式识别
        """
        window = max(5, int(window))
        patterns = pd.Series('无模式', index=K.index)

        for i in range(window, len(K)):
            k_window = K.iloc[i - window:i + 1]
            d_window = D.iloc[i - window:i + 1]
            j_window = J.iloc[i - window:i + 1]

            # W底模式：K值形成双底
            if (k_window.iloc[-5] < k_window.iloc[-3] and
                    k_window.iloc[-3] > k_window.iloc[-1] and
                    abs(k_window.iloc[-5] - k_window.iloc[-1]) < 10):
                patterns.iloc[i] = 'W底'

            # M顶模式：K值形成双顶
            elif (k_window.iloc[-5] > k_window.iloc[-3] and
                  k_window.iloc[-3] < k_window.iloc[-1] and
                  abs(k_window.iloc[-5] - k_window.iloc[-1]) < 10):
                patterns.iloc[i] = 'M顶'

            # 金叉后多头排列
            elif (K.iloc[i] > D.iloc[i] and
                  K.iloc[i - 1] > D.iloc[i - 1] and
                  K.iloc[i] > K.iloc[i - 1] and
                  D.iloc[i] > D.iloc[i - 1]):
                patterns.iloc[i] = '多头排列'

            # 死叉后空头排列
            elif (K.iloc[i] < D.iloc[i] and
                  K.iloc[i - 1] < D.iloc[i - 1] and
                  K.iloc[i] < K.iloc[i - 1] and
                  D.iloc[i] < D.iloc[i - 1]):
                patterns.iloc[i] = '空头排列'

            # J值钝化
            elif (J.iloc[i] > 90 and
                  J.iloc[i - 1] > 90 and
                  J.iloc[i - 2] > 90):
                patterns.iloc[i] = '高位钝化'

            elif (J.iloc[i] < 10 and
                  J.iloc[i - 1] < 10 and
                  J.iloc[i - 2] < 10):
                patterns.iloc[i] = '低位钝化'

        return {
            'pattern': patterns,
            'pattern_changed': (patterns != patterns.shift(1)).astype(int)
        }

    @staticmethod
    def extract_kdj_features(high, low, close, n=20, m1=3, m2=3):
        """
        提取所有KDJ特征
        """
        # 计算基础KDJ值
        K, D, J = TechnicalFeatureExtractor.calculate_kdj(high, low, close, n, m1, m2)

        features = {
            'KDJ_K': K,
            'KDJ_D': D,
            'KDJ_J': J,
            'KDJ_KD_diff': K - D,
            'KDJ_KJ_diff': K - J,
            'KDJ_DJ_diff': D - J
        }

        # 金叉死叉信号
        cross_signal, cross_type = TechnicalFeatureExtractor.kdj_cross_signal(K, D)
        features['KDJ_cross_signal'] = cross_signal
        features['KDJ_cross_type'] = cross_type

        # 超买超卖状态
        overbought_oversold = TechnicalFeatureExtractor.kdj_overbought_oversold(K, D, J)
        features.update({f'KDJ_{k}': v for k, v in overbought_oversold.items()})

        # 背离检测
        divergence = TechnicalFeatureExtractor.kdj_divergence(high, low, close, K, D)
        features.update({f'KDJ_{k}': v for k, v in divergence.items()})

        # 动量分析
        momentum = TechnicalFeatureExtractor.kdj_momentum(K, D, J)
        features.update({f'KDJ_{k}': v for k, v in momentum.items()})

        # 波动率分析
        volatility = TechnicalFeatureExtractor.kdj_volatility(K, D, J)
        features.update({f'KDJ_{k}': v for k, v in volatility.items()})

        # 位置分析
        position = TechnicalFeatureExtractor.kdj_position_analysis(K, D, J)
        features.update({f'KDJ_{k}': v for k, v in position.items()})

        # 趋势强度
        trend = TechnicalFeatureExtractor.kdj_trend_strength(K, D)
        features.update({f'KDJ_{k}': v for k, v in trend.items()})

        # 模式识别
        pattern = TechnicalFeatureExtractor.kdj_pattern_recognition(K, D, J)
        features.update({f'KDJ_{k}': v for k, v in pattern.items()})

        # 综合信号
        features['KDJ_composite_signal'] = (
                cross_signal +
                features['KDJ_overbought'].astype(int) * -0.5 +
                features['KDJ_oversold'].astype(int) * 0.5 +
                features['KDJ_bottom_divergence'].astype(int) * 0.3 -
                features['KDJ_top_divergence'].astype(int) * 0.3
        )

        return features

    # ========== 新增：正态分布分析指标 ==========

    @staticmethod
    def normality_test_jarque_bera(returns):
        """
        Jarque-Bera正态性检验
        返回JB统计量和p值
        """
        if len(returns.dropna()) < 4:
            return np.nan, np.nan

        jb_stat, p_value = jarque_bera(returns.dropna())
        return jb_stat, p_value

    @staticmethod
    def normality_test_shapiro_wilk(returns):
        """
        Shapiro-Wilk正态性检验（适合小样本）
        返回统计量和p值
        """
        if len(returns.dropna()) < 3 or len(returns.dropna()) > 5000:
            return np.nan, np.nan

        stat, p_value = shapiro(returns.dropna())
        return stat, p_value

    @staticmethod
    def skewness_kurtosis_analysis(returns):
        """
        偏度和峰度分析
        返回偏度、峰度及其正态性判断
        """
        if len(returns.dropna()) < 2:
            return {
                'skewness': np.nan,
                'kurtosis': np.nan,
                'is_normal_skewness': False,
                'is_normal_kurtosis': False,
                'distribution_type': '未知'
            }

        skew = returns.skew()
        kurt = returns.kurtosis()

        # 判断是否符合正态分布特征
        is_normal_skewness = abs(skew) < 0.5
        is_normal_kurtosis = abs(kurt) < 1.0

        # 分布类型判断
        if abs(skew) > 1.0:
            skew_type = "高度偏态" if abs(skew) > 1.0 else "偏态"
        else:
            skew_type = "基本对称"

        if kurt > 1.0:
            kurt_type = "尖峰" if kurt > 3.0 else "稍尖"
        elif kurt < -1.0:
            kurt_type = "低峰" if kurt < -3.0 else "稍平"
        else:
            kurt_type = "适中"

        distribution_type = f"{skew_type}_{kurt_type}"

        return {
            'skewness': float(skew) if not pd.isna(skew) else np.nan,
            'kurtosis': float(kurt) if not pd.isna(kurt) else np.nan,
            'is_normal_skewness': bool(is_normal_skewness),
            'is_normal_kurtosis': bool(is_normal_kurtosis),
            'distribution_type': str(distribution_type)
        }

    @staticmethod
    def qq_plot_analysis(returns, quantiles=100):
        """
        Q-Q图分析
        返回分位数数据和拟合线
        """
        returns_clean = returns.dropna()
        if len(returns_clean) < 10:
            return None

        # 理论分位数
        theoretical_quantiles = np.percentile(
            np.random.randn(10000),
            np.linspace(0, 100, quantiles)
        )

        # 实际分位数
        actual_quantiles = np.percentile(
            returns_clean,
            np.linspace(0, 100, quantiles)
        )

        # 线性拟合
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            theoretical_quantiles,
            actual_quantiles
        )

        return {
            'theoretical_quantiles': theoretical_quantiles,
            'actual_quantiles': actual_quantiles,
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }

    @staticmethod
    def normality_score(returns, window=20):
        """
        正态性综合评分
        基于多个检验的综合评分（0-1，越高越接近正态分布）
        """
        window = max(5, int(window))
        if len(returns.dropna()) < window:
            return np.nan

        scores = []

        # 1. Jarque-Bera检验
        jb_stat, jb_p = TechnicalFeatureExtractor.normality_test_jarque_bera(returns)
        if not np.isnan(jb_p):
            scores.append(min(jb_p * 10, 1.0))

        # 2. 偏度得分
        skew = returns.skew()
        if not pd.isna(skew):
            skew_score = max(0, 1 - abs(skew))
            scores.append(skew_score)

        # 3. 峰度得分
        kurt = returns.kurtosis()
        if not pd.isna(kurt):
            kurt_score = max(0, 1 - abs(kurt) / 5)
            scores.append(kurt_score)

        # 4. Q-Q图R²得分
        qq_result = TechnicalFeatureExtractor.qq_plot_analysis(returns)
        if qq_result and not np.isnan(qq_result['r_squared']):
            scores.append(qq_result['r_squared'])

        return float(np.mean(scores)) if scores else np.nan

    @staticmethod
    def deviation_from_normal(returns, confidence_level=0.95):
        """
        偏离正态分布的程度
        返回各置信区间的实际vs预期比例
        """
        returns_clean = returns.dropna()
        if len(returns_clean) < 10:
            return {}

        # 正态分布的理论百分比
        z_scores = [1, 2, 3]
        theoretical_props = [
            stats.norm.cdf(z) - stats.norm.cdf(-z) for z in z_scores
        ]

        # 实际数据的百分比
        mean_val = returns_clean.mean()
        std_val = returns_clean.std()

        actual_props = []
        for z in z_scores:
            lower = mean_val - z * std_val
            upper = mean_val + z * std_val
            prop = ((returns_clean >= lower) & (returns_clean <= upper)).mean()
            actual_props.append(float(prop))

        # 计算偏差
        deviations = [actual - theo for actual, theo in zip(actual_props, theoretical_props)]

        return {
            '1_std_theoretical': float(theoretical_props[0]),
            '1_std_actual': actual_props[0],
            '1_std_deviation': float(deviations[0]),
            '2_std_theoretical': float(theoretical_props[1]),
            '2_std_actual': actual_props[1],
            '2_std_deviation': float(deviations[1]),
            '3_std_theoretical': float(theoretical_props[2]),
            '3_std_actual': actual_props[2],
            '3_std_deviation': float(deviations[2]),
            'fat_tails': bool(deviations[2] < -0.01),
            'thin_tails': bool(deviations[2] > 0.01),
        }

    @staticmethod
    def rolling_normality(returns, window=20):
        """
        滚动正态性分析
        返回滚动窗口的正态性评分
        """
        window = max(5, int(window))
        if len(returns) < window:
            return pd.Series([np.nan] * len(returns), index=returns.index)

        scores = []
        for i in range(len(returns) - window + 1):
            window_returns = returns.iloc[i:i + window]
            score = TechnicalFeatureExtractor.normality_score(window_returns, window)
            scores.append(score)

        # 填充NaN
        nan_prefix = [np.nan] * (window - 1)
        all_scores = nan_prefix + scores

        return pd.Series(all_scores[:len(returns)], index=returns.index)

    @staticmethod
    def extract_normality_features(returns, window=20):
        """
        提取所有正态分布相关特征
        """
        window = max(5, int(window))
        features = {}

        # 基本统计量
        features['returns_mean'] = float(returns.mean())
        features['returns_std'] = float(returns.std())

        # 偏度和峰度分析
        skew_kurt = TechnicalFeatureExtractor.skewness_kurtosis_analysis(returns)
        for k, v in skew_kurt.items():
            features[f'skewness_{k}'] = v

        # Jarque-Bera检验
        jb_stat, jb_p = TechnicalFeatureExtractor.normality_test_jarque_bera(returns)
        features['jb_statistic'] = float(jb_stat) if not np.isnan(jb_stat) else np.nan
        features['jb_p_value'] = float(jb_p) if not np.isnan(jb_p) else np.nan
        features['jb_normal'] = bool(jb_p > 0.05) if not np.isnan(jb_p) else False

        # Shapiro-Wilk检验（小样本）
        if len(returns.dropna()) < 5000:
            sw_stat, sw_p = TechnicalFeatureExtractor.normality_test_shapiro_wilk(returns)
            features['sw_statistic'] = float(sw_stat) if not np.isnan(sw_stat) else np.nan
            features['sw_p_value'] = float(sw_p) if not np.isnan(sw_p) else np.nan
            features['sw_normal'] = bool(sw_p > 0.05) if not np.isnan(sw_p) else False

        # 正态性综合评分
        features['normality_score'] = TechnicalFeatureExtractor.normality_score(returns, window)

        # 偏离正态分布程度
        deviation = TechnicalFeatureExtractor.deviation_from_normal(returns)
        for k, v in deviation.items():
            features[f'deviation_{k}'] = v

        # Q-Q图分析
        qq_result = TechnicalFeatureExtractor.qq_plot_analysis(returns)
        if qq_result:
            features['qq_r_squared'] = qq_result.get('r_squared', np.nan)
            features['qq_slope'] = qq_result.get('slope', np.nan)
            features['qq_intercept'] = qq_result.get('intercept', np.nan)

        # 滚动正态性
        rolling_norm = TechnicalFeatureExtractor.rolling_normality(returns, window)
        features['rolling_normality_mean'] = float(rolling_norm.mean()) if not rolling_norm.empty else np.nan
        features['rolling_normality_std'] = float(rolling_norm.std()) if not rolling_norm.empty else np.nan
        if not rolling_norm.empty and not pd.isna(rolling_norm.iloc[-1]):
            features['rolling_normality_last'] = float(rolling_norm.iloc[-1])
        else:
            features['rolling_normality_last'] = np.nan

        return features

    # ========== 以下为通达信指标转Python函数 ==========

    @staticmethod
    def high_volume_distance(volume, period=1500):
        """
        计算高量柱距今周期数
        对应：高量距今 = CONST(REF(HHVBARS(VOL,1500),1))+1
        """
        period = max(100, int(period))
        if len(volume) < period:
            return np.nan
        highest_idx = volume.rolling(window=period).apply(lambda x: x.argmax(), raw=False)
        distance = highest_idx.shift(1) + 1
        return distance.iloc[-1] if not distance.isna().all() else np.nan

    @staticmethod
    def winner(close):
        """
        获利盘比例（简化模拟）
        WINNER(CLOSE) 在通达信中计算收盘价以下的获利比例
        此处简化为：收盘价高于过去N日最低价的比例
        """
        return (close > close.rolling(20).min()).astype(int)

    @staticmethod
    def zf(close):
        """涨幅"""
        return close.pct_change() * 100

    @staticmethod
    def volume_ratio(volume, ma_period=5):
        """量比：今日成交量 / 过去5日平均成交量"""
        ma_period = max(1, int(ma_period))
        return volume / volume.rolling(window=ma_period).mean().shift(1)

    @staticmethod
    def volume_quantify(volume, ma_period=5):
        """
        量比量化分类
        返回每个成交量所属的量比等级标签
        """
        ma_period = max(1, int(ma_period))
        vr = TechnicalFeatureExtractor.volume_ratio(volume, ma_period)
        labels = pd.Series(index=volume.index, data='未知')
        labels[(vr >= 0.001) & (vr < 0.5)] = '严重缩量'
        labels[(vr >= 0.5) & (vr < 0.8)] = '缩量'
        labels[(vr >= 0.8) & (vr < 1.5)] = '正常量比'
        labels[(vr >= 1.5) & (vr < 2.5)] = '温和放量'
        labels[(vr >= 2.5) & (vr < 5.0)] = '明显放量'
        labels[(vr >= 5.0) & (vr < 10.0)] = '剧烈放量'
        labels[vr >= 10.0] = '反向操作'
        return labels

    @staticmethod
    def ma_position(close, periods=[20, 60, 120]):
        """
        均线位置判断
        返回收盘价是否在各均线之上
        """
        result = {}
        for period in periods:
            period = max(1, int(period))
            ma = close.ewm(span=period, adjust=False).mean()
            result[f'MA{period}'] = (close > ma).astype(int)
        return pd.DataFrame(result)

    @staticmethod
    def volume_conditions(volume, close, open_price):
        """
        成交量特殊条件判断
        返回布尔序列，标记是否符合条件
        """
        # AA: 成交量翻倍且收盘价上涨
        cond1 = (volume >= volume.shift(1) * 2) & (close > close.shift(1))

        # AAD: 成交量萎缩至前一日一半以内
        cond2 = (volume < volume.shift(1) * 0.5) & (volume > volume.shift(1) * 0.02)

        return pd.DataFrame({
            'volume_double_up': cond1,
            'volume_shrink': cond2
        })

    def extract_all_features_with_config(self, df, price_col='close', volume_col='volume',
                                         open_col='open', high_col='high', low_col='low'):
        """
        批量提取所有技术特征（使用配置参数）
        df: 包含OHLCV数据的DataFrame
        """
        features = {}

        # 记录哪些指标计算成功
        success_indicators = []
        failed_indicators = []

        # 基础价格特征
        close = df[price_col]
        volume = df[volume_col]
        open_price = df[open_col]
        high = df[high_col]
        low = df[low_col]

        # 计算收益率（用于正态分布分析）
        returns = close.pct_change()

        try:
            # RSI - 使用实例方法（带配置参数）
            features['RSI'] = self.calculate_rsi(close)
            success_indicators.append('RSI')
        except Exception as e:
            print(f"RSI计算失败: {e}")
            failed_indicators.append(('RSI', str(e)))
            features['RSI'] = pd.Series(50.0, index=close.index)

        try:
            # MACD - 使用配置参数
            macd_fast = self._get_param('macd', 'fast', 14)
            macd_slow = self._get_param('macd', 'slow', 53)
            macd_signal = self._get_param('macd', 'signal', 5)
            macd, signal, hist = TechnicalFeatureExtractor.calculate_macd(
                close, fast=macd_fast, slow=macd_slow, signal=macd_signal
            )
            features['MACD'] = macd
            features['MACD_Signal'] = signal
            features['MACD_Hist'] = hist
            success_indicators.append('MACD')
        except Exception as e:
            print(f"MACD计算失败: {e}")
            failed_indicators.append(('MACD', str(e)))
            default_series = pd.Series(0.0, index=close.index)
            features['MACD'] = default_series
            features['MACD_Signal'] = default_series
            features['MACD_Hist'] = default_series

        try:
            # 布林带 - 使用配置参数
            bb_window = self._get_param('bollinger_bands', 'window', 20)
            bb_num_std = self._get_param('bollinger_bands', 'num_std', 2)
            upper, mid, lower = TechnicalFeatureExtractor.calculate_bollinger_bands(
                close, window=bb_window, num_std=bb_num_std
            )
            features['BB_Upper'] = upper
            features['BB_Mid'] = mid
            features['BB_Lower'] = lower
            success_indicators.append('BollingerBands')
        except Exception as e:
            print(f"布林带计算失败: {e}")
            failed_indicators.append(('BollingerBands', str(e)))
            default_series = pd.Series(close.values, index=close.index)
            features['BB_Upper'] = default_series
            features['BB_Mid'] = default_series
            features['BB_Lower'] = default_series

        # ========== 新增：金三角指标 ==========
        try:
            # 获取金三角配置参数
            triangle_short = self._get_param('golden_triangle', 'short_period', 5)
            triangle_medium = self._get_param('golden_triangle', 'medium_period', 10)
            triangle_long = self._get_param('golden_triangle', 'long_period', 20)

            triangle_features = TechnicalFeatureExtractor.extract_golden_triangle_features(
                close,
                short_period=triangle_short,
                medium_period=triangle_medium,
                long_period=triangle_long
            )

            # 添加所有金三角特征
            for k, v in triangle_features.items():
                if isinstance(v, pd.Series):
                    # 确保Series长度与数据一致
                    if len(v) == len(close):
                        features[k] = v
                    else:
                        # 对齐索引
                        aligned_v = pd.Series(np.nan, index=close.index, dtype=float)
                        if len(v) > 0:
                            aligned_v.iloc[-len(v):] = v.values[-len(v):]
                        features[k] = aligned_v
                else:
                    # 对于标量值，创建Series
                    features[k] = pd.Series([v] * len(close), index=close.index)

            success_indicators.append('GoldenTriangle')
        except Exception as e:
            print(f"警告：金三角指标计算部分失败: {e}")
            failed_indicators.append(('GoldenTriangle', str(e)))
            # 创建默认金三角特征
            default_series = pd.Series(0.0, index=close.index, dtype=float)
            features['triangle_ma_short'] = default_series
            features['triangle_ma_medium'] = default_series
            features['triangle_ma_long'] = default_series
            features['triangle_golden_triangle_signal'] = default_series

        # ========== 新增：KDJ指标 ==========
        try:
            # 获取KDJ配置参数
            kdj_n = self._get_param('kdj', 'n', 9)
            kdj_m1 = self._get_param('kdj', 'm1', 3)
            kdj_m2 = self._get_param('kdj', 'm2', 3)

            kdj_features = TechnicalFeatureExtractor.extract_kdj_features(
                high, low, close, n=kdj_n, m1=kdj_m1, m2=kdj_m2
            )

            # 直接添加所有KDJ特征，保持原命名
            for k, v in kdj_features.items():
                if isinstance(v, pd.Series):
                    # 确保Series长度与数据一致
                    if len(v) == len(close):
                        features[k] = v
                    else:
                        # 对齐索引
                        aligned_v = pd.Series(np.nan, index=close.index)
                        if len(v) > 0:
                            aligned_v.iloc[-len(v):] = v.values[-len(v):]
                        features[k] = aligned_v
                else:
                    # 对于标量值，创建Series
                    features[k] = pd.Series([v] * len(close), index=close.index)

            success_indicators.append('KDJ')
        except Exception as e:
            print(f"警告：KDJ指标计算部分失败: {e}")
            failed_indicators.append(('KDJ', str(e)))
            # 创建默认KDJ特征
            default_kdj = pd.Series(50.0, index=close.index, dtype=float)
            features['KDJ_K'] = default_kdj
            features['KDJ_D'] = default_kdj
            features['KDJ_J'] = default_kdj

        # 高量距离
        try:
            # 获取高量距离配置参数
            high_volume_period = self._get_param('high_volume', 'period', 1500)
            distance_val = TechnicalFeatureExtractor.high_volume_distance(volume, period=high_volume_period)
            features['High_Volume_Distance'] = pd.Series([distance_val] * len(close), index=close.index, dtype=float)
            success_indicators.append('HighVolumeDistance')
        except Exception as e:
            print(f"高量距离计算失败: {e}")
            failed_indicators.append(('HighVolumeDistance', str(e)))
            features['High_Volume_Distance'] = pd.Series(0.0, index=volume.index, dtype=float)

        # 获利盘比例（简化）
        try:
            features['Winner'] = TechnicalFeatureExtractor.winner(close)
            success_indicators.append('Winner')
        except Exception as e:
            print(f"获利盘比例计算失败: {e}")
            failed_indicators.append(('Winner', str(e)))
            features['Winner'] = pd.Series(0, index=close.index)

        # 涨幅
        try:
            features['ZF'] = TechnicalFeatureExtractor.zf(close)
            success_indicators.append('ZF')
        except Exception as e:
            print(f"涨幅计算失败: {e}")
            failed_indicators.append(('ZF', str(e)))
            features['ZF'] = pd.Series(0.0, index=close.index)

        # 量比 - 使用配置参数
        try:
            volume_ma_period = self._get_param('volume_ratio', 'ma_period', 5)
            features['Volume_Ratio'] = TechnicalFeatureExtractor.volume_ratio(volume, ma_period=volume_ma_period)
            success_indicators.append('VolumeRatio')
        except Exception as e:
            print(f"量比计算失败: {e}")
            failed_indicators.append(('VolumeRatio', str(e)))
            features['Volume_Ratio'] = pd.Series(1.0, index=close.index)

        # 量比量化标签
        try:
            volume_ma_period = self._get_param('volume_ratio', 'ma_period', 5)
            features['Volume_Category'] = TechnicalFeatureExtractor.volume_quantify(volume, ma_period=volume_ma_period)
            success_indicators.append('VolumeCategory')
        except Exception as e:
            print(f"量比量化标签计算失败: {e}")
            failed_indicators.append(('VolumeCategory', str(e)))
            features['Volume_Category'] = pd.Series('正常量比', index=close.index)

        # 均线位置 - 使用配置参数
        try:
            ma_periods = self._get_param('ma_position', 'periods', [20, 60, 120])
            ma_pos = TechnicalFeatureExtractor.ma_position(close, periods=ma_periods)
            for col in ma_pos.columns:
                features[col] = ma_pos[col]
            success_indicators.append('MAPosition')
        except Exception as e:
            print(f"均线位置计算失败: {e}")
            failed_indicators.append(('MAPosition', str(e)))

        # 成交量条件
        try:
            vol_conds = TechnicalFeatureExtractor.volume_conditions(volume, close, open_price)
            for col in vol_conds.columns:
                features[col] = vol_conds[col]
            success_indicators.append('VolumeConditions')
        except Exception as e:
            print(f"成交量条件计算失败: {e}")
            failed_indicators.append(('VolumeConditions', str(e)))
            pass

        # 成交量均线 - 使用配置参数
        try:
            volume_ma_short = self._get_param('volume_ma', 'short', 4)
            volume_ma_medium = self._get_param('volume_ma', 'medium', 11)
            volume_ma_long = self._get_param('volume_ma', 'long', 117)

            features['MA_VOL_4'] = volume.rolling(volume_ma_short).mean()
            features['MA_VOL_11'] = volume.rolling(volume_ma_medium).mean()
            features['MA_VOL_117'] = volume.rolling(volume_ma_long).mean()
            success_indicators.append('VolumeMA')
        except Exception as e:
            print(f"成交量均线计算失败: {e}")
            failed_indicators.append(('VolumeMA', str(e)))
            features['MA_VOL_4'] = volume.rolling(4).mean()
            features['MA_VOL_11'] = volume.rolling(11).mean()
            features['MA_VOL_117'] = volume.rolling(117).mean()

        # ========== 新增：正态分布特征 ==========
        try:
            # 获取正态分布配置参数
            normality_window = self._get_param('normality', 'window', 20)
            normality_features = TechnicalFeatureExtractor.extract_normality_features(returns, window=normality_window)
            for key, value in normality_features.items():
                if isinstance(value, (int, float, bool, np.float64, np.int64)):
                    # 对于数值和布尔值，扩展为Series
                    features[f'normality_{key}'] = pd.Series(
                        [float(value) if isinstance(value, (int, float, np.float64, np.int64)) else value] * len(close),
                        index=close.index)
                elif isinstance(value, str):
                    # 对于字符串值
                    features[f'normality_{key}'] = pd.Series([value] * len(close), index=close.index)
                elif isinstance(value, pd.Series):
                    # 对齐Series
                    if len(value) == len(close):
                        features[f'normality_{key}'] = value
                    else:
                        aligned = pd.Series(np.nan, index=close.index)
                        if len(value) > 0:
                            aligned.iloc[-len(value):] = value.values[-len(value):]
                        features[f'normality_{key}'] = aligned
                else:
                    # 其他类型转换为字符串Series
                    features[f'normality_{key}'] = pd.Series([str(value)] * len(close), index=close.index)

            success_indicators.append('Normality')
        except Exception as e:
            print(f"警告：正态分布特征提取失败: {e}")
            failed_indicators.append(('Normality', str(e)))
            # 填充默认值
            features['normality_score'] = pd.Series(0.5, index=close.index, dtype=float)
            features['normality_returns_mean'] = pd.Series(returns.mean(), index=close.index, dtype=float)
            features['normality_returns_std'] = pd.Series(returns.std(), index=close.index, dtype=float)

        # 转换为DataFrame
        features_df = pd.DataFrame(features, index=close.index)

        # 处理NaN值
        try:
            features_df = features_df.ffill().bfill()
        except:
            # 如果ffill/bfill失败，使用fillna
            features_df = features_df.fillna(0)

        # 打印统计信息
        #print("成功指标列表:", success_indicators)
        # print(f"\n特征提取统计:")
        # print(f"成功计算的指标: {len(success_indicators)}个")
        # print(f"失败的指标: {len(failed_indicators)}个")
        # if failed_indicators:
        #     print("失败详情:")
        #     for indicator, error in failed_indicators:
        #         print(f"  - {indicator}: {error}")

        return features_df

    @staticmethod
    def extract_all_features(df, price_col='close', volume_col='volume',
                             open_col='open', high_col='high', low_col='low'):
        """
        批量提取所有技术特征（静态方法，不使用配置参数）
        df: 包含OHLCV数据的DataFrame
        """
        extractor = TechnicalFeatureExtractor()
        return extractor.extract_all_features_with_config(
            df, price_col, volume_col, open_col, high_col, low_col
        )


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例数据加载
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 105,
        'low': np.random.randn(200).cumsum() + 95,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)


    # 改进的Mock对象设置
    class MockConfigLoader:
        def __init__(self):
            self.config = {
                'rsi': {'period': 10},
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger_bands': {'window': 30, 'num_std': 1.5},
                'golden_triangle': {'short_period': 10, 'medium_period': 20, 'long_period': 30},
                'kdj': {'n': 5, 'm1': 2, 'm2': 2},
                'high_volume': {'period': 1000},
                'volume_ratio': {'ma_period': 3},
                'ma_position': {'periods': [30, 90, 180]},
                'volume_ma': {'short': 3, 'medium': 8, 'long': 60},
                'normality': {'window': 30}
            }

        def get(self, key, default=None):
            return self.config.get(key, default)

        def get_recommended_value(self, indicator, param_name, default_value):
            """模拟获取推荐值的方法"""
            if indicator in self.config and param_name in self.config[indicator]:
                return self.config[indicator][param_name]
            return default_value


    # 创建模拟的param_helper
    class MockParamHelper:
        def get_feature_param(self, param_loader, indicator, param_name, default_value, config=None):
            try:
                # 尝试从配置加载器获取
                indicator_config = param_loader.get(indicator, {})
                return indicator_config.get(param_name, default_value)
            except:
                return default_value


    # 方法1：使用静态方法（无需实例化）
    print("=" * 48)
    print("方法1：使用静态方法提取特征")
    print("=" * 48)
    features_df = TechnicalFeatureExtractor.extract_all_features(
        sample_data,
        price_col='close',
        volume_col='volume',
        open_col='open'
    )

    print(f"提取的特征数据形状: {features_df.shape}")
    print(f"\n特征总数: {len(features_df.columns)}")
    print(f"\n前3行特征:")
    print(features_df.head(3))

    # 方法2：使用实例方法（支持配置参数）
    print(f"\n" + "=" * 48)
    print("方法2：使用实例方法提取特征（带配置参数）")
    print("=" * 48)

    # 创建带配置的提取器
    mock_param_loader = MockConfigLoader()
    extractor = TechnicalFeatureExtractor(param_loader=mock_param_loader)
    extractor.param_helper = MockParamHelper()

    # 使用带配置的方法提取特征
    try:
        features_config_df = extractor.extract_all_features_with_config(
            sample_data,
            price_col='close',
            volume_col='volume',
            open_col='open'
        )

        print(f"使用配置参数提取的特征数据形状: {features_config_df.shape}")
        print(f"\n特征总数: {len(features_config_df.columns)}")

        # 显示使用配置参数的RSI统计
        if 'RSI' in features_config_df.columns:
            print(f"\n使用配置参数的RSI统计信息:")
            print(f"RSI范围: {features_config_df['RSI'].min():.2f} - {features_config_df['RSI'].max():.2f}")
            print(f"RSI均值: {features_config_df['RSI'].mean():.2f}")

        # 检查金三角相关特征
        print(f"\n" + "=" * 48)
        print("金三角相关特征分析（使用配置参数）")
        print("=" * 48)
        triangle_cols = [col for col in features_config_df.columns if 'triangle' in col.lower()]
        print(f"金三角特征数量: {len(triangle_cols)}")

        # 金三角信号统计
        if 'triangle_golden_triangle_signal' in features_config_df.columns:
            print(f"\n金三角信号详细统计:")
            signal_counts = features_config_df['triangle_golden_triangle_signal'].value_counts()
            for signal, count in signal_counts.items():
                if signal == 1:
                    signal_name = "金三角买入信号"
                elif signal == -1:
                    signal_name = "死三角卖出信号"
                else:
                    signal_name = "无信号"
                print(f"{signal_name}: {count}次 ({count / len(features_config_df) * 100:.1f}%)")

        # 比较不同参数的差异
        print(f"\n" + "=" * 48)
        print("配置参数对比")
        print("=" * 48)
        print("RSI周期: 默认14 vs 配置10")
        print("MACD参数: 默认(14,53,5) vs 配置(12,26,9)")
        print("布林带: 默认(20,2) vs 配置(30,1.5)")
        print("金三角: 默认(5,10,20) vs 配置(10,20,30)")
        print("KDJ: 默认(9,3,3) vs 配置(5,2,2)")

        print(f"\n" + "=" * 48)
        print("特征提取完成！")
        print(f"共提取 {len(features_config_df.columns)} 个技术特征")
        print(f"包含 {len(triangle_cols)} 个金三角相关特征")
        print("=" * 48)

    except Exception as e:
        print(f"带配置参数的特征提取失败: {e}")
        import traceback

        traceback.print_exc()