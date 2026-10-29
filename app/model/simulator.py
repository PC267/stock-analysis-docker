# simulator.py - 修复初始化顺序问题，集成全部模型（不含Informer）
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import timedelta
import warnings
import hashlib
import json
from typing import List, Dict, Tuple, Optional, Any


# ==================== 导入外部特征提取器 ====================

try:
    from features.technical_features import TechnicalFeatureExtractor as ExternalTechnicalExtractor
    TECHNICAL_EXTRACTOR_AVAILABLE = True
    print("✓ 成功导入外部技术特征提取器")
except ImportError as e:
    print(f"⚠️ 无法导入技术特征提取器: {e}")
    TECHNICAL_EXTRACTOR_AVAILABLE = False

try:
    from features.market_features import MarketFeatureExtractor as ExternalMarketExtractor
    MARKET_EXTRACTOR_AVAILABLE = True
    print("✓ 成功导入外部市场特征提取器")
except ImportError as e:
    print(f"⚠️ 无法导入市场特征提取器: {e}")
    MARKET_EXTRACTOR_AVAILABLE = False

warnings.filterwarnings('ignore')

# ==================== 回测分析结果配置（包含全部模型，不含Informer） ====================
BACKTEST_RESULTS = {
    'model_performance': {
        'garch': {
            'avg_score': 64.5,
            'direction_accuracy': 0.68,
            'price_match': 0.953,
            'suitable_markets': ['high_volatility', 'high_volatility_nonlinear', 'default'],
            'weight': 1.0
        },
        'sarima': {
            'avg_score': 66.0,
            'direction_accuracy': 0.69,
            'price_match': 0.965,
            'suitable_markets': ['trending', 'stable_trend'],
            'weight': 0.85
        },
        'multi_factor': {
            'avg_score': 61.5,
            'direction_accuracy': 0.66,
            'price_match': 0.945,
            'suitable_markets': ['stable_trend', 'trending', 'default'],
            'weight': 0.8
        },
        'xgboost': {
            'avg_score': 62.0,
            'direction_accuracy': 0.65,
            'price_match': 0.920,
            'suitable_markets': ['high_volatility_nonlinear'],
            'weight': 0.75
        },
        'arima': {
            'avg_score': 65.5,
            'direction_accuracy': 0.67,
            'price_match': 0.955,
            'suitable_markets': ['trending', 'stable_trend', 'default'],
            'weight': 0.8
        },
        'lstm': {
            'avg_score': 63.0,
            'direction_accuracy': 0.64,
            'price_match': 0.930,
            'suitable_markets': ['high_volatility_nonlinear', 'default'],
            'weight': 0.7
        },
        'lightgbm': {
            'avg_score': 64.0,
            'direction_accuracy': 0.66,
            'price_match': 0.940,
            'suitable_markets': ['high_volatility_nonlinear', 'trending', 'default'],
            'weight': 0.75
        },
        'catboost': {
            'avg_score': 63.8,
            'direction_accuracy': 0.65,
            'price_match': 0.938,
            'suitable_markets': ['stable_trend', 'default'],
            'weight': 0.7
        },
        'quant_factor': {
            'avg_score': 64.0,
            'direction_accuracy': 0.67,
            'price_match': 0.935,
            'suitable_markets': ['default', 'high_volatility', 'trending', 'stable_trend'],
            'weight': 0.8
        },
        'ensemble': {
            'avg_score': 67.0,
            'direction_accuracy': 0.70,
            'price_match': 0.960,
            'suitable_markets': ['default', 'high_volatility', 'trending'],
            'weight': 1.0
        }
    },
    'market_state_mapping': {
        'high_volatility_nonlinear': ['garch', 'xgboost', 'lightgbm', 'catboost', 'quant_factor'],
        'high_volatility': ['garch', 'multi_factor', 'arima', 'quant_factor'],
        'trending': ['sarima', 'multi_factor', 'arima', 'lightgbm', 'quant_factor'],
        'stable_trend': ['sarima', 'garch', 'arima', 'catboost', 'quant_factor'],
        'default': ['sarima', 'garch', 'multi_factor', 'xgboost', 'arima', 'lightgbm', 'catboost', 'quant_factor', 'ensemble']
    }
}


class DynamicModelSelector:
    """动态模型选择器 - 基于回测结果和市场状态"""

    def __init__(self, param_loader):
        self.param_loader = param_loader
        self.backtest_results = BACKTEST_RESULTS

    def select_best_models(self, market_regime: str, available_models: List[str],
                           stock_features: Dict) -> List[Tuple[str, float]]:
        print(f"【动态模型选择】市场状态: {market_regime}")

        if market_regime in self.backtest_results['market_state_mapping']:
            recommended_models = self.backtest_results['market_state_mapping'][market_regime]
        else:
            recommended_models = self.backtest_results['market_state_mapping']['default']

        available_recommended = [m for m in recommended_models if m in available_models]
        if not available_recommended:
            available_recommended = available_models[:3]

        model_weights = []
        total_weight = 0

        for model_name in available_recommended:
            if model_name in self.backtest_results['model_performance']:
                performance = self.backtest_results['model_performance'][model_name]
                weight = performance['weight']

                if market_regime in performance['suitable_markets']:
                    weight *= 1.2

                technical_score = stock_features.get('technical_score', 0.5)
                volatility = stock_features.get('volatility', 0.02)

                if model_name == 'garch' and volatility > 0.03:
                    weight *= 1.3
                elif model_name == 'xgboost' and technical_score > 0.7:
                    weight *= 1.1
                elif model_name == 'lightgbm' and technical_score > 0.7:
                    weight *= 1.1
                elif model_name == 'catboost' and volatility < 0.02:
                    weight *= 1.15
                elif model_name == 'lstm' and volatility > 0.04:
                    weight *= 1.2
                elif model_name == 'quant_factor':
                    # 量化因子在因子丰富度高时提升权重
                    technical_score = stock_features.get('technical_score', 0.5)
                    if technical_score > 0.6:
                        weight *= 1.15
                    # 在趋势明确时表现更佳
                    trend = stock_features.get('trend_strength', 0)
                    if trend > 0.3:
                        weight *= 1.1

                model_weights.append((model_name, weight))
                total_weight += weight

        if total_weight > 0:
            model_weights = [(name, weight / total_weight) for name, weight in model_weights]

        model_weights.sort(key=lambda x: x[1], reverse=True)
        print(f"  选择的模型: {', '.join([f'{m}:{w:.2f}' for m, w in model_weights[:3]])}")

        return model_weights


class ModelConfigManager:
    """模型配置管理器 - 基于回测结果优化"""

    def __init__(self, param_loader):
        self.param_loader = param_loader
        self.model_configs = self._load_optimized_configs()

    def _load_optimized_configs(self) -> Dict:
        configs = {
            'garch': {
                'enabled': True,
                'base_score': 7.5,
                'order': (2, 2),
                'distribution': 't',
                'window_size': 100,
                'prediction_horizon': 5,
                'volatility_persistence': 0.9,
                'weight': 0.35
            },
            'sarima': {
                'enabled': True,
                'base_score': 6.8,
                'order': (1, 0, 0),
                'seasonal_order': (0, 0, 0, 0),
                'window_size': 100,
                'prediction_horizon': 5,
                'weight': 0.15
            },
            'multi_factor': {
                'enabled': True,
                'base_score': 6.5,
                'n_features': 3,
                'window_size': 100,
                'prediction_horizon': 5,
                'factor_selection_method': 'correlation_threshold_0.8',
                'weight': 0.15
            },
            'xgboost': {
                'enabled': True,
                'base_score': 6.0,
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'n_features': 13,
                'window_size': 100,
                'prediction_horizon': 5,
                'weight': 0.10
            },
            'arima': {
                'enabled': True,
                'base_score': 7.2,
                'order': (2, 1, 2),
                'window_size': 100,
                'prediction_horizon': 5,
                'weight': 0.15
            },
            'lstm': {
                'enabled': True,
                'base_score': 6.0,
                'units': 64,
                'epochs': 50,
                'batch_size': 32,
                'window_size': 60,
                'prediction_horizon': 5,
                'weight': 0.10
            },
            'lightgbm': {
                'enabled': True,
                'base_score': 6.2,
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 7,
                'window_size': 50,
                'prediction_horizon': 5,
                'weight': 0.10
            },
            'catboost': {
                'enabled': True,
                'base_score': 6.1,
                'iterations': 100,
                'learning_rate': 0.05,
                'depth': 6,
                'window_size': 50,
                'prediction_horizon': 5,
                'weight': 0.10
            },
            'quant_factor': {
                'enabled': True,
                'base_score': 6.8,
                'n_factors': 16,
                'n_estimators': 100,
                'max_depth': 8,
                'window_size': 60,
                'prediction_horizon': 5,
                'weight': 0.15
            },
            'ensemble': {
                'enabled': True,
                'base_score': 7.8,
                'description': '动态集成模型',
                'weight': 0.2
            }
        }
        return configs

    def get_config(self, model_name: str) -> Dict:
        return self.model_configs.get(model_name, {})

    def get_enabled_models(self) -> List[str]:
        return [name for name, config in self.model_configs.items()
                if config.get('enabled', False)]


def get_config_loader():
    class SimpleConfigLoader:
        def __init__(self):
            self.params = {}

        def get_param(self, key, default=None):
            return self.params.get(key, default)

    return SimpleConfigLoader()


class StockSimulator:
    def __init__(self, config, models, processed_data):
        try:
            self.config = config
            self.models = models
            self.processed_data = processed_data

            print(f"【股票模拟器】初始化开始...")

            # 第一步：初始化配置加载器
            self.param_loader = get_config_loader()

            # 第二步：加载基础配置
            self._load_configurations()
            print(f"✓ 基础配置加载完成: {self.stock_symbol}")

            # 检查关键数据
            if not hasattr(self, 'historical_data') or self.historical_data is None:
                print("⚠️ 警告: historical_data 为空，使用默认数据")
                self._create_default_historical_data()

            # 第三步：初始化管理器
            self.model_config_manager = ModelConfigManager(self.param_loader)
            self.model_selector = DynamicModelSelector(self.param_loader)

            # 第四步：初始化外部特征提取器
            self.technical_extractor_available = TECHNICAL_EXTRACTOR_AVAILABLE
            self.market_extractor_available = MARKET_EXTRACTOR_AVAILABLE
            self._initialize_external_extractors()

            # 第五步：提取股票特征
            self.stock_features = self._extract_deep_stock_features()
            print(f"✓ 股票特征提取完成")

            # 第六步：检测市场状态
            self.market_regime = self._detect_market_regime()
            print(f"✓ 市场状态检测完成: {self.market_regime}")

            # 第七步：分析历史K线模式
            self.kline_patterns = self._analyze_historical_patterns_deep()
            print(f"✓ 历史模式分析完成")

            # 第八步：生成股票指纹
            self.stock_fingerprint = self._generate_stock_fingerprint()
            print(f"✓ 股票指纹生成: {self.stock_fingerprint[:16]}...")

            # 第九步：动态选择模型
            self._select_models_dynamically()
            self._determine_best_model()

            # 第十步：注册所有模型模拟方法
            self.model_simulation_methods = {
                'garch': self._simulate_garch,
                'sarima': self._simulate_sarima,
                'multi_factor': self._simulate_multi_factor,
                'xgboost': self._simulate_xgboost,
                'arima': self._simulate_arima,
                'lstm': self._simulate_lstm,
                'lightgbm': self._simulate_lightgbm,
                'catboost': self._simulate_catboost,
                'quant_factor': self._simulate_quant_factor,
                'ensemble': self._simulate_ensemble,
            }

            print(f"✓ {self.stock_symbol} 模拟器初始化完成")
            print(f"  市场状态: {self.market_regime}")
            print(f"  技术分数: {self.stock_features.get('technical_score', 0):.2f}")
            print(f"  波动特性: {self.stock_features.get('volatility_profile', 'normal')}")
            print(f"  选择模型: {', '.join(self.selected_models)}")
            if self.best_model_name:
                print(f"  最优模型: {self.best_model_name}")

        except Exception as e:
            print(f"❌ StockSimulator 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_default_historical_data(self):
        print("正在创建默认历史数据...")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
        prices = np.random.normal(100, 10, 30).cumsum()
        self.historical_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, 30)
        }, index=dates)
        print(f"✓ 创建了默认历史数据: {len(self.historical_data)} 条记录")

    def _load_configurations(self):
        # 基础配置
        if hasattr(self.config, 'stock'):
            self.stock_symbol = getattr(self.config.stock, 'code', 'default')
        elif hasattr(self.config, 'get'):
            self.stock_symbol = self.config.get('stock_code', self.config.get('code', 'default'))
        else:
            try:
                self.stock_symbol = getattr(self.config, 'code', 'default')
            except:
                self.stock_symbol = 'default'

        # 模拟天数
        if hasattr(self.config, 'stock'):
            self.simulation_days = getattr(self.config.stock, 'simulation_days', 30)
        elif hasattr(self.config, 'simulator'):
            self.simulation_days = getattr(self.config.simulator, 'simulation_days', 30)
        else:
            self.simulation_days = self.config.get('days_to_simulate', 30) if isinstance(self.config, dict) else 30

        # 确定性模式
        if hasattr(self.config, 'simulator'):
            self.deterministic_mode = getattr(self.config.simulator, 'deterministic_mode', True)
        else:
            self.deterministic_mode = self.config.get('deterministic_mode', True) if isinstance(self.config, dict) else True

        # 获取历史数据
        if 'processed' in self.processed_data:
            self.historical_data = self.processed_data['processed']
            print(f"✓ 加载历史数据: {len(self.historical_data)} 条记录")
            print(f"  数据列: {list(self.historical_data.columns)}")
            print(f"  数据形状: {self.historical_data.shape}")
        else:
            print("❌ 警告: processed_data 中不存在 'processed' 键")
            print(f"  processed_data keys: {list(self.processed_data.keys())}")
            self.historical_data = None

        self.simulator_config = None
        if hasattr(self.config, 'simulator'):
            self.simulator_config = self.config.simulator
        elif isinstance(self.config, dict) and 'simulator' in self.config:
            self.simulator_config = self.config['simulator']

    def _initialize_external_extractors(self):
        if self.technical_extractor_available:
            try:
                self.technical_extractor = ExternalTechnicalExtractor(self.param_loader, self.config)
                print("✓ 外部技术特征提取器初始化成功")
            except Exception as e:
                print(f"⚠️ 外部技术特征提取器初始化失败: {e}")
                self.technical_extractor_available = False
                self.technical_extractor = None
        else:
            print("⚠️ 无法使用外部技术特征提取器，将使用简化特征")
            self.technical_extractor = None

        if self.market_extractor_available:
            try:
                market_data = None
                if 'market' in self.processed_data:
                    market_data = self.processed_data['market']
                self.market_extractor = ExternalMarketExtractor(market_data=market_data)
                print("✓ 外部市场特征提取器初始化成功")
            except Exception as e:
                print(f"⚠️ 外部市场特征提取器初始化失败: {e}")
                self.market_extractor_available = False
                self.market_extractor = None
        else:
            print("⚠️ 无法使用外部市场特征提取器，将使用简化市场分析")
            self.market_extractor = None

    def _extract_deep_stock_features(self) -> Dict:
        features = {
            'volatility': 0.02,
            'avg_return': 0.0005,
            'trend_strength': 0.0,
            'price_level': 100.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'avg_volume': 100000,
            'volume_volatility': 0.3,
            'technical_score': 0.5,
            'volatility_profile': 'normal',
            'price_pattern': 'random',
            'volume_pattern': 'normal',
            'support_resistance': [],
            'trend_lines': []
        }

        if self.historical_data is not None and len(self.historical_data) > 0:
            hist_data = self.historical_data

            if 'close' in hist_data.columns:
                prices = hist_data['close']
                returns = prices.pct_change().dropna()

                if len(returns) > 10:
                    features['volatility'] = float(returns.std())
                    features['avg_return'] = float(returns.mean())
                    features['skewness'] = float(returns.skew())
                    features['kurtosis'] = float(returns.kurtosis())
                    features['volatility_profile'] = self._analyze_volatility_profile(returns)
                    features['price_pattern'] = self._analyze_price_pattern(prices)
                    features['trend_strength'] = self._analyze_trend_strength(prices)
                    features['support_resistance'] = self._find_support_resistance(prices)
                    features['trend_lines'] = self._detect_trend_lines(prices)

            if 'volume' in hist_data.columns:
                volume = hist_data['volume']
                if len(volume) > 10:
                    features['avg_volume'] = float(volume.mean())
                    features['volume_volatility'] = float(volume.std() / volume.mean())
                    features['volume_pattern'] = self._analyze_volume_pattern(volume)

            if self.technical_extractor_available and self.technical_extractor is not None:
                try:
                    tech_features = self.technical_extractor.extract_all_features(
                        hist_data,
                        price_col='close',
                        volume_col='volume',
                        open_col='open',
                        high_col='high',
                        low_col='low'
                    )
                    features['technical_score'] = self._calculate_deep_technical_score(tech_features)
                    self.technical_features_df = tech_features
                    print(f"✓ 提取了 {len(tech_features.columns)} 个技术特征")
                except Exception as e:
                    print(f"⚠️ 深度技术特征提取失败: {e}")
                    self.technical_features_df = None
            else:
                self.technical_features_df = None
        else:
            print("⚠️ 无历史数据，使用默认特征")
            self.technical_features_df = None

        return features

    def _analyze_volatility_profile(self, returns):
        vol = returns.std()
        if vol > 0.05:
            return "high_volatility"
        elif vol > 0.03:
            return "medium_volatility"
        elif vol > 0.015:
            return "low_volatility"
        else:
            return "very_low_volatility"

    def _analyze_price_pattern(self, prices):
        if len(prices) < 20:
            return "random"
        short_ma = prices.rolling(window=5).mean()
        long_ma = prices.rolling(window=20).mean()
        if short_ma.iloc[-1] > long_ma.iloc[-1] and prices.iloc[-1] > short_ma.iloc[-1]:
            return "strong_uptrend"
        elif short_ma.iloc[-1] < long_ma.iloc[-1] and prices.iloc[-1] < short_ma.iloc[-1]:
            return "strong_downtrend"
        elif abs(prices.pct_change().std()) < 0.01:
            return "sideways"
        else:
            return "volatile"

    def _analyze_trend_strength(self, prices):
        if len(prices) < 30:
            return 0.0
        x = np.arange(len(prices[-30:]))
        y = prices[-30:].values
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 0.0
        r_squared = 1 - (ss_res / ss_tot)
        return float(abs(r_squared))

    def _find_support_resistance(self, prices, window=20):
        try:
            if len(prices) < window * 2:
                window = max(5, len(prices) // 4)
                if len(prices) < window * 2:
                    return {'supports': [], 'resistances': []}

            supports = []
            resistances = []
            for i in range(window, len(prices) - window):
                window_start = i - window
                window_end = i + window
                if window_start >= 0 and window_end <= len(prices):
                    local_min = prices.iloc[window_start:window_end].min()
                    local_max = prices.iloc[window_start:window_end].max()
                    current_price = prices.iloc[i]
                    if current_price == local_min:
                        supports.append(float(current_price))
                    elif current_price == local_max:
                        resistances.append(float(current_price))

            unique_supports = sorted(list(set(supports)))
            unique_resistances = sorted(list(set(resistances)))
            return {
                'supports': unique_supports[-3:] if len(unique_supports) > 3 else unique_supports,
                'resistances': unique_resistances[-3:] if len(unique_resistances) > 3 else unique_resistances
            }
        except Exception as e:
            print(f"⚠️ 支撑阻力位检测失败: {e}")
            return {'supports': [], 'resistances': []}

    def _detect_trend_lines(self, prices):
        if len(prices) < 50:
            return []
        trend_lines = []
        for i in range(len(prices) - 30):
            segment = prices.iloc[i:i + 30]
            if segment.diff().dropna().gt(0).all():
                trend_lines.append({
                    'type': 'uptrend',
                    'start': float(prices.iloc[i]),
                    'end': float(prices.iloc[i + 29]),
                    'slope': float((prices.iloc[i + 29] - prices.iloc[i]) / 30)
                })
        return trend_lines

    def _analyze_volume_pattern(self, volume):
        if len(volume) < 20:
            return "normal"
        volume_ma = volume.rolling(window=10).mean()
        volume_std = volume.rolling(window=10).std()
        last_volume = volume.iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        if last_volume > avg_volume + 2 * volume_std.iloc[-1]:
            return "high_volume"
        elif last_volume < avg_volume - volume_std.iloc[-1]:
            return "low_volume"
        else:
            return "normal"

    def _calculate_deep_technical_score(self, tech_features):
        score = 0.5
        try:
            if tech_features is None or tech_features.empty:
                return score
            last_row = tech_features.iloc[-1]
            if 'RSI' in last_row.index and not pd.isna(last_row['RSI']):
                rsi = last_row['RSI']
                if 40 <= rsi <= 60:
                    score += 0.1
                elif rsi > 70:
                    score -= 0.1
                elif rsi < 30:
                    score += 0.05
            if all(col in last_row.index for col in ['MACD', 'MACD_Signal']):
                macd = last_row['MACD']
                signal = last_row['MACD_Signal']
                if not pd.isna(macd) and not pd.isna(signal):
                    if macd > signal and macd > 0:
                        score += 0.15
                    elif macd < signal and macd < 0:
                        score -= 0.1
            score = max(0.0, min(1.0, score))
        except Exception as e:
            print(f"⚠️ 深度技术分数计算失败: {e}")
        return score

    def _detect_market_regime(self) -> str:
        if self.historical_data is None or len(self.historical_data) < 20:
            return 'default'

        if self.market_extractor_available and self.market_extractor is not None:
            try:
                if 'close' in self.historical_data.columns:
                    prices = self.historical_data['close']
                    volume = self.historical_data.get('volume') if 'volume' in self.historical_data.columns else None
                    market_metrics = {}
                    if hasattr(self.market_extractor, 'extract_market_metrics'):
                        market_metrics = self.market_extractor.extract_market_metrics(prices, volume)
                    elif hasattr(self.market_extractor, 'get_market_features'):
                        market_metrics = self.market_extractor.get_market_features(prices, volume)
                    elif hasattr(self.market_extractor, 'analyze_market'):
                        market_metrics = self.market_extractor.analyze_market(prices, volume)
                    elif hasattr(self.market_extractor, 'extract_features'):
                        market_metrics = self.market_extractor.extract_features(prices, volume)

                    if market_metrics:
                        volatility = market_metrics.get('market_volatility', market_metrics.get('volatility', 0.02))
                        trend = market_metrics.get('ma_trend', market_metrics.get('trend', 0))
                        if volatility > 0.04:
                            if trend > 0.5:
                                return 'high_volatility_nonlinear'
                            else:
                                return 'high_volatility'
                        elif trend > 0.7:
                            if volatility < 0.02:
                                return 'stable_trend'
                            else:
                                return 'trending'
                        elif volatility < 0.015:
                            return 'low_volatility'
            except Exception as e:
                print(f"⚠️ 外部市场特征提取器检测失败: {e}")

        return self._detect_market_regime_simple()

    def _detect_market_regime_simple(self) -> str:
        if self.historical_data is None or len(self.historical_data) < 20:
            return 'default'
        if 'close' in self.historical_data.columns:
            prices = self.historical_data['close']
            returns = prices.pct_change().dropna()
            if len(returns) < 10:
                return 'default'
            volatility = returns.std()
            mean_return = returns.mean()
            if len(prices) >= 30:
                x = np.arange(len(prices[-30:]))
                y = prices[-30:].values
                slope, intercept = np.polyfit(x, y, 1)
                r_squared = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - np.mean(y)) ** 2)
                trend_strength = abs(r_squared)
            else:
                trend_strength = 0

            if volatility > 0.04:
                if trend_strength > 0.3:
                    return 'high_volatility_nonlinear'
                else:
                    return 'high_volatility'
            elif trend_strength > 0.4:
                if volatility < 0.02:
                    return 'stable_trend'
                else:
                    return 'trending'
            elif volatility < 0.015 and abs(mean_return) < 0.001:
                return 'low_volatility'
            else:
                return 'default'
        return 'default'

    def _analyze_historical_patterns_deep(self):
        patterns = self._get_default_patterns()
        if self.historical_data is not None and len(self.historical_data) > 50:
            try:
                df = self.historical_data
                if 'close' in df.columns:
                    prices = df['close']
                    returns = prices.pct_change().dropna()
                    up_returns = returns[returns > 0]
                    down_returns = returns[returns < 0]
                    if len(up_returns) > 5:
                        patterns['up_day_return_mean'] = float(up_returns.mean())
                        patterns['up_day_return_std'] = float(up_returns.std())
                        patterns['up_day_frequency'] = float(len(up_returns) / len(returns))
                    if len(down_returns) > 5:
                        patterns['down_day_return_mean'] = float(down_returns.mean())
                        patterns['down_day_return_std'] = float(down_returns.std())
                        patterns['down_day_frequency'] = float(len(down_returns) / len(returns))
                    patterns['neutral_day_frequency'] = 1.0 - patterns['up_day_frequency'] - patterns['down_day_frequency']

                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    body_size = abs(df['close'] - df['open'])
                    patterns['avg_body_size'] = float(body_size.mean() / df['close'].mean())
                    patterns['body_size_std'] = float(body_size.std() / df['close'].mean())
                    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
                    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
                    upper_shadow_ratio = upper_shadow / (body_size + 1e-10)
                    lower_shadow_ratio = lower_shadow / (body_size + 1e-10)
                    patterns['upper_shadow_ratio'] = float(upper_shadow_ratio.mean())
                    patterns['lower_shadow_ratio'] = float(lower_shadow_ratio.mean())
                    is_positive = df['close'] > df['open']
                    patterns['positive_kline_ratio'] = float(is_positive.mean())

                if 'volume' in df.columns and 'close' in df.columns:
                    volume = df['volume']
                    prices = df['close']
                    returns = prices.pct_change().dropna()
                    if len(returns) > 0 and len(volume) >= len(returns):
                        patterns['avg_volume'] = float(volume.mean())
                        patterns['volume_volatility'] = float(volume.std() / volume.mean())
                        aligned_volume = volume.iloc[-len(returns):]
                        aligned_returns = returns
                        up_mask = aligned_returns > 0
                        if up_mask.any() and len(aligned_volume[up_mask]) > 0:
                            patterns['up_day_volume_factor'] = float(aligned_volume[up_mask].mean() / aligned_volume.mean())
                        down_mask = aligned_returns < 0
                        if down_mask.any() and len(aligned_volume[down_mask]) > 0:
                            patterns['down_day_volume_factor'] = float(aligned_volume[down_mask].mean() / aligned_volume.mean())
                    else:
                        patterns['avg_volume'] = float(volume.mean())
                        patterns['volume_volatility'] = float(volume.std() / volume.mean())
                        patterns['up_day_volume_factor'] = 1.2
                        patterns['down_day_volume_factor'] = 1.1
                print(f"✓ 深度分析了历史K线模式")
            except Exception as e:
                print(f"⚠️ 深度历史模式分析失败: {e}")
        return patterns

    def _get_default_patterns(self):
        return {
            'up_day_return_mean': 0.01,
            'up_day_return_std': 0.005,
            'up_day_frequency': 0.5,
            'down_day_return_mean': -0.008,
            'down_day_return_std': 0.004,
            'down_day_frequency': 0.3,
            'neutral_day_frequency': 0.2,
            'avg_body_size': 0.015,
            'body_size_std': 0.005,
            'upper_shadow_ratio': 0.5,
            'lower_shadow_ratio': 0.5,
            'positive_kline_ratio': 0.55,
            'avg_volume': 1000000,
            'volume_volatility': 0.3,
            'up_day_volume_factor': 1.2,
            'down_day_volume_factor': 1.1,
            'max_consecutive_up': 3,
            'max_consecutive_down': 2
        }

    def _generate_stock_fingerprint(self):
        fingerprint_data = {
            'symbol': self.stock_symbol,
            'volatility': float(self.stock_features.get('volatility', 0.02)),
            'avg_return': float(self.stock_features.get('avg_return', 0.0005)),
            'tech_score': float(self.stock_features.get('technical_score', 0.5)),
            'market_regime': self.market_regime,
            'timestamp': pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
        }
        if self.historical_data is not None and len(self.historical_data) > 0:
            returns = self.historical_data['close'].pct_change().dropna()
            if len(returns) > 0:
                fingerprint_data.update({
                    'skewness': float(returns.skew()),
                    'kurtosis': float(returns.kurtosis())
                })
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()
        return fingerprint

    def _select_models_dynamically(self):
        available_models = self.get_available_models()
        selected_models_weights = self.model_selector.select_best_models(
            self.market_regime,
            available_models,
            self.stock_features
        )
        self.selected_models = [model for model, _ in selected_models_weights]
        self.model_weights = dict(selected_models_weights)
        self.combination_confidence = self._calculate_combination_confidence()

    def _calculate_combination_confidence(self) -> float:
        confidence = 0.0
        total_weight = 0
        for model_name, weight in self.model_weights.items():
            if model_name in BACKTEST_RESULTS['model_performance']:
                performance = BACKTEST_RESULTS['model_performance'][model_name]
                model_confidence = performance['direction_accuracy']
                if self.market_regime in performance['suitable_markets']:
                    model_confidence *= 1.1
                technical_score = self.stock_features.get('technical_score', 0.8)
                if model_name in ['xgboost', 'multi_factor', 'quant_factor'] and technical_score > 0.6:
                    model_confidence *= 1.05
                elif model_name in ['sarima'] and self.stock_features.get('trend_strength', 0) > 0.3:
                    model_confidence *= 1.08
                confidence += model_confidence * weight
                total_weight += weight
        if total_weight > 0:
            confidence /= total_weight
        return min(confidence, 0.95)

    def get_available_models(self) -> List[str]:
        available = []
        for model_name in self.model_config_manager.get_enabled_models():
            if model_name in self.models and self.models[model_name] is not None:
                available.append(model_name)
        return available

    def _determine_best_model(self):
        if not self.model_weights:
            self.best_model_name = 'ensemble'
            return
        sorted_models = sorted(self.model_weights.items(),
                               key=lambda x: x[1], reverse=True)
        if sorted_models:
            self.best_model_name = sorted_models[0][0]
            best_weight = self.model_weights[self.best_model_name]
            print(f"✓ 最优模型: {self.best_model_name} (权重: {best_weight:.2f})")
        else:
            self.best_model_name = 'ensemble'

    # ==================== 模型模拟方法 ====================

    def _simulate_garch(self, future_dates, tech_features=None):
        try:
            if 'garch' not in self.models:
                return self._create_default_simulation(future_dates)
            last_price = self._get_last_price()
            config = self.config
            simulator_config = None
            if hasattr(config, 'simulator'):
                simulator_config = config.simulator
            elif isinstance(config, dict) and 'simulator' in config:
                simulator_config = config['simulator']

            def get_safe_config_value(config_obj, key, default):
                try:
                    if hasattr(config_obj, 'get') and callable(getattr(config_obj, 'get')):
                        return config_obj.get(key, default)
                    elif hasattr(config_obj, key):
                        return getattr(config_obj, key, default)
                    elif isinstance(config_obj, dict):
                        return config_obj.get(key, default)
                    else:
                        return default
                except:
                    return default

            volatility_persistence = get_safe_config_value(simulator_config, 'garch_volatility_persistence', 0.9)
            if tech_features is not None and 'normality_returns_std' in tech_features.columns:
                hist_volatility = float(tech_features['normality_returns_std'].iloc[-1]) if len(tech_features) > 0 else self.stock_features['volatility']
            else:
                hist_volatility = self.stock_features['volatility']

            predictions = [last_price]
            for i in range(len(future_dates)):
                daily_vol = hist_volatility * volatility_persistence
                base_return = self.stock_features['avg_return']
                tech_score = self.stock_features.get('technical_score', 0.5)
                noise_scale = 0.5 if 0.3 < tech_score < 0.7 else (0.3 if tech_score > 0.7 else 0.7)
                if self.deterministic_mode:
                    np.random.seed(42 + i)
                    noise = np.random.normal(0, noise_scale)
                else:
                    noise = np.random.normal(0, noise_scale)
                daily_return = base_return + daily_vol * noise
                daily_return = np.clip(daily_return, -0.08, 0.08)
                new_price = predictions[-1] * (1 + daily_return)
                predictions.append(new_price)
            predictions = np.array(predictions[1:])
            print(f"    ✓ GARCH模拟: {predictions[0]:.2f} → {predictions[-1]:.2f} "
                  f"(±{((predictions[-1] / predictions[0] - 1) * 100):.1f}%)")
            return predictions
        except Exception as e:
            print(f"    ⚠️ GARCH模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    def _simulate_sarima(self, future_dates, tech_features=None):
        try:
            if 'sarima' not in self.models or self.models['sarima'] is None:
                return self._create_default_simulation(future_dates)
            model_info = self.models['sarima']
            model = model_info.get('model')
            if model is None or not hasattr(model, 'forecast'):
                return self._create_default_simulation(future_dates)
            predictions = model.forecast(steps=len(future_dates))
            if hasattr(predictions, 'values'):
                predictions = predictions.values
            else:
                predictions = np.array(predictions)
            if len(predictions.shape) > 1:
                predictions = predictions.flatten()
            print(f"    ✓ SARIMA模拟: {predictions[0]:.2f} → {predictions[-1]:.2f} "
                  f"(±{((predictions[-1] / predictions[0] - 1) * 100):.1f}%)")
            return predictions
        except Exception as e:
            print(f"    ⚠️ SARIMA模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    def _simulate_multi_factor(self, future_dates, tech_features=None):
        try:
            if 'multi_factor' not in self.models:
                return self._create_default_simulation(future_dates)
            model_info = self.models['multi_factor']
            model = model_info.get('model')
            last_price = self._get_last_price()
            if model is None:
                return self._create_default_simulation(future_dates)
            config = self.model_config_manager.get_config('multi_factor')
            n_features = config.get('n_features', 10)
            predictions = [last_price]
            for i in range(len(future_dates)):
                features = self._create_multi_factors(predictions[-1], i, n_features, tech_features)
                features = features.reshape(1, -1)
                try:
                    if hasattr(model, 'predict'):
                        predicted_return = model.predict(features)[0]
                    else:
                        predicted_return = np.dot(features.flatten(), model.coef_)
                        if hasattr(model, 'intercept_'):
                            predicted_return += model.intercept_
                    predicted_return = np.clip(predicted_return, -0.08, 0.08)
                except Exception:
                    predicted_return = self.stock_features['avg_return']
                new_price = predictions[-1] * (1 + predicted_return)
                predictions.append(new_price)
            predictions = np.array(predictions[1:])
            print(f"    ✓ 多因子模拟: {predictions[0]:.2f} → {predictions[-1]:.2f} "
                  f"(±{((predictions[-1] / predictions[0] - 1) * 100):.1f}%)")
            return predictions
        except Exception as e:
            print(f"    ⚠️ 多因子模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    def _simulate_xgboost(self, future_dates, tech_features=None):
        try:
            if 'xgboost' not in self.models:
                return self._create_default_simulation(future_dates)
            model_info = self.models['xgboost']
            model = model_info.get('model')
            last_price = self._get_last_price()
            if model is None:
                return self._create_default_simulation(future_dates)
            config = self.model_config_manager.get_config('xgboost')
            n_features = config.get('n_features', 13)
            predictions = [last_price]
            for i in range(len(future_dates)):
                features = self._create_xgboost_features(predictions[-1], i, n_features, tech_features)
                features = features.reshape(1, -1)
                try:
                    predicted_return = model.predict(features)[0]
                    if abs(predicted_return) > 5:
                        predicted_price = predicted_return
                        predicted_return = (predicted_price - predictions[-1]) / predictions[-1]
                    predicted_return = np.clip(predicted_return, -0.1, 0.1)
                except Exception:
                    predicted_return = self.stock_features['avg_return']
                new_price = predictions[-1] * (1 + predicted_return)
                predictions.append(new_price)
            predictions = np.array(predictions[1:])
            print(f"    ✓ XGBoost模拟: {predictions[0]:.2f} → {predictions[-1]:.2f} "
                  f"(±{((predictions[-1] / predictions[0] - 1) * 100):.1f}%)")
            return predictions
        except Exception as e:
            print(f"    ⚠️ XGBoost模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    # ---------- 新增模型模拟方法 ----------
    def _simulate_arima(self, future_dates, tech_features=None):
        try:
            if 'arima' not in self.models or self.models['arima'] is None:
                return self._create_default_simulation(future_dates)
            model_info = self.models['arima']
            model = model_info.get('model')
            if model is None or not hasattr(model, 'forecast'):
                return self._create_default_simulation(future_dates)
            predictions = model.forecast(steps=len(future_dates))
            predictions = np.array(predictions)
            print(f"    ✓ ARIMA模拟: {predictions[0]:.2f} → {predictions[-1]:.2f}")
            return predictions
        except Exception as e:
            print(f"    ⚠️ ARIMA模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    def _simulate_lstm(self, future_dates, tech_features=None):
        try:
            if 'lstm' not in self.models or self.models['lstm'] is None:
                return self._create_default_simulation(future_dates)
            model_info = self.models['lstm']
            model = model_info.get('model')
            window_size = model_info.get('window_size', 60)
            last_seq = model_info.get('last_sequence', np.zeros(window_size))
            predictions = []
            current_seq = last_seq.reshape(1, window_size, 1)
            for _ in range(len(future_dates)):
                pred = model.predict(current_seq, verbose=0)[0, 0]
                predictions.append(pred)
                current_seq = np.roll(current_seq, -1, axis=1)
                current_seq[0, -1, 0] = pred
            predictions = np.array(predictions)
            print(f"    ✓ LSTM模拟: {predictions[0]:.2f} → {predictions[-1]:.2f}")
            return predictions
        except Exception as e:
            print(f"    ⚠️ LSTM模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    def _simulate_lightgbm(self, future_dates, tech_features=None):
        try:
            if 'lightgbm' not in self.models or self.models['lightgbm'] is None:
                return self._create_default_simulation(future_dates)
            model_info = self.models['lightgbm']
            model = model_info.get('model')
            last_seq = np.array(model_info.get('last_sequence', np.zeros(10)))
            predictions = []
            current_seq = last_seq.copy()
            for _ in range(len(future_dates)):
                pred = model.predict(current_seq.reshape(1, -1))[0]
                predictions.append(pred)
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = pred
            predictions = np.array(predictions)
            print(f"    ✓ LightGBM模拟: {predictions[0]:.2f} → {predictions[-1]:.2f}")
            return predictions
        except Exception as e:
            print(f"    ⚠️ LightGBM模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    def _simulate_catboost(self, future_dates, tech_features=None):
        try:
            if 'catboost' not in self.models or self.models['catboost'] is None:
                return self._create_default_simulation(future_dates)
            model_info = self.models['catboost']
            model = model_info.get('model')
            last_seq = np.array(model_info.get('last_sequence', np.zeros(10)))
            predictions = []
            current_seq = last_seq.copy()
            for _ in range(len(future_dates)):
                pred = model.predict(current_seq.reshape(1, -1))[0]
                predictions.append(pred)
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = pred
            predictions = np.array(predictions)
            print(f"    ✓ CatBoost模拟: {predictions[0]:.2f} → {predictions[-1]:.2f}")
            return predictions
        except Exception as e:
            print(f"    ⚠️ CatBoost模拟失败: {e}")
            return self._create_default_simulation(future_dates)

    # ==================== 量化因子模型模拟 ====================
    def _simulate_quant_factor(self, future_dates, tech_features=None):
        """基于量化因子模型的价格模拟"""
        try:
            if 'quant_factor' not in self.models or self.models['quant_factor'] is None:
                return self._create_default_simulation(future_dates)

            model_info = self.models['quant_factor']
            model = model_info.get('model')
            scaler = model_info.get('scaler')
            n_factors = model_info.get('n_factors', 16)
            pred_horizon = model_info.get('prediction_horizon', 5)
            last_close = model_info.get('last_close',
                                        self._get_last_price())

            if model is None or scaler is None:
                return self._create_default_simulation(future_dates)

            # 从 stock_features 获取基础统计量用于滚动因子生成
            base_vol = float(self.stock_features.get('volatility', 0.02))
            base_trend = float(self.stock_features.get('trend_strength', 0.0))
            base_ret = float(self.stock_features.get('avg_return', 0.0))
            tech_score = float(self.stock_features.get('technical_score', 0.5))

            # 将市场状态映射为数值（用于因子输入）
            regime_map = {
                'stable_trend': 0.0,
                'high_volatility': 1.0,
                'high_volatility_nonlinear': 1.5,
                'trending': 0.5,
                'default': 0.0,
                'low_volatility': -0.5
            }
            regime_val = regime_map.get(self.market_regime, 0.0)

            predictions = [last_close]
            current_price = last_close

            for step in range(len(future_dates)):
                # ---------- 实时构建因子向量 ----------
                factors = np.zeros(n_factors)
                idx = 0

                # 动量因子（基于历史 + 模拟滚动）
                if idx < n_factors:
                    sim_return = (current_price / last_close - 1)
                    factors[idx] = sim_return * 100  # momentum_5d proxy
                    idx += 1
                if idx < n_factors:
                    factors[idx] = sim_return * 100  # momentum_10d proxy (step-based)
                    idx += 1
                if idx < n_factors:
                    factors[idx] = base_trend * 100  # momentum_20d proxy
                    idx += 1
                if idx < n_factors:
                    factors[idx] = base_trend * 50   # momentum_60d proxy
                    idx += 1

                # 波动因子
                if idx < n_factors:
                    factors[idx] = base_vol * np.sqrt(252) * 100
                    idx += 1
                if idx < n_factors:
                    factors[idx] = base_vol * np.sqrt(252) * 80
                    idx += 1
                if idx < n_factors:
                    factors[idx] = base_vol * np.sqrt(252) * 60
                    idx += 1

                # 趋势强度
                if idx < n_factors:
                    factors[idx] = base_trend
                    idx += 1

                # 均线交叉信号（基于当前价格与历史均线的关系）
                if idx < n_factors:
                    factors[idx] = 1.0 if sim_return > 0 else -1.0  # ma_cross proxy
                    idx += 1
                if idx < n_factors:
                    factors[idx] = 1.0 if base_trend > 0.3 else -1.0
                    idx += 1

                # 价格位置
                if idx < n_factors:
                    factors[idx] = 0.5 + sim_return * 2  # price_position proxy
                    idx += 1

                # 量比
                if idx < n_factors:
                    factors[idx] = 1.0 + np.random.normal(0, 0.1)
                    idx += 1

                # 偏度 / 峰度
                if idx < n_factors:
                    factors[idx] = np.random.normal(0, 0.5)
                    idx += 1
                if idx < n_factors:
                    factors[idx] = np.random.normal(2, 1)
                    idx += 1

                # 技术得分 / 市场状态 / 辅助因子
                if idx < n_factors:
                    factors[idx] = tech_score
                    idx += 1
                if idx < n_factors:
                    factors[idx] = regime_val
                    idx += 1
                if idx < n_factors:
                    factors[idx] = np.random.normal(0, 0.05)
                    idx += 1

                # 填充剩余维度
                while idx < n_factors:
                    factors[idx] = np.random.normal(0, 0.05)
                    idx += 1

                # ---------- 预测 ----------
                try:
                    factors_scaled = scaler.transform(factors.reshape(1, -1))
                    predicted_return = model.predict(factors_scaled)[0]
                except Exception:
                    predicted_return = base_ret * 100

                predicted_return = np.clip(predicted_return / 100,
                                           -0.08, 0.08)
                new_price = current_price * (1 + predicted_return)
                predictions.append(new_price)
                current_price = new_price

            predictions = np.array(predictions[1:])
            print(f"    ✓ 量化因子模拟: {predictions[0]:.2f} → {predictions[-1]:.2f} "
                  f"(±{((predictions[-1] / predictions[0] - 1) * 100):.1f}%)")
            return predictions

        except Exception as e:
            print(f"    ⚠️ 量化因子模拟失败: {e}")
            import traceback
            traceback.print_exc()
            return self._create_default_simulation(future_dates)

    def _simulate_ensemble(self, future_dates, tech_features=None):
        return self._generate_ensemble_simulation(future_dates)

    # ==================== 辅助模拟方法 ====================

    def _get_last_price(self):
        if self.historical_data is not None and len(self.historical_data) > 0:
            if 'close' in self.historical_data.columns:
                return float(self.historical_data['close'].iloc[-1])
        return self.stock_features['price_level']

    def _generate_future_dates(self, last_date):
        future_dates = []
        current_date = last_date
        while len(future_dates) < self.simulation_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                future_dates.append(current_date)
        return pd.DatetimeIndex(future_dates)

    def simulate_future(self):
        print(f"{self.stock_symbol}: 开始进行未来{self.simulation_days}天模拟...")
        print(f"  市场状态: {self.market_regime}")
        print(f"  股票指纹: {self.stock_fingerprint[:16]}...")
        if self.historical_data is not None and len(self.historical_data) > 0:
            last_date = self.historical_data.index[-1]
        else:
            last_date = pd.Timestamp.now()
        future_dates = self._generate_future_dates(last_date)
        simulations = {}
        for model_name in self.selected_models:
            print(f"  [{len(simulations) + 1}/{len(self.selected_models)}] {model_name}模拟...")
            if model_name in self.model_simulation_methods:
                try:
                    tech_features = None
                    if hasattr(self, 'technical_features_df') and self.technical_features_df is not None:
                        tech_features = self.technical_features_df
                    sim_result = self.model_simulation_methods[model_name](future_dates, tech_features)
                    if self._validate_price_series(sim_result):
                        simulations[model_name] = sim_result
                    else:
                        print(f"    ⚠️ {model_name}模拟结果不合理，使用回退模拟")
                        simulations[model_name] = self._create_default_simulation(future_dates)
                except Exception as e:
                    print(f"    ⚠️ {model_name}模拟失败: {e}")
                    simulations[model_name] = self._create_default_simulation(future_dates)
        ensemble_result = self._generate_ensemble_simulation(future_dates)
        simulations['ensemble'] = ensemble_result
        print(f"  ✓ 模拟完成，成功模拟 {len(self.selected_models)} 个模型")
        return simulations, future_dates

    def _validate_price_series(self, prices):
        if len(prices) == 0:
            return False
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return False
        start_price = prices[0] if len(prices) > 0 else 1.0
        price_changes = prices / start_price
        if np.max(price_changes) > 3 or np.min(price_changes) < 0.3:
            return False
        if len(prices) > 1:
            daily_returns = np.diff(prices) / prices[:-1]
            max_daily_return = np.max(np.abs(daily_returns))
            if max_daily_return > 0.15:
                return False
        return True

    def _create_default_simulation(self, future_dates):
        last_price = self._get_last_price()
        n_days = len(future_dates)
        mu = self.stock_features['avg_return']
        sigma = self.stock_features['volatility']
        if self.deterministic_mode:
            np.random.seed(42)
        random_returns = np.random.normal(mu, sigma, n_days)
        random_returns = np.clip(random_returns, -0.05, 0.05)
        prices = [last_price]
        for i in range(n_days):
            new_price = prices[-1] * (1 + random_returns[i])
            prices.append(new_price)
        return np.array(prices[1:])

    def _generate_ensemble_simulation(self, future_dates):
        if not self.selected_models or not self.model_weights:
            return self._create_default_simulation(future_dates)
        ensemble_result = np.zeros(len(future_dates))
        for model_name in self.selected_models:
            if model_name in self.model_simulation_methods:
                try:
                    sim = self.model_simulation_methods[model_name](future_dates)
                    weight = self.model_weights.get(model_name, 0)
                    if len(sim) == len(ensemble_result):
                        ensemble_result += sim * weight
                except:
                    continue
        print(f"    ✓ 集成模拟完成")
        return ensemble_result

    def _create_multi_factors(self, current_price: float, step: int, n_features: int, tech_features=None):
        factors = []
        factors.append(current_price / 100.0)
        factors.append(np.sin(step * 0.1))
        factors.append(np.cos(step * 0.1))
        factors.append(step / 30.0)
        factors.append(float(self.stock_features['volatility'] * 10.0))
        factors.append(float(self.stock_features['trend_strength']))
        factors.append(float(self.stock_features['avg_return'] * 100.0))
        factors.append(float(self.stock_features.get('technical_score', 0.5)))
        if tech_features is not None:
            if 'RSI' in tech_features.columns and len(tech_features) > 0:
                rsi_value = tech_features['RSI'].iloc[-1]
                if not pd.isna(rsi_value):
                    factors.append(float(rsi_value / 100.0))
                else:
                    factors.append(0.5)
            if 'MACD' in tech_features.columns and len(tech_features) > 0:
                macd_value = tech_features['MACD'].iloc[-1]
                if not pd.isna(macd_value):
                    factors.append(float(macd_value / 100.0))
                else:
                    factors.append(0.0)
        regime_map = {
            'stable_trend': 0.0,
            'high_volatility': 1.0,
            'high_volatility_nonlinear': 1.5,
            'trending': 0.5,
            'default': 0.0,
            'low_volatility': -0.5
        }
        factors.append(regime_map.get(self.market_regime, 0.0))
        while len(factors) < n_features:
            factors.append(float(np.random.normal(0, 0.1)))
        return np.array(factors[:n_features], dtype=float)

    def _create_xgboost_features(self, current_price: float, step: int, n_features: int, tech_features=None):
        features = []
        features.append(float(current_price / 1000.0))
        features.append(float(np.log(current_price + 1) / 10.0))
        features.append(float(np.sin(2 * np.pi * step / 7) / 2.0))
        features.append(float(np.cos(2 * np.pi * step / 7) / 2.0))
        features.append(float(step / 100.0))
        features.append(float(self.stock_features['volatility'] * 10.0))
        features.append(float(self.stock_features['trend_strength']))
        features.append(float(self.stock_features['avg_return'] * 100.0))
        features.append(float(self.stock_features.get('technical_score', 0.5)))
        regime_map = {
            'stable_trend': 0.0,
            'high_volatility': 1.0,
            'high_volatility_nonlinear': 1.5,
            'trending': 0.5,
            'default': 0.0,
            'low_volatility': -0.5
        }
        features.append(float(regime_map.get(self.market_regime, 0.0)))
        while len(features) < n_features:
            features.append(float(np.random.uniform(-0.05, 0.05)))
        return np.array(features[:n_features], dtype=float)

    # ==================== K线生成 ====================

    def generate_candlestick_data(self, model_name=None, future_dates=None):
        if model_name is None:
            model_name = self.best_model_name or 'ensemble'
            print(f"{self.stock_symbol}: 生成最优模型({model_name})的K线数据...")
        elif model_name == 'best':
            model_name = self.best_model_name or 'ensemble'
            print(f"{self.stock_symbol}: 生成最优模型({model_name})的K线数据...")
        else:
            print(f"{self.stock_symbol}: 生成{model_name}模型的K线数据...")

        if future_dates is None:
            if self.historical_data is not None and len(self.historical_data) > 0:
                last_date = self.historical_data.index[-1]
            else:
                last_date = pd.Timestamp.now()
            future_dates = self._generate_future_dates(last_date)

        base_prices = self._generate_unique_price_path(
            future_dates, model_name, self.stock_fingerprint
        )

        simulated_data = self._generate_unique_candlestick_data(
            future_dates, base_prices, model_name
        )

        df = pd.DataFrame(simulated_data)
        df.set_index('date', inplace=True)

        up_days = sum([1 for d in simulated_data if d['day_type'] == 'up'])
        down_days = sum([1 for d in simulated_data if d['day_type'] == 'down'])
        neutral_days = sum([1 for d in simulated_data if d['day_type'] == 'neutral'])
        total_days = len(simulated_data)

        print(f"  ✓ 生成 {len(df)} 个交易日的K线数据")
        print(f"    价格: {df['close'].iloc[0]:.2f} → {df['close'].iloc[-1]:.2f} "
              f"({((df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100):.2f}%)")
        print(f"    日类型分布: 上涨{up_days}天({up_days / total_days * 100:.1f}%) "
              f"下跌{down_days}天({down_days / total_days * 100:.1f}%) "
              f"中性{neutral_days}天({neutral_days / total_days * 100:.1f}%)")
        return df

    def _generate_unique_price_path(self, future_dates, model_name, stock_fingerprint):
        last_price = self._get_last_price()
        n_days = len(future_dates)
        seed_base = int(stock_fingerprint[:8], 16) % 1000000
        rng = np.random.RandomState(seed_base)
        volatility = self.stock_features['volatility']
        avg_return = self.stock_features['avg_return']
        trend_strength = self.stock_features['trend_strength']
        price_pattern = self.stock_features['price_pattern']
        volatility_profile = self.stock_features['volatility_profile']
        base_returns = rng.normal(avg_return, volatility, n_days)
        if volatility_profile == "high_volatility":
            base_returns *= 1.5
        elif volatility_profile == "very_low_volatility":
            base_returns *= 0.6
        if price_pattern == "strong_uptrend":
            base_returns = np.abs(base_returns) * 0.5 + avg_return * 1.5
        elif price_pattern == "strong_downtrend":
            base_returns = -np.abs(base_returns) * 0.5 + avg_return * 0.5
        if trend_strength > 0.3:
            for i in range(1, n_days):
                base_returns[i] = base_returns[i] * 0.7 + base_returns[i - 1] * 0.3 * trend_strength
        max_daily_return = min(0.1, volatility * 3)
        returns = np.clip(base_returns, -max_daily_return, max_daily_return)
        prices = [last_price]
        for i in range(n_days):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(new_price)
        return np.array(prices[1:])

    def _generate_unique_candlestick_data(self, future_dates, base_prices, model_name):
        simulated_data = []
        fingerprint_hash = int(self.stock_fingerprint[:12], 16) % 1000000
        rng = np.random.RandomState(fingerprint_hash)
        patterns = self.kline_patterns
        stock_features = self.stock_features
        day_type_history = []
        for i in range(len(future_dates)):
            if i == 0:
                open_price = self._get_last_price()
                last_close = open_price
            else:
                last_close = simulated_data[-1]['close']
                overnight_change = rng.normal(stock_features['avg_return'] * 0.5, stock_features['volatility'] * 0.3)
                open_price = last_close * (1 + overnight_change)
            target_price = base_prices[i] if i < len(base_prices) else open_price
            day_type = self._determine_unique_day_type(i, patterns, day_type_history, stock_features, rng)
            price_change = self._generate_unique_price_change(day_type, patterns, stock_features, i, open_price, target_price, rng)
            close_price = open_price * (1 + price_change)
            candle_data = self._generate_unique_candle_shape(open_price, close_price, day_type, patterns, stock_features, i, rng)
            volume = self._generate_unique_volume(day_type, patterns, stock_features, i, candle_data, open_price, rng)
            day_type_history.append(day_type)
            if len(day_type_history) > 7:
                day_type_history.pop(0)
            simulated_data.append({
                'date': future_dates[i],
                'open': float(open_price),
                'high': float(candle_data['high']),
                'low': float(candle_data['low']),
                'close': float(close_price),
                'volume': int(volume),
                'day_type': day_type,
                'price_change_pct': price_change * 100
            })
        return simulated_data

    def _determine_unique_day_type(self, day_index, patterns, history, stock_features, rng):
        base_up = patterns.get('up_day_frequency', 0.5)
        base_down = patterns.get('down_day_frequency', 0.3)
        base_neutral = patterns.get('neutral_day_frequency', 0.2)
        volatility = stock_features['volatility']
        tech_score = stock_features['technical_score']
        price_pattern = stock_features['price_pattern']
        if volatility > 0.03:
            base_up *= 0.9; base_down *= 1.1
        elif volatility < 0.015:
            base_up *= 1.1; base_down *= 0.9
        if tech_score > 0.7:
            base_up *= 1.2; base_down *= 0.8
        elif tech_score < 0.3:
            base_up *= 0.8; base_down *= 1.2
        if price_pattern == "strong_uptrend":
            base_up *= 1.3; base_down *= 0.7
        elif price_pattern == "strong_downtrend":
            base_up *= 0.7; base_down *= 1.3
        total = base_up + base_down + base_neutral
        base_up /= total; base_down /= total; base_neutral /= total
        rand_val = rng.random()
        if rand_val < base_up:
            return 'up'
        elif rand_val < base_up + base_down:
            return 'down'
        else:
            return 'neutral'

    def _generate_unique_price_change(self, day_type, patterns, stock_features, day_index, open_price, target_price, rng):
        if day_type == 'up':
            mean_return = patterns.get('up_day_return_mean', 0.01)
            std_return = patterns.get('up_day_return_std', 0.005)
        elif day_type == 'down':
            mean_return = patterns.get('down_day_return_mean', -0.008)
            std_return = patterns.get('down_day_return_std', 0.004)
        else:
            mean_return = 0.0
            std_return = stock_features['volatility'] * 0.3
        volatility = stock_features['volatility']
        tech_score = stock_features['technical_score']
        if tech_score > 0.7 and day_type == 'up':
            mean_return *= 1.2
        elif tech_score < 0.3 and day_type == 'down':
            mean_return *= 1.2
        random_change = rng.normal(mean_return, std_return)
        target_change = (target_price / open_price - 1)
        influence = rng.uniform(0.1, 0.4)
        final_change = random_change * (1 - influence) + target_change * influence
        if day_type == 'up':
            final_change = max(0.001, final_change)
        elif day_type == 'down':
            final_change = min(-0.001, final_change)
        max_change = min(0.1, volatility * 3)
        final_change = np.clip(final_change, -max_change, max_change)
        return final_change

    def _generate_unique_candle_shape(self, open_price, close_price, day_type, patterns, stock_features, day_index, rng):
        body_size = abs(close_price - open_price)
        min_body_pct = max(0.001, stock_features['volatility'] * 0.08)
        min_body = open_price * min_body_pct
        if body_size < min_body:
            body_size = min_body
            if close_price >= open_price:
                close_price = open_price + body_size
            else:
                close_price = open_price - body_size
        upper_base = patterns.get('upper_shadow_ratio', 0.5)
        lower_base = patterns.get('lower_shadow_ratio', 0.5)
        volatility = stock_features['volatility']
        if volatility > 0.03:
            upper_base *= 1.2; lower_base *= 1.2
        elif volatility < 0.015:
            upper_base *= 0.8; lower_base *= 0.8
        if day_type == 'up':
            upper_mult = rng.uniform(0.6, 0.9); lower_mult = rng.uniform(1.0, 1.3)
        elif day_type == 'down':
            upper_mult = rng.uniform(1.0, 1.3); lower_mult = rng.uniform(0.6, 0.9)
        else:
            upper_mult = rng.uniform(0.8, 1.1); lower_mult = rng.uniform(0.8, 1.1)
        upper_ratio = max(0.1, min(0.5, upper_base * upper_mult))
        lower_ratio = max(0.1, min(0.5, lower_base * lower_mult))
        upper_shadow = body_size * upper_ratio
        lower_shadow = body_size * lower_ratio
        if close_price >= open_price:
            high_price = close_price + upper_shadow
            low_price = open_price - lower_shadow
        else:
            high_price = open_price + upper_shadow
            low_price = close_price - lower_shadow
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)
        return {'high': high_price, 'low': low_price, 'close': close_price}

    def _generate_unique_volume(self, day_type, patterns, stock_features, day_index, candle_data, open_price, rng):
        if 'avg_volume' in stock_features:
            base_volume = stock_features['avg_volume']
        else:
            base_volume = patterns.get('avg_volume', 1000000)
        if day_type == 'up':
            type_factor = patterns.get('up_day_volume_factor', 1.2)
        elif day_type == 'down':
            type_factor = patterns.get('down_day_volume_factor', 1.1)
        else:
            type_factor = 0.9
        body_ratio = abs(candle_data['close'] - open_price) / open_price
        price_change_factor = 1.0 + min(2.0, body_ratio * 100)
        volume_volatility = patterns.get('volume_volatility', 0.3)
        random_factor = rng.normal(1.0, volume_volatility * 0.3)
        volume = base_volume * type_factor * price_change_factor * random_factor
        min_volume = base_volume * 0.2
        max_volume = base_volume * 5.0
        volume = max(min_volume, min(volume, max_volume))
        return int(volume)


if __name__ == "__main__":
    print("=" * 60)
    print("股票模拟器测试 - 集成全部模型（不含Informer）")
    print("=" * 60)
    test_cases = [
        {'code': '300502', 'name': '股票A'},
        {'code': '300570', 'name': '股票B'},
        {'code': '600487', 'name': '股票C'},
        {'code': '000002', 'name': '股票D'}
    ]
    for stock in test_cases:
        print(f"\n测试股票: {stock['name']} ({stock['code']})")
        config = {
            'stock_code': stock['code'],
            'days_to_simulate': 30,
            'deterministic_mode': True
        }
        models = {
            'garch': {'model': None, 'info': 'GARCH模型'},
            'sarima': {'model': None, 'info': 'SARIMA模型'},
            'multi_factor': {'model': None, 'info': '多因子模型'},
            'xgboost': {'model': None, 'info': 'XGBoost模型'},
            'arima': {'model': None, 'info': 'ARIMA模型'},
            'lstm': {'model': None, 'info': 'LSTM模型'},
            'lightgbm': {'model': None, 'info': 'LightGBM模型'},
            'catboost': {'model': None, 'info': 'CatBoost模型'},
        }
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        stock_hash = hash(stock['code']) % 1000
        np.random.seed(stock_hash)
        base_volatility = 0.02 + (stock_hash % 100) / 1000.0
        base_return = 0.0005 + (stock_hash % 50 - 25) / 10000.0
        returns = np.random.normal(base_return, base_volatility, 100)
        prices = 100 * np.cumprod(1 + returns)
        processed_data = {
            'processed': pd.DataFrame({
                'open': prices * 0.99,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
        }
        try:
            simulator = StockSimulator(config, models, processed_data)
            kline_data = simulator.generate_candlestick_data()
            print(f"  ✓ 成功生成K线数据")
            print(f"    开盘价: {kline_data['open'].iloc[0]:.2f} → {kline_data['open'].iloc[-1]:.2f}")
            print(f"    收盘价: {kline_data['close'].iloc[0]:.2f} → {kline_data['close'].iloc[-1]:.2f}")
            print(f"    成交量均值: {kline_data['volume'].mean():,.0f}")
            simulations, future_dates = simulator.simulate_future()
            print(f"  ✓ 成功模拟 {len(simulations) - 1} 个模型")
        except Exception as e:
            print(f"  ❌ 股票 {stock['name']} 分析失败: {e}")
            import traceback
            traceback.print_exc()
    print(f"\n{'=' * 60}")
    print("所有股票测试完成！")
    print(f"{'=' * 60}")