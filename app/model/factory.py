# model_factory.py
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
from pykalman import KalmanFilter

# 动态导入statsmodels的SARIMAX
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False
    print("⚠️ statsmodels SARIMAX不可用，将跳过SARIMA模型")

# 新增模型相关库的检测
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("⚠️ ARIMA模型库不可用，将跳过ARIMA")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow/Keras不可用，将跳过LSTM")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    print("⚠️ LightGBM库不可用，将跳过LightGBM")

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False
    print("⚠️ CatBoost库不可用，将跳过CatBoost")


class ModelFactory:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.market_regime = None  # 市场状态标记

        # 从config_loader获取参数配置器
        from config_loader import get_config_loader
        self.param_loader = get_config_loader()

        # 如果配置器没有加载配置文件，尝试加载默认配置
        if not self.param_loader.loaded:
            # 尝试加载默认配置文件
            default_config = "backtest_results/parameter_recommendations/parameter_recommendations.txt"
            self.param_loader.load_config_file(default_config)

        # 初始化动态配置
        self._init_dynamic_config()

        # 初始化参数推荐
        self._init_parameter_recommendations()

    def _init_parameter_recommendations(self):
        """初始化基于回测的参数推荐"""
        self.parameter_recommendations = {
            'garch': {
                'p': 1,
                'q': 2,
                'window_size': 50,
                'prediction_horizon': 5,
                'min_data_required': 200
            },
            'xgboost': {
                'n_estimators': 50,
                'learning_rate': 0.01,
                'max_depth': 5,
                'window_size': 50,
                'prediction_horizon': 5,
                'min_data_required': 100
            },
            'sarima': {
                'window_size': 100,
                'prediction_horizon': 5,
                'min_data_required': 150
            },
            'multi_factor': {
                'window_size': 150,
                'prediction_horizon': 10,
                'min_data_required': 100,
                'n_features': 10
            },
            'arima': {
                'window_size': 100,
                'prediction_horizon': 5,
                'min_data_required': 100
            },
            'lstm': {
                'window_size': 60,
                'prediction_horizon': 5,
                'epochs': 50,
                'batch_size': 32,
                'units': 64,
                'min_data_required': 200
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.05,
                'max_depth': 7,
                'window_size': 50,
                'prediction_horizon': 5,
                'min_data_required': 100
            },
            'catboost': {
                'iterations': 100,
                'learning_rate': 0.05,
                'depth': 6,
                'window_size': 50,
                'prediction_horizon': 5,
                'min_data_required': 100
            }
        }

    def _init_dynamic_config(self):
        """初始化动态配置"""
        self.model_selection_config = self._get_model_selection_config()
        self.model_weight_config = self._get_model_weight_config()

    def _get_param_value(self, section, key, default):
        """统一获取参数值的方法，适配 ParameterConfigLoader 接口"""
        try:
            if section in self.parameter_recommendations and key in self.parameter_recommendations[section]:
                return self.parameter_recommendations[section][key]

            if section in ['sarima', 'garch', 'xgboost', 'multi_factor', 'arima', 'lstm', 'lightgbm', 'catboost']:
                recommendation = self.param_loader.get_model_recommendation(section)
                if recommendation and recommendation.parameters and key in recommendation.parameters:
                    return recommendation.parameters[key]

            general_param = self.param_loader.get_general_parameter(key)
            if general_param:
                if hasattr(general_param, 'recommended_value'):
                    return general_param.recommended_value
                elif hasattr(general_param, 'value'):
                    return general_param.value
                elif isinstance(general_param, (int, float, str, bool)):
                    return general_param
                else:
                    try:
                        return str(general_param)
                    except:
                        return default
        except Exception as e:
            print(f"⚠️ 获取参数值时出错: {e}")
        return default

    def _get_model_params(self, model_name):
        try:
            if model_name in self.parameter_recommendations:
                return self.parameter_recommendations[model_name]
            recommendation = self.param_loader.get_model_recommendation(model_name)
            if recommendation and recommendation.parameters:
                return recommendation.parameters
        except:
            pass
        return {}

    def train_all_models(self, processed_data):
        if processed_data is None:
            print("❌ 输入数据为None")
            return self.models
        if 'prepared' not in processed_data:
            print("❌ processed_data中没有'prepared'键")
            return self.models

        prepared_data = processed_data['prepared']
        if 'close_scaled' not in prepared_data:
            print("❌ prepared_data中没有'close_scaled'数据")
            return self.models

        try:
            close_scaled = prepared_data['close_scaled']
            returns = prepared_data.get('returns')

            if 'processed' in processed_data:
                df = processed_data['processed']
                if df is not None and len(df) > 0:
                    close_prices = df['close'].values
                    volume = df['volume'].values if 'volume' in df.columns else None
                    high = df['high'].values if 'high' in df.columns else None
                    low = df['low'].values if 'low' in df.columns else None
                    open_prices = df['open'].values if 'open' in df.columns else None
                else:
                    close_prices = volume = high = low = open_prices = None
            else:
                close_prices = volume = high = low = open_prices = None

            if close_scaled is None or len(close_scaled) == 0:
                print("❌ close_scaled数据为空")
                return self.models

            self._analyze_market_regime(df, returns)
            self._select_and_train_models(df, close_scaled, returns, close_prices, volume, high, low, open_prices)

            successful_models = [name for name, model in self.models.items() if model is not None]
            if successful_models:
                print(f"  可用模型: {', '.join(successful_models)}")
            return self.models
        except Exception as e:
            print(f"❌ 模型训练过程出错: {e}")
            import traceback
            traceback.print_exc()
            return self.models

    def _analyze_market_regime(self, df, returns):
        try:
            if df is None or returns is None:
                self.market_regime = 'default'
                return

            volatility_threshold = self._get_param_value("market_analysis", "volatility_threshold", 0.03)
            trend_strength_threshold = self._get_param_value("market_analysis", "trend_strength_threshold", 0.7)
            autocorrelation_threshold = self._get_param_value("market_analysis", "autocorrelation_threshold", 0.5)
            skewness_threshold = self._get_param_value("market_analysis", "skewness_threshold", 0.5)
            kurtosis_threshold = self._get_param_value("market_analysis", "kurtosis_threshold", 3)
            adf_significance = self._get_param_value("market_analysis", "adf_significance", 0.05)

            try:
                from scipy import stats
            except ImportError:
                print("⚠️ scipy.stats不可用，使用简化统计计算")
                self.market_regime = 'default'
                return

            volatility = np.std(returns) if len(returns) > 0 else 0
            trend_strength = self._calculate_trend_strength(df['close'].values)
            autocorrelation = self._calculate_autocorrelation(returns)
            skewness = stats.skew(returns) if len(returns) > 0 else 0
            kurtosis = stats.kurtosis(returns) if len(returns) > 0 else 0

            adf_result = adfuller(df['close'].dropna())
            is_stationary = adf_result[1] < adf_significance

            if volatility > volatility_threshold:
                if abs(skewness) > skewness_threshold or kurtosis > kurtosis_threshold:
                    self.market_regime = 'high_volatility_nonlinear'
                else:
                    self.market_regime = 'high_volatility'
            elif trend_strength > trend_strength_threshold:
                self.market_regime = 'stable_trend' if is_stationary else 'trending'
            else:
                self.market_regime = 'default'
        except Exception as e:
            print(f"市场状态分析失败: {e}")
            self.market_regime = 'default'

    def _calculate_trend_strength(self, prices):
        min_data_length = self._get_param_value("trend_strength", "min_data_length", 20)
        if len(prices) < min_data_length:
            return 0
        try:
            from scipy import stats
        except ImportError:
            return 0
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        return abs(r_value)

    def _calculate_autocorrelation(self, returns, lag=1):
        min_data_length = self._get_param_value("autocorrelation", "min_data_length", 10)
        if len(returns) < lag + min_data_length:
            return 0
        return np.corrcoef(returns[:-lag], returns[lag:])[0, 1] if not np.isnan(returns).all() else 0

    def _select_and_train_models(self, df, close_scaled, returns, close_prices, volume, high, low, open_prices):
        model_train_methods = {
            'sarima': lambda: self._train_sarima(close_scaled),
            'garch': lambda: self._train_garch(returns),
            'xgboost': lambda: self._train_xgboost(close_prices, returns, volume),
            'multi_factor': lambda: self._train_multi_factor(close_prices, volume, high, low, open_prices),
            'arima': lambda: self._train_arima(close_prices),
            'lstm': lambda: self._train_lstm(close_prices),
            'lightgbm': lambda: self._train_lightgbm(close_prices, returns, volume),
            'catboost': lambda: self._train_catboost(close_prices, returns, volume),
            # ensemble 不需要单独训练，仅用于模拟组合
        }

        models_to_train = self.model_selection_config.get(self.market_regime,
                                                          self.model_selection_config.get('default',
                                                                                          ['xgboost', 'multi_factor', 'garch', 'arima', 'lightgbm']))
        print(f"根据市场状态 '{self.market_regime}' 选择模型: {', '.join(models_to_train)}")

        for i, model_name in enumerate(models_to_train):
            print(f"[{i + 1}/{len(models_to_train)}] 训练{model_name}模型...")
            try:
                if model_name in model_train_methods:
                    model_train_methods[model_name]()
                elif model_name == 'ensemble':
                    # 集成模型不单独训练，将其标记为可用
                    self.models['ensemble'] = {'status': 'ready', 'description': '动态集成模型'}
                    print(f"  ✓ 集成模型已就绪")
                else:
                    print(f"  ⚠️ 模型 {model_name} 不在可训练模型列表中")
            except Exception as e:
                print(f"  ⚠️ {model_name}训练失败: {e}")

    def _get_model_selection_config(self):
        try:
            from config_loader import get_config_loader
            config_loader = get_config_loader()
            if hasattr(config_loader, 'model_selection') and config_loader.model_selection:
                return config_loader.model_selection
        except:
            pass

        return {
            'high_volatility_nonlinear': ['garch', 'xgboost', 'lightgbm', 'catboost'],
            'high_volatility': ['garch', 'multi_factor', 'arima'],
            'trending': ['sarima', 'multi_factor', 'arima', 'lightgbm'],
            'stable_trend': ['sarima', 'garch', 'arima', 'catboost'],
            'default': ['sarima', 'garch', 'multi_factor', 'xgboost', 'arima', 'lightgbm', 'catboost']
        }

    def _get_model_weight_config(self):
        try:
            from config_loader import get_config_loader
            config_loader = get_config_loader()
            if hasattr(config_loader, 'model_weights') and config_loader.model_weights:
                return config_loader.model_weights
        except:
            pass

        return {
            'high_volatility_nonlinear': [('xgboost', 0.30), ('lightgbm', 0.25), ('catboost', 0.25), ('garch', 0.20)],
            'high_volatility': [('garch', 0.40), ('multi_factor', 0.35), ('arima', 0.25)],
            'trending': [('sarima', 0.35), ('multi_factor', 0.30), ('arima', 0.20), ('lightgbm', 0.15)],
            'stable_trend': [('sarima', 0.40), ('garch', 0.30), ('arima', 0.20), ('catboost', 0.10)],
            'default': [('sarima', 0.20), ('garch', 0.20), ('xgboost', 0.15), ('multi_factor', 0.15),
                        ('arima', 0.10), ('lightgbm', 0.10), ('catboost', 0.10)]
        }

    # ==================== 原有模型训练方法（完整保留） ====================

    def _train_sarima(self, close_scaled):
        """训练SARIMA模型 - 基于推荐参数优化"""
        if not SARIMAX_AVAILABLE:
            print("  ⚠️ SARIMAX不可用，跳过SARIMA模型")
            self.models['sarima'] = None
            return

        try:
            sarima_params = self._get_model_params('sarima')
            min_data_required = sarima_params.get('min_data_required', 150)
            window_size = sarima_params.get('window_size', 100)
            prediction_horizon = sarima_params.get('prediction_horizon', 5)

            if close_scaled is None or len(close_scaled) < min_data_required:
                print(f"  ⚠️ 数据不足（最少需要{min_data_required}个点），跳过SARIMA模型")
                self.models['sarima'] = None
                return

            data = close_scaled.flatten() if close_scaled.ndim > 1 else close_scaled
            data_series = pd.Series(data)
            if data_series.isna().any():
                print(f"  ℹ️ 发现{data_series.isna().sum()}个NaN值，进行填充")
                data_series = data_series.fillna(method='ffill').fillna(method='bfill')

            print(f"  ℹ️ SARIMA数据准备完成: {len(data_series)}个有效点，窗口大小: {window_size}")

            def check_stationarity(data, significance=0.05):
                adf_result = adfuller(data.dropna())
                return {
                    'is_stationary': adf_result[1] < significance,
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'n_lags': adf_result[2],
                    'n_obs': adf_result[3],
                    'critical_values': adf_result[4]
                }

            stationarity_check = check_stationarity(data_series)
            seasonal_period = self._detect_seasonality(data_series.values)

            if seasonal_period <= 1:
                print("  ⚠️ 未检测到明显季节性，使用ARIMA而非SARIMA")
                self.models['sarima'] = None
                return

            print(f"  ✓ 确定季节性周期: {seasonal_period}")

            def find_optimal_order(data, seasonal_period, max_p=2, max_d=1, max_q=2, max_sp=1, max_sd=1, max_sq=1):
                import itertools
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warnings.simplefilter('ignore', ConvergenceWarning)

                def determine_differencing(data, max_d=1):
                    for d in range(max_d + 1):
                        if d == 0:
                            diff_data = data
                        else:
                            diff_data = np.diff(data, n=d)
                        adf_result = adfuller(diff_data.dropna() if isinstance(diff_data, pd.Series) else diff_data)
                        if adf_result[1] < 0.05:
                            return d, diff_data
                    return max_d, np.diff(data, n=max_d)

                d, stationary_data = determine_differencing(data, max_d)
                seasonal_d = 1
                best_aic = np.inf
                best_order = (1, d, 1)
                best_seasonal_order = (1, seasonal_d, 1, seasonal_period)

                p_values = range(0, min(max_p + 1, 2))
                q_values = range(0, min(max_q + 1, 2))
                P_values = range(0, min(max_sp + 1, 1))
                Q_values = range(0, min(max_sq + 1, 1))

                param_combinations = list(itertools.product(p_values, q_values, P_values, Q_values))
                print(f"  ⏳ 搜索{len(param_combinations)}组参数组合...")

                for i, (p, q, P, Q) in enumerate(param_combinations[:10]):
                    try:
                        model = SARIMAX(
                            data,
                            order=(p, d, q),
                            seasonal_order=(P, seasonal_d, Q, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=True
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            results = model.fit(disp=False, maxiter=100)
                        current_aic = results.aic
                        if current_aic < best_aic:
                            best_aic = current_aic
                            best_order = (p, d, q)
                            best_seasonal_order = (P, seasonal_d, Q, seasonal_period)
                    except:
                        continue

                print(f"  ✓ 最优参数: SARIMA{best_order}x{best_seasonal_order}, AIC={best_aic:.2f}")
                return best_order, best_seasonal_order, best_aic

            if len(data_series) > 200:
                best_order, best_seasonal_order, best_aic = find_optimal_order(data_series, seasonal_period)
            else:
                best_order = (1, 1, 1)
                best_seasonal_order = (1, 1, 1, seasonal_period)
                best_aic = None

            try:
                print(f"  ⏳ 训练SARIMA{best_order}x{best_seasonal_order}模型...")
                sarima_model = SARIMAX(
                    data_series,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    simple_differencing=True
                )
                sarima_fit = sarima_model.fit(disp=False, maxiter=200, method='lbfgs', cov_type='robust')

                self.models['sarima'] = {
                    'model': sarima_fit,
                    'model_summary': sarima_fit.summary(),
                    'params': sarima_fit.params,
                    'aic': sarima_fit.aic,
                    'bic': sarima_fit.bic,
                    'hqic': sarima_fit.hqic,
                    'seasonal_period': seasonal_period,
                    'order': best_order,
                    'seasonal_order': best_seasonal_order,
                    'window_size': window_size,
                    'prediction_horizon': prediction_horizon,
                    'is_stationary': stationarity_check['is_stationary'],
                    'data_length': len(data_series),
                    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"  ✓ SARIMA模型训练完成!")
                print(f"    参数: SARIMA{best_order}x{best_seasonal_order}")
                print(f"    窗口大小: {window_size}, 预测步长: {prediction_horizon}")
                print(f"    信息准则: AIC={sarima_fit.aic:.2f}, BIC={sarima_fit.bic:.2f}")
            except Exception as e:
                print(f"  ❌ SARIMA模型训练失败: {str(e)}")
                self.models['sarima'] = None
        except Exception as e:
            print(f"  ❌ SARIMA模型训练失败: {str(e)}")
            self.models['sarima'] = None

    def _detect_seasonality(self, data, max_period=30):
        if len(data) < 2 * max_period:
            return 0
        from statsmodels.tsa.stattools import acf
        acf_values = acf(data, nlags=max_period, fft=True)
        threshold = 2 / np.sqrt(len(data))
        significant_lags = np.where(np.abs(acf_values) > threshold)[0]
        if len(significant_lags) > 1:
            for lag in significant_lags[1:]:
                if 5 <= lag <= max_period:
                    return lag
        return 0

    def _train_garch(self, returns):
        """训练GARCH模型 - 基于推荐参数优化"""
        try:
            garch_params = self._get_model_params('garch')
            min_data_required = garch_params.get('min_data_required', 200)
            recommended_p = garch_params.get('p', 1)
            recommended_q = garch_params.get('q', 2)
            window_size = garch_params.get('window_size', 50)
            prediction_horizon = garch_params.get('prediction_horizon', 5)

            if returns is None or len(returns) < min_data_required:
                print(f"  ⚠️ 收益率数据不足（最少需要{min_data_required}个点），跳过GARCH模型")
                self.models['garch'] = None
                return

            try:
                from arch import arch_model
            except ImportError:
                print("  ❌ 未安装arch包，跳过GARCH模型")
                self.models['garch'] = None
                return

            import warnings
            warnings.filterwarnings('ignore')

            print(f"  ℹ️ 使用推荐参数: p={recommended_p}, q={recommended_q}, window={window_size}")

            returns_series = pd.Series(returns).dropna()
            if len(returns_series) < 100:
                print(f"  ⚠️ 有效收益率数据不足（{len(returns_series)}个点），跳过GARCH模型")
                self.models['garch'] = None
                return

            def analyze_returns_characteristics(returns):
                from scipy import stats
                returns_clean = returns.dropna()
                if len(returns_clean) < 10:
                    return {}
                characteristics = {
                    'mean': float(np.mean(returns_clean)),
                    'std': float(np.std(returns_clean)),
                    'skewness': float(stats.skew(returns_clean)),
                    'kurtosis': float(stats.kurtosis(returns_clean))
                }
                return characteristics

            returns_chars = analyze_returns_characteristics(returns_series)
            print(f"  ℹ️ 收益率特征: 均值={returns_chars.get('mean', 0):.4f}, "
                  f"标准差={returns_chars.get('std', 0):.4f}, "
                  f"偏度={returns_chars.get('skewness', 0):.3f}, "
                  f"峰度={returns_chars.get('kurtosis', 0):.3f}")

            try:
                p = recommended_p
                q = recommended_q
                returns_scaled = returns_series * 100
                model = arch_model(returns_scaled, vol='GARCH', p=p, q=q, mean='Zero', dist='normal')
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    res = model.fit(disp='off', show_warning=False)

                if res.convergence_flag != 0:
                    print(f"  ⚠️ GARCH模型未收敛，尝试简化参数")
                    model = arch_model(returns_scaled, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
                    res = model.fit(disp='off')

                print(f"  ✓ GARCH({p},{q})模型训练完成")

                def diagnose_garch_model(model_fit):
                    standardized_residuals = model_fit.resid / model_fit.conditional_volatility
                    return {
                        'standardized_residuals': standardized_residuals,
                        'residual_mean': float(np.mean(standardized_residuals)),
                        'residual_std': float(np.std(standardized_residuals)),
                        'conditional_vol_mean': float(np.mean(model_fit.conditional_volatility)),
                        'conditional_vol_std': float(np.std(model_fit.conditional_volatility))
                    }

                diagnosis = diagnose_garch_model(res)

                self.models['garch'] = {
                    'model': res,
                    'model_summary': res.summary(),
                    'params': res.params,
                    'std_err': res.std_err,
                    'aic': res.aic,
                    'bic': res.bic,
                    'order': (p, q),
                    'window_size': window_size,
                    'prediction_horizon': prediction_horizon,
                    'diagnosis': diagnosis,
                    'returns_characteristics': returns_chars,
                    'data_length': len(returns_series),
                    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                print(f"    窗口大小: {window_size}, 预测步长: {prediction_horizon}")
                print(f"    信息准则: AIC={res.aic:.2f}, BIC={res.bic:.2f}")
            except Exception as e:
                print(f"  ❌ GARCH模型训练失败: {str(e)}")
                self.models['garch'] = None
        except Exception as e:
            print(f"  ❌ GARCH模型训练失败: {str(e)}")
            self.models['garch'] = None

    def _train_xgboost(self, close_prices, returns, volume):
        """训练XGBoost模型 - 基于推荐参数优化"""
        try:
            xgboost_params = self._get_model_params('xgboost')
            min_data_required = xgboost_params.get('min_data_required', 100)
            n_estimators = xgboost_params.get('n_estimators', 50)
            learning_rate = xgboost_params.get('learning_rate', 0.01)
            max_depth = xgboost_params.get('max_depth', 5)
            window_size = xgboost_params.get('window_size', 50)
            prediction_horizon = xgboost_params.get('prediction_horizon', 5)

            if close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 价格数据不足（最少需要{min_data_required}个点），跳过XGBoost模型")
                self.models['xgboost'] = None
                return

            print(f"  ℹ️ 使用推荐参数: n_estimators={n_estimators}, learning_rate={learning_rate}, "
                  f"max_depth={max_depth}, window={window_size}")

            xgb_params = self._get_xgboost_params()

            prices_series = pd.Series(close_prices)
            seq_length = min(10, window_size // 5)
            X, y = [], []
            for i in range(seq_length, len(prices_series)):
                X.append(prices_series[i - seq_length:i].values)
                y.append(prices_series[i])
            X, y = np.array(X), np.array(y)

            if len(X) < 50:
                print(f"  ⚠️ 序列数据不足，跳过XGBoost模型")
                self.models['xgboost'] = None
                return

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            feature_importance = model.feature_importances_
            important_indices = np.where(feature_importance > np.mean(feature_importance))[0]
            important_features = feature_importance[important_indices]

            self.models['xgboost'] = {
                'model': model,
                'feature_names': [f'lag_{i + 1}' for i in range(seq_length)],
                'important_features': important_features,
                'important_indices': important_indices,
                'feature_importance': feature_importance.tolist(),
                'train_score': train_score,
                'val_score': val_score,
                'params': xgb_params,
                'sequence_length': seq_length,
                'window_size': window_size,
                'prediction_horizon': prediction_horizon,
                'last_sequence': prices_series[-seq_length:].tolist()
            }
            print(f"  ✓ XGBoost模型训练完成")
            print(f"    训练集R²={train_score:.4f}，验证集R²={val_score:.4f}")
            print(f"    窗口大小: {window_size}, 预测步长: {prediction_horizon}")
            print(f"    序列长度: {seq_length}, 特征数: {len(feature_importance)}")
        except Exception as e:
            print(f"  ❌ XGBoost模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.models['xgboost'] = None

    def _get_xgboost_params(self):
        xgboost_params = self._get_model_params("xgboost")
        return {
            'n_estimators': xgboost_params.get('n_estimators', 50),
            'max_depth': xgboost_params.get('max_depth', 5),
            'learning_rate': xgboost_params.get('learning_rate', 0.01),
            'subsample': xgboost_params.get('subsample', 0.8),
            'colsample_bytree': xgboost_params.get('colsample_bytree', 0.8),
            'min_child_weight': xgboost_params.get('min_child_weight', 1),
            'reg_alpha': xgboost_params.get('reg_alpha', 0),
            'reg_lambda': xgboost_params.get('reg_lambda', 1),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }

    def _train_multi_factor(self, close_prices, volume, high, low, open_prices):
        """训练多因子模型"""
        try:
            multi_factor_params = self._get_model_params('multi_factor')
            min_data_required = multi_factor_params.get('min_data_required', 100)
            window_size = multi_factor_params.get('window_size', 150)
            prediction_horizon = multi_factor_params.get('prediction_horizon', 10)
            n_features = multi_factor_params.get('n_features', 10)

            if close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 数据不足（最少需要{min_data_required}个点），跳过多因子模型")
                self.models['multi_factor'] = None
                return

            print(f"  ℹ️ 使用推荐参数: window={window_size}, horizon={prediction_horizon}, features={n_features}")

            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            min_length = len(close_prices)
            volume_data = volume if volume is not None else np.ones(min_length)
            high_data = high if high is not None else close_prices
            low_data = low if low is not None else close_prices
            open_data = open_prices if open_prices is not None else close_prices

            min_length = min(min_length, len(volume_data), len(high_data), len(low_data), len(open_data))
            data = pd.DataFrame({
                'close': close_prices[:min_length],
                'volume': volume_data[:min_length],
                'high': high_data[:min_length],
                'low': low_data[:min_length],
                'open': open_data[:min_length]
            })
            data.index = range(min_length)
            try:
                data = data.fillna(method='ffill').fillna(method='bfill')
            except TypeError:
                data = data.ffill().bfill()
            data = data.fillna(0)

            def create_features(df, n_features=15):
                features = pd.DataFrame(index=df.index)
                close = df['close']
                windows = [5, 10, 20, 30, 60]
                for w in windows:
                    features[f'ma_ratio_{w}'] = close / close.rolling(w, min_periods=1).mean()
                for w in [5, 10, 20]:
                    features[f'momentum_{w}'] = (close / close.shift(w) - 1) * 100
                returns = df['close'].pct_change()
                for w in [10, 20, 30]:
                    rolling_std = returns.rolling(w, min_periods=1).std()
                    features[f'volatility_{w}'] = rolling_std * np.sqrt(252)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                rs = gain / loss
                features['rsi'] = 100 - (100 / (1 + rs))
                window = 20
                high_roll = df['high'].rolling(window, min_periods=1).max()
                low_roll = df['low'].rolling(window, min_periods=1).min()
                features['channel_position'] = (close - low_roll) / (high_roll - low_roll + 1e-10)
                volume_mean = df['volume'].rolling(20, min_periods=1).mean()
                features['volume_ratio'] = df['volume'] / volume_mean
                features['price_volume_corr'] = 0.0
                if len(df) >= 30:
                    from scipy import stats
                    trend_strength_values = []
                    for i in range(30, len(df)):
                        x = np.arange(30)
                        y = close.iloc[i - 30:i].values
                        try:
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            trend_strength_values.append(abs(r_value))
                        except:
                            trend_strength_values.append(0.0)
                    if len(trend_strength_values) > 0:
                        full_trend = [0.0] * 30 + trend_strength_values
                        features['trend_strength'] = full_trend[:len(df)]
                    else:
                        features['trend_strength'] = 0.0
                mean_50 = close.rolling(50, min_periods=1).mean()
                features['mean_reversion'] = (close / mean_50 - 1) * 100
                try:
                    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
                except TypeError:
                    features = features.ffill().bfill().fillna(0)
                if len(features.columns) > n_features:
                    variances = features.var()
                    top_features = variances.nlargest(n_features).index
                    features = features[top_features]
                elif len(features.columns) < n_features:
                    missing = n_features - len(features.columns)
                    for i in range(missing):
                        features[f'feature_{i}'] = 0.0
                return features

            X = create_features(data, n_features=n_features)
            future_returns = (data['close'].shift(-prediction_horizon) / data['close'] - 1) * 100
            aligned_idx = X.index.intersection(future_returns.dropna().index)
            X = X.loc[aligned_idx]
            y = future_returns.loc[aligned_idx]

            if len(X) < 50:
                print(f"  ⚠️ 有效数据不足（{len(X)}个样本），跳过多因子模型")
                self.models['multi_factor'] = None
                return

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            feature_importance = model.feature_importances_
            important_features = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            self.models['multi_factor'] = {
                'model': model,
                'scaler': scaler,
                'feature_names': X.columns.tolist(),
                'feature_importance': feature_importance.tolist(),
                'important_features': important_features,
                'train_score': train_score,
                'test_score': test_score,
                'window_size': window_size,
                'prediction_horizon': prediction_horizon,
                'n_features_expected': X.shape[1],
                'last_features': X_scaled[-1:],
                'last_close': data['close'].iloc[-1],
                'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            print(f"  ✓ 多因子模型训练完成")
            print(f"    训练集R²={train_score:.4f}，测试集R²={test_score:.4f}")
            print(f"    窗口大小: {window_size}, 预测步长: {prediction_horizon}")
            print(f"    特征数量: {X.shape[1]}")
            print(f"    最重要特征: {important_features.iloc[0]['feature']} "
                  f"(重要性={important_features.iloc[0]['importance']:.3f})")
        except Exception as e:
            print(f"  ❌ 多因子模型训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            self.models['multi_factor'] = None

    # ==================== 新增模型训练方法 ====================
    def _train_arima(self, close_prices):
        """训练ARIMA模型"""
        if not ARIMA_AVAILABLE:
            print("  ⚠️ ARIMA库不可用，跳过")
            self.models['arima'] = None
            return

        try:
            arima_params = self._get_model_params('arima')
            min_data_required = arima_params.get('min_data_required', 100)
            window_size = arima_params.get('window_size', 100)
            prediction_horizon = arima_params.get('prediction_horizon', 5)

            if close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 数据不足，跳过ARIMA")
                self.models['arima'] = None
                return

            series = pd.Series(close_prices).dropna()
            model = ARIMA(series, order=(2, 1, 2))
            arima_fit = model.fit()

            self.models['arima'] = {
                'model': arima_fit,
                'aic': arima_fit.aic,
                'bic': arima_fit.bic,
                'window_size': window_size,
                'prediction_horizon': prediction_horizon,
                'data_length': len(series)
            }
            print(f"  ✓ ARIMA训练完成 AIC={arima_fit.aic:.2f}")
        except Exception as e:
            print(f"  ❌ ARIMA训练失败: {e}")
            self.models['arima'] = None

    def _train_lstm(self, close_prices):
        """训练LSTM模型"""
        if not TF_AVAILABLE:
            print("  ⚠️ TensorFlow不可用，跳过LSTM")
            self.models['lstm'] = None
            return

        try:
            lstm_params = self._get_model_params('lstm')
            min_data_required = lstm_params.get('min_data_required', 200)
            window_size = lstm_params.get('window_size', 60)
            epochs = lstm_params.get('epochs', 50)
            batch_size = lstm_params.get('batch_size', 32)
            units = lstm_params.get('units', 64)

            if close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 数据不足，跳过LSTM")
                self.models['lstm'] = None
                return

            series = close_prices.reshape(-1, 1)
            X, y = [], []
            for i in range(window_size, len(series)):
                X.append(series[i - window_size:i, 0])
                y.append(series[i, 0])
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(window_size, 1)),
                Dropout(0.2),
                LSTM(units // 2),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

            self.models['lstm'] = {
                'model': model,
                'window_size': window_size,
                'prediction_horizon': lstm_params.get('prediction_horizon', 5),
                'last_sequence': series[-window_size:].flatten()
            }
            print(f"  ✓ LSTM训练完成")
        except Exception as e:
            print(f"  ❌ LSTM训练失败: {e}")
            self.models['lstm'] = None

    def _train_lightgbm(self, close_prices, returns, volume):
        """训练LightGBM模型"""
        if not LGB_AVAILABLE:
            print("  ⚠️ LightGBM库不可用，跳过")
            self.models['lightgbm'] = None
            return

        try:
            params = self._get_model_params('lightgbm')
            min_data_required = params.get('min_data_required', 100)
            window_size = params.get('window_size', 50)
            n_estimators = params.get('n_estimators', 100)
            lr = params.get('learning_rate', 0.05)
            max_depth = params.get('max_depth', 7)

            if close_prices is None or len(close_prices) < min_data_required:
                print("  ⚠️ 数据不足，跳过LightGBM")
                self.models['lightgbm'] = None
                return

            seq_length = min(10, window_size // 5)
            X, y = [], []
            prices_series = pd.Series(close_prices)
            for i in range(seq_length, len(prices_series)):
                X.append(prices_series[i - seq_length:i].values)
                y.append(prices_series[i])
            X, y = np.array(X), np.array(y)

            model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=max_depth, verbose=-1)
            model.fit(X, y)
            self.models['lightgbm'] = {
                'model': model,
                'window_size': window_size,
                'prediction_horizon': params.get('prediction_horizon', 5),
                'last_sequence': prices_series[-seq_length:].tolist()
            }
            print("  ✓ LightGBM训练完成")
        except Exception as e:
            print(f"  ❌ LightGBM训练失败: {e}")
            self.models['lightgbm'] = None

    def _train_catboost(self, close_prices, returns, volume):
        """训练CatBoost模型"""
        if not CB_AVAILABLE:
            print("  ⚠️ CatBoost库不可用，跳过")
            self.models['catboost'] = None
            return

        try:
            params = self._get_model_params('catboost')
            min_data_required = params.get('min_data_required', 100)
            window_size = params.get('window_size', 50)
            iterations = params.get('iterations', 100)
            lr = params.get('learning_rate', 0.05)
            depth = params.get('depth', 6)

            if close_prices is None or len(close_prices) < min_data_required:
                print("  ⚠️ 数据不足，跳过CatBoost")
                self.models['catboost'] = None
                return

            seq_length = min(10, window_size // 5)
            X, y = [], []
            prices_series = pd.Series(close_prices)
            for i in range(seq_length, len(prices_series)):
                X.append(prices_series[i - seq_length:i].values)
                y.append(prices_series[i])
            X, y = np.array(X), np.array(y)

            model = cb.CatBoostRegressor(iterations=iterations, learning_rate=lr, depth=depth, verbose=0)
            model.fit(X, y)
            self.models['catboost'] = {
                'model': model,
                'window_size': window_size,
                'prediction_horizon': params.get('prediction_horizon', 5),
                'last_sequence': prices_series[-seq_length:].tolist()
            }
            print("  ✓ CatBoost训练完成")
        except Exception as e:
            print(f"  ❌ CatBoost训练失败: {e}")
            self.models['catboost'] = None

    def evaluate_models(self):
        """评估模型（包含新增模型）"""
        try:
            scores = {}
            for model_name, model_info in self.models.items():
                if model_info is None:
                    scores[model_name] = 0.0
                    continue

                base_score = 0.5
                if model_name in ['arima', 'sarima', 'garch']:
                    if 'aic' in model_info:
                        aic = model_info['aic']
                        base_score = 1.0 / (1.0 + abs(aic) / 1000)
                elif model_name in ['xgboost', 'lightgbm', 'catboost']:
                    if 'val_score' in model_info:
                        base_score = max(0, model_info['val_score'])
                    elif 'train_score' in model_info:
                        base_score = max(0, model_info['train_score'])
                elif model_name == 'multi_factor':
                    base_score = max(0, model_info.get('test_score', 0.5))
                elif model_name == 'lstm':
                    base_score = 0.6
                elif model_name == 'ensemble':
                    base_score = 0.7

                market_adjustment = 1.0
                if self.market_regime == 'high_volatility':
                    if model_name in ['garch', 'multi_factor', 'lightgbm']:
                        market_adjustment = 1.2
                elif self.market_regime == 'trending':
                    if model_name in ['sarima', 'arima', 'lightgbm', 'catboost']:
                        market_adjustment = 1.2

                final_score = min(1.0, base_score * market_adjustment)
                scores[model_name] = final_score

            total = sum(scores.values())
            if total > 0:
                for k in scores:
                    scores[k] /= total
            print(f"模型评估分数: {scores}")
            return scores
        except Exception as e:
            print(f"模型评估出错: {e}")
            return {m: 0.2 for m in self.models.keys()}