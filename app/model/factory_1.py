"""
优化要点：
- 技术指标循环 stats.linregress / stats.skew / stats.kurtosis 全部向量化
  (基于 numpy.lib.stride_tricks.sliding_window_view)
- 训练后 del 大中间对象 + gc.collect()
- model_summary / diagnosis 只在 FACTORY_DEBUG=1 下保存
- last_features/last_factors 使用 .copy() 避免 dangling view
"""
import os
import gc
import pandas as pd
import numpy as np
import warnings
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
from scipy import stats as sp_stats

DEBUG = os.environ.get("FACTORY_DEBUG", "0") == "1"


def _rolling_trend_r2(prices_arr, window=20):
    """向量化：每 window 个点的线性回归 |R|"""
    n = len(prices_arr)
    result = np.zeros(n, dtype=float)
    if n < window:
        return result
    y_win = sliding_window_view(np.asarray(prices_arr, dtype=float), window)
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    yc = y_win - y_win.mean(axis=1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum() * (yc ** 2).sum(axis=1)) + 1e-10
    r = np.abs(num / den)
    result[window - 1:] = r
    return result


class ModelFactory:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.market_regime = None

        from config_loader import get_config_loader
        self.param_loader = get_config_loader()

        if not self.param_loader.loaded:
            default_config = (
                "backtest_results/parameter_recommendations/parameter_recommendations.txt"
            )
            self.param_loader.load_config_file(default_config)

        self._init_dynamic_config()
        self._init_parameter_recommendations()

    def _init_parameter_recommendations(self):
        self.parameter_recommendations = {
            "garch": {"p": 1, "q": 2, "window_size": 50, "prediction_horizon": 5, "min_data_required": 200},
            "xgboost": {"n_estimators": 50, "learning_rate": 0.01, "max_depth": 5,
                        "window_size": 50, "prediction_horizon": 5, "min_data_required": 100},
            "sarima": {"window_size": 100, "prediction_horizon": 5, "min_data_required": 150},
            "multi_factor": {"window_size": 150, "prediction_horizon": 10, "min_data_required": 100, "n_features": 10},
            "arima": {"window_size": 100, "prediction_horizon": 5, "min_data_required": 100},
            "lstm": {"window_size": 60, "prediction_horizon": 5, "epochs": 50,
                     "batch_size": 32, "units": 64, "min_data_required": 200},
            "lightgbm": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 7,
                         "window_size": 50, "prediction_horizon": 5, "min_data_required": 100},
            "catboost": {"iterations": 100, "learning_rate": 0.05, "depth": 6,
                         "window_size": 50, "prediction_horizon": 5, "min_data_required": 100},
            "quant_factor": {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.05,
                             "n_factors": 16, "window_size": 60, "prediction_horizon": 5,
                             "min_data_required": 120, "lookback_windows": [5, 10, 20, 60]},
        }

    def _init_dynamic_config(self):
        self.model_selection_config = self._get_model_selection_config()
        self.model_weight_config = self._get_model_weight_config()

    def _get_param_value(self, section, key, default):
        try:
            if section in self.parameter_recommendations and key in self.parameter_recommendations[section]:
                return self.parameter_recommendations[section][key]
            if section in ["sarima", "garch", "xgboost", "multi_factor", "arima",
                           "lstm", "lightgbm", "catboost", "quant_factor"]:
                recommendation = self.param_loader.get_model_recommendation(section)
                if recommendation and recommendation.parameters and key in recommendation.parameters:
                    return recommendation.parameters[key]
            general_param = self.param_loader.get_general_parameter(key)
            if general_param:
                if hasattr(general_param, "recommended_value"):
                    return general_param.recommended_value
                if hasattr(general_param, "value"):
                    return general_param.value
                if isinstance(general_param, (int, float, str, bool)):
                    return general_param
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
        except Exception:
            pass
        return {}

    def train_all_models(self, processed_data):
        if processed_data is None:
            print("❌ 输入数据为None")
            return self.models
        if "prepared" not in processed_data:
            print("❌ processed_data中没有'prepared'键")
            return self.models

        prepared_data = processed_data["prepared"]
        if "close_scaled" not in prepared_data:
            print("❌ prepared_data中没有'close_scaled'数据")
            return self.models

        try:
            close_scaled = prepared_data["close_scaled"]
            returns = prepared_data.get("returns")

            df = None
            close_prices = volume = high = low = open_prices = None
            if "processed" in processed_data:
                df = processed_data["processed"]
                if df is not None and len(df) > 0:
                    close_prices = df["close"].values
                    volume = df["volume"].values if "volume" in df.columns else None
                    high = df["high"].values if "high" in df.columns else None
                    low = df["low"].values if "low" in df.columns else None
                    open_prices = df["open"].values if "open" in df.columns else None

            if close_scaled is None or len(close_scaled) == 0:
                print("❌ close_scaled数据为空")
                return self.models

            self._analyze_market_regime(df, returns)
            self._select_and_train_models(
                df, close_scaled, returns, close_prices, volume, high, low, open_prices
            )

            successful_models = [name for name, model in self.models.items() if model is not None]
            if successful_models:
                print(f"  可用模型: {', '.join(successful_models)}")

            # 清理训练期临时大对象
            gc.collect()
            return self.models
        except Exception as e:
            print(f"❌ 模型训练过程出错: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            return self.models

    def _analyze_market_regime(self, df, returns):
        try:
            if df is None or returns is None:
                self.market_regime = "default"
                return

            volatility_threshold = self._get_param_value("market_analysis", "volatility_threshold", 0.03)
            trend_strength_threshold = self._get_param_value("market_analysis", "trend_strength_threshold", 0.7)
            skewness_threshold = self._get_param_value("market_analysis", "skewness_threshold", 0.5)
            kurtosis_threshold = self._get_param_value("market_analysis", "kurtosis_threshold", 3)
            adf_significance = self._get_param_value("market_analysis", "adf_significance", 0.05)

            try:
                from scipy import stats
            except ImportError:
                print("⚠️ scipy.stats不可用，使用简化统计计算")
                self.market_regime = "default"
                return

            volatility = np.std(returns) if len(returns) > 0 else 0
            trend_strength = self._calculate_trend_strength(df["close"].values)
            skewness = stats.skew(returns) if len(returns) > 0 else 0
            kurtosis = stats.kurtosis(returns) if len(returns) > 0 else 0

            adf_result = adfuller(df["close"].dropna())
            is_stationary = adf_result[1] < adf_significance

            if volatility > volatility_threshold:
                if abs(skewness) > skewness_threshold or kurtosis > kurtosis_threshold:
                    self.market_regime = "high_volatility_nonlinear"
                else:
                    self.market_regime = "high_volatility"
            elif trend_strength > trend_strength_threshold:
                self.market_regime = "stable_trend" if is_stationary else "trending"
            else:
                self.market_regime = "default"
        except Exception as e:
            print(f"市场状态分析失败: {e}")
            self.market_regime = "default"

    def _calculate_trend_strength(self, prices):
        min_data_length = self._get_param_value("trend_strength", "min_data_length", 20)
        if len(prices) < min_data_length:
            return 0
        try:
            from scipy import stats
        except ImportError:
            return 0
        x = np.arange(len(prices))
        _, _, r_value, _, _ = stats.linregress(x, prices)
        return abs(r_value)

    def _select_and_train_models(self, df, close_scaled, returns, close_prices,
                                  volume, high, low, open_prices):
        model_train_methods = {
            "sarima": lambda: self._train_sarima(close_scaled),
            "garch": lambda: self._train_garch(returns),
            "xgboost": lambda: self._train_xgboost(close_prices, returns, volume),
            "multi_factor": lambda: self._train_multi_factor(close_prices, volume, high, low, open_prices),
            "arima": lambda: self._train_arima(close_prices),
            "lstm": lambda: self._train_lstm(close_prices),
            "lightgbm": lambda: self._train_lightgbm(close_prices, returns, volume),
            "catboost": lambda: self._train_catboost(close_prices, returns, volume),
            "quant_factor": lambda: self._train_quant_factor(df, close_prices, volume, high, low, open_prices),
        }

        models_to_train = self.model_selection_config.get(
            self.market_regime,
            self.model_selection_config.get(
                "default", ["xgboost", "multi_factor", "garch", "arima", "lightgbm"]
            ),
        )
        print(f"根据市场状态 '{self.market_regime}' 选择模型: {', '.join(models_to_train)}")

        for i, model_name in enumerate(models_to_train):
            print(f"[{i + 1}/{len(models_to_train)}] 训练{model_name}模型...")
            try:
                if model_name in model_train_methods:
                    model_train_methods[model_name]()
                elif model_name == "ensemble":
                    self.models["ensemble"] = {"status": "ready", "description": "动态集成模型"}
                    print("  ✓ 集成模型已就绪")
                else:
                    print(f"  ⚠️ 模型 {model_name} 不在可训练模型列表中")
            except Exception as e:
                print(f"  ⚠️ {model_name}训练失败: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
            finally:
                gc.collect()

    def _get_model_selection_config(self):
        try:
            from config_loader import get_config_loader
            loader = get_config_loader()
            if hasattr(loader, "model_selection") and loader.model_selection:
                return loader.model_selection
        except Exception:
            pass
        return {
            "high_volatility_nonlinear": ["garch", "xgboost", "lightgbm", "catboost", "quant_factor"],
            "high_volatility": ["garch", "multi_factor", "arima", "quant_factor"],
            "trending": ["sarima", "multi_factor", "arima", "lightgbm", "quant_factor"],
            "stable_trend": ["sarima", "garch", "arima", "catboost", "quant_factor"],
            "default": ["sarima", "garch", "multi_factor", "xgboost", "arima",
                        "lightgbm", "catboost", "quant_factor"],
        }

    def _get_model_weight_config(self):
        try:
            from config_loader import get_config_loader
            loader = get_config_loader()
            if hasattr(loader, "model_weights") and loader.model_weights:
                return loader.model_weights
        except Exception:
            pass
        return {
            "high_volatility_nonlinear": [("xgboost", 0.25), ("quant_factor", 0.22), ("lightgbm", 0.20), ("catboost", 0.18), ("garch", 0.15)],
            "high_volatility": [("garch", 0.30), ("multi_factor", 0.25), ("quant_factor", 0.25), ("arima", 0.20)],
            "trending": [("sarima", 0.28), ("multi_factor", 0.24), ("quant_factor", 0.22), ("arima", 0.15), ("lightgbm", 0.11)],
            "stable_trend": [("sarima", 0.30), ("garch", 0.22), ("quant_factor", 0.22), ("arima", 0.16), ("catboost", 0.10)],
            "default": [("sarima", 0.17), ("garch", 0.17), ("xgboost", 0.13), ("multi_factor", 0.13),
                        ("quant_factor", 0.13), ("arima", 0.09), ("lightgbm", 0.09), ("catboost", 0.09)],
        }

    # ==================== 模型训练方法 ====================

    def _train_sarima(self, close_scaled):
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
        except ImportError:
            print("  ⚠️ SARIMAX不可用，跳过SARIMA模型")
            self.models["sarima"] = None
            return

        try:
            sarima_params = self._get_model_params("sarima")
            min_data_required = sarima_params.get("min_data_required", 150)
            window_size = sarima_params.get("window_size", 100)
            prediction_horizon = sarima_params.get("prediction_horizon", 5)

            if close_scaled is None or len(close_scaled) < min_data_required:
                print(f"  ⚠️ 数据不足（最少需要{min_data_required}个点），跳过SARIMA模型")
                self.models["sarima"] = None
                return

            data = close_scaled.flatten() if close_scaled.ndim > 1 else close_scaled
            data_series = pd.Series(data)
            if data_series.isna().any():
                data_series = data_series.ffill().bfill()

            adf_result = adfuller(data_series.dropna())
            is_stationary = adf_result[1] < 0.05

            from statsmodels.tsa.stattools import acf
            seasonal_period = 0
            if len(data_series) >= 60:
                acf_vals = acf(data_series.values, nlags=30, fft=True)
                threshold = 2 / np.sqrt(len(data_series))
                significant = np.where(np.abs(acf_vals) > threshold)[0]
                if len(significant) > 1:
                    for lag in significant[1:]:
                        if 5 <= lag <= 30:
                            seasonal_period = lag
                            break

            if seasonal_period <= 1:
                print("  ⚠️ 未检测到明显季节性，使用ARIMA而非SARIMA")
                self.models["sarima"] = None
                return

            best_order = (1, 1, 1)
            best_seasonal_order = (1, 1, 1, seasonal_period)

            if len(data_series) > 200:
                import itertools
                from statsmodels.tools.sm_exceptions import ConvergenceWarning
                warnings.simplefilter("ignore", ConvergenceWarning)

                p_values = range(0, 2)
                q_values = range(0, 2)
                P_values = range(0, 2)
                Q_values = range(0, 2)

                best_aic = np.inf
                for p, q, P, Q in itertools.product(p_values, q_values, P_values, Q_values):
                    try:
                        model = SARIMAX(
                            data_series, order=(p, 1, q),
                            seasonal_order=(P, 1, Q, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                            simple_differencing=True,
                        )
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore")
                            results = model.fit(disp=False, maxiter=100)
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, 1, q)
                            best_seasonal_order = (P, 1, Q, seasonal_period)
                    except Exception:
                        continue
                print(f"  ✓ 最优参数: SARIMA{best_order}x{best_seasonal_order}, AIC={best_aic:.2f}")

            print(f"  ⏳ 训练SARIMA{best_order}x{best_seasonal_order}模型...")
            sarima_model = SARIMAX(
                data_series, order=best_order,
                seasonal_order=best_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True,
            )
            sarima_fit = sarima_model.fit(disp=False, maxiter=200, method="lbfgs", cov_type="robust")

            info = {
                "model": sarima_fit,
                "params": sarima_fit.params,
                "aic": sarima_fit.aic,
                "bic": sarima_fit.bic,
                "hqic": sarima_fit.hqic,
                "seasonal_period": seasonal_period,
                "order": best_order,
                "seasonal_order": best_seasonal_order,
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "is_stationary": is_stationary,
                "data_length": len(data_series),
                "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if DEBUG:
                info["model_summary"] = sarima_fit.summary()
            self.models["sarima"] = info
            print(f"  ✓ SARIMA模型训练完成! AIC={sarima_fit.aic:.2f}, BIC={sarima_fit.bic:.2f}")
        except Exception as e:
            print(f"  ❌ SARIMA模型训练失败: {str(e)}")
            self.models["sarima"] = None
        finally:
            gc.collect()

    def _train_garch(self, returns):
        try:
            garch_params = self._get_model_params("garch")
            min_data_required = garch_params.get("min_data_required", 200)
            recommended_p = garch_params.get("p", 1)
            recommended_q = garch_params.get("q", 2)
            window_size = garch_params.get("window_size", 50)
            prediction_horizon = garch_params.get("prediction_horizon", 5)

            if returns is None or len(returns) < min_data_required:
                print(f"  ⚠️ 收益率数据不足（最少需要{min_data_required}个点），跳过GARCH")
                self.models["garch"] = None
                return

            try:
                from arch import arch_model
            except ImportError:
                print("  ❌ 未安装arch包，跳过GARCH模型")
                self.models["garch"] = None
                return

            returns_series = pd.Series(returns).dropna()
            if len(returns_series) < 100:
                print(f"  ⚠️ 有效收益率数据不足（{len(returns_series)}），跳过GARCH")
                self.models["garch"] = None
                return

            returns_scaled = returns_series * 100
            model = arch_model(returns_scaled, vol="GARCH", p=recommended_p, q=recommended_q,
                               mean="Zero", dist="normal")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                res = model.fit(disp="off", show_warning=False)

            if res.convergence_flag != 0:
                print("  ⚠️ GARCH未收敛，尝试简化参数(1,1)")
                model = arch_model(returns_scaled, vol="GARCH", p=1, q=1,
                                   mean="Zero", dist="normal")
                res = model.fit(disp="off")

            info = {
                "model": res,
                "params": res.params,
                "std_err": res.std_err,
                "aic": res.aic,
                "bic": res.bic,
                "order": (recommended_p, recommended_q),
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "data_length": len(returns_series),
                "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if DEBUG:
                info["model_summary"] = res.summary()
                std_resid = res.resid / res.conditional_volatility
                info["diagnosis"] = {
                    "residual_mean": float(np.mean(std_resid)),
                    "residual_std": float(np.std(std_resid)),
                }
            self.models["garch"] = info
            print(f"  ✓ GARCH({recommended_p},{recommended_q})训练完成 AIC={res.aic:.2f}")
        except Exception as e:
            print(f"  ❌ GARCH训练失败: {str(e)}")
            self.models["garch"] = None
        finally:
            gc.collect()

    def _get_xgboost_params(self):
        p = self._get_model_params("xgboost")
        return {
            "n_estimators": p.get("n_estimators", 50),
            "max_depth": p.get("max_depth", 5),
            "learning_rate": p.get("learning_rate", 0.01),
            "subsample": p.get("subsample", 0.8),
            "colsample_bytree": p.get("colsample_bytree", 0.8),
            "min_child_weight": p.get("min_child_weight", 1),
            "reg_alpha": p.get("reg_alpha", 0),
            "reg_lambda": p.get("reg_lambda", 1),
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

    def _train_xgboost(self, close_prices, returns, volume):
        try:
            import xgboost as xgb
        except ImportError:
            print("  ⚠️ XGBoost库不可用，跳过")
            self.models["xgboost"] = None
            return

        try:
            params = self._get_model_params("xgboost")
            min_data_required = params.get("min_data_required", 100)
            window_size = params.get("window_size", 50)
            prediction_horizon = params.get("prediction_horizon", 5)

            if close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 价格数据不足（最少需要{min_data_required}个点），跳过XGBoost")
                self.models["xgboost"] = None
                return

            xgb_params = self._get_xgboost_params()
            prices_series = pd.Series(close_prices)
            seq_length = min(10, window_size // 5)
            X, y = [], []
            for i in range(seq_length, len(prices_series)):
                X.append(prices_series[i - seq_length:i].values)
                y.append(prices_series[i])
            X, y = np.array(X), np.array(y)

            if len(X) < 50:
                print("  ⚠️ 序列数据不足，跳过XGBoost")
                self.models["xgboost"] = None
                return

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            feature_importance = model.feature_importances_

            self.models["xgboost"] = {
                "model": model,
                "feature_names": [f"lag_{i + 1}" for i in range(seq_length)],
                "feature_importance": feature_importance.tolist(),
                "train_score": train_score,
                "val_score": val_score,
                "params": xgb_params,
                "sequence_length": seq_length,
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "last_sequence": prices_series[-seq_length:].tolist(),
            }
            print(f"  ✓ XGBoost训练完成 R² train={train_score:.4f}, val={val_score:.4f}")

            del X, y, X_train, X_val, y_train, y_val
        except Exception as e:
            print(f"  ❌ XGBoost训练失败: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            self.models["xgboost"] = None
        finally:
            gc.collect()

    def _train_multi_factor(self, close_prices, volume, high, low, open_prices):
        try:
            params = self._get_model_params("multi_factor")
            min_data_required = params.get("min_data_required", 100)
            window_size = params.get("window_size", 150)
            prediction_horizon = params.get("prediction_horizon", 10)
            n_features = params.get("n_features", 10)

            if close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 数据不足（最少需要{min_data_required}个点），跳过多因子模型")
                self.models["multi_factor"] = None
                return

            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split

            min_length = min(
                len(close_prices),
                len(volume) if volume is not None else len(close_prices),
                len(high) if high is not None else len(close_prices),
                len(low) if low is not None else len(close_prices),
                len(open_prices) if open_prices is not None else len(close_prices),
            )

            data = pd.DataFrame({
                "close": close_prices[:min_length],
                "volume": volume[:min_length] if volume is not None else np.ones(min_length),
                "high": high[:min_length] if high is not None else close_prices[:min_length],
                "low": low[:min_length] if low is not None else close_prices[:min_length],
                "open": open_prices[:min_length] if open_prices is not None else close_prices[:min_length],
            })
            data = data.ffill().bfill().fillna(0)

            close = data["close"]
            returns = close.pct_change()

            features = pd.DataFrame(index=data.index)
            for w in [5, 10, 20, 30, 60]:
                features[f"ma_ratio_{w}"] = close / close.rolling(w, min_periods=1).mean()
            for w in [5, 10, 20]:
                features[f"momentum_{w}"] = (close / close.shift(w) - 1) * 100
            for w in [10, 20, 30]:
                features[f"volatility_{w}"] = returns.rolling(w, min_periods=1).std() * np.sqrt(252)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
            features["rsi"] = 100 - (100 / (1 + gain / loss))
            high_roll = data["high"].rolling(20, min_periods=1).max()
            low_roll = data["low"].rolling(20, min_periods=1).min()
            features["channel_position"] = (close - low_roll) / (high_roll - low_roll + 1e-10)
            volume_mean = data["volume"].rolling(20, min_periods=1).mean()
            features["volume_ratio"] = data["volume"] / volume_mean
            features["price_volume_corr"] = 0.0
            features["trend_strength"] = _rolling_trend_r2(close.values, window=20)
            features["mean_reversion"] = (close / close.rolling(50, min_periods=1).mean() - 1) * 100

            features = features.ffill().bfill().fillna(0)
            if len(features.columns) > n_features:
                top_cols = features.var().nlargest(n_features).index
                features = features[top_cols]
            elif len(features.columns) < n_features:
                for i in range(n_features - len(features.columns)):
                    features[f"feature_{i}"] = 0.0

            future_returns = (data["close"].shift(-prediction_horizon) / data["close"] - 1) * 100
            aligned_idx = features.index.intersection(future_returns.dropna().index)
            X = features.loc[aligned_idx]
            y = future_returns.loc[aligned_idx]

            if len(X) < 50:
                print(f"  ⚠️ 有效数据不足（{len(X)}），跳过多因子模型")
                self.models["multi_factor"] = None
                return

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y.values, test_size=0.2, random_state=42, shuffle=False
            )

            model = RandomForestRegressor(
                n_estimators=100, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            )
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            feature_importance = model.feature_importances_

            last_features = X_scaled[-1:].copy()
            imp_df = pd.DataFrame({
                "feature": X.columns,
                "importance": feature_importance,
            }).sort_values("importance", ascending=False)

            self.models["multi_factor"] = {
                "model": model,
                "scaler": scaler,
                "feature_names": X.columns.tolist(),
                "feature_importance": feature_importance.tolist(),
                "important_features": imp_df,
                "train_score": train_score,
                "test_score": test_score,
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "n_features_expected": X.shape[1],
                "last_features": last_features,
                "last_close": float(data["close"].iloc[-1]),
                "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  ✓ 多因子模型训练完成 R² train={train_score:.4f}, test={test_score:.4f}")

            del X, y, X_scaled, X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"  ❌ 多因子模型训练失败: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            self.models["multi_factor"] = None
        finally:
            gc.collect()

    def _train_arima(self, close_prices):
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            print("  ⚠️ ARIMA库不可用，跳过")
            self.models["arima"] = None
            return
        try:
            params = self._get_model_params("arima")
            min_data_required = params.get("min_data_required", 100)
            window_size = params.get("window_size", 100)
            prediction_horizon = params.get("prediction_horizon", 5)

            if close_prices is None or len(close_prices) < min_data_required:
                print("  ⚠️ 数据不足，跳过ARIMA")
                self.models["arima"] = None
                return

            series = pd.Series(close_prices).dropna()
            model = ARIMA(series, order=(2, 1, 2))
            arima_fit = model.fit()

            self.models["arima"] = {
                "model": arima_fit,
                "aic": arima_fit.aic,
                "bic": arima_fit.bic,
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "data_length": len(series),
            }
            print(f"  ✓ ARIMA训练完成 AIC={arima_fit.aic:.2f}")
        except Exception as e:
            print(f"  ❌ ARIMA训练失败: {e}")
            self.models["arima"] = None
        finally:
            gc.collect()

    def _train_lstm(self, close_prices):
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
        except ImportError:
            print("  ⚠️ TensorFlow/Keras不可用，跳过LSTM")
            self.models["lstm"] = None
            return

        try:
            params = self._get_model_params("lstm")
            min_data_required = params.get("min_data_required", 200)
            window_size = params.get("window_size", 60)
            epochs = params.get("epochs", 50)
            batch_size = params.get("batch_size", 32)
            units = params.get("units", 64)

            if close_prices is None or len(close_prices) < min_data_required:
                print("  ⚠️ 数据不足，跳过LSTM")
                self.models["lstm"] = None
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
                Dense(1),
            ])
            model.compile(optimizer="adam", loss="mse")
            model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

            self.models["lstm"] = {
                "model": model,
                "window_size": window_size,
                "prediction_horizon": params.get("prediction_horizon", 5),
                "last_sequence": series[-window_size:].flatten(),
            }
            print("  ✓ LSTM训练完成")
            del X, y
        except Exception as e:
            print(f"  ❌ LSTM训练失败: {e}")
            self.models["lstm"] = None
        finally:
            gc.collect()

    def _train_lightgbm(self, close_prices, returns, volume):
        try:
            import lightgbm as lgb
        except ImportError:
            print("  ⚠️ LightGBM库不可用，跳过")
            self.models["lightgbm"] = None
            return
        try:
            params = self._get_model_params("lightgbm")
            min_data_required = params.get("min_data_required", 100)
            window_size = params.get("window_size", 50)
            n_estimators = params.get("n_estimators", 100)
            lr = params.get("learning_rate", 0.05)
            max_depth = params.get("max_depth", 7)

            if close_prices is None or len(close_prices) < min_data_required:
                print("  ⚠️ 数据不足，跳过LightGBM")
                self.models["lightgbm"] = None
                return

            seq_length = min(10, window_size // 5)
            prices_series = pd.Series(close_prices)
            X, y = [], []
            for i in range(seq_length, len(prices_series)):
                X.append(prices_series[i - seq_length:i].values)
                y.append(prices_series[i])
            X, y = np.array(X), np.array(y)

            model = lgb.LGBMRegressor(
                n_estimators=n_estimators, learning_rate=lr,
                max_depth=max_depth, verbose=-1,
            )
            model.fit(X, y)
            self.models["lightgbm"] = {
                "model": model,
                "window_size": window_size,
                "prediction_horizon": params.get("prediction_horizon", 5),
                "last_sequence": prices_series[-seq_length:].tolist(),
            }
            print("  ✓ LightGBM训练完成")
            del X, y
        except Exception as e:
            print(f"  ❌ LightGBM训练失败: {e}")
            self.models["lightgbm"] = None
        finally:
            gc.collect()

    def _train_catboost(self, close_prices, returns, volume):
        try:
            import catboost as cb
        except ImportError:
            print("  ⚠️ CatBoost库不可用，跳过")
            self.models["catboost"] = None
            return
        try:
            params = self._get_model_params("catboost")
            min_data_required = params.get("min_data_required", 100)
            window_size = params.get("window_size", 50)
            iterations = params.get("iterations", 100)
            lr = params.get("learning_rate", 0.05)
            depth = params.get("depth", 6)

            if close_prices is None or len(close_prices) < min_data_required:
                print("  ⚠️ 数据不足，跳过CatBoost")
                self.models["catboost"] = None
                return

            seq_length = min(10, window_size // 5)
            prices_series = pd.Series(close_prices)
            X, y = [], []
            for i in range(seq_length, len(prices_series)):
                X.append(prices_series[i - seq_length:i].values)
                y.append(prices_series[i])
            X, y = np.array(X), np.array(y)

            model = cb.CatBoostRegressor(
                iterations=iterations, learning_rate=lr, depth=depth, verbose=0,
            )
            model.fit(X, y)
            self.models["catboost"] = {
                "model": model,
                "window_size": window_size,
                "prediction_horizon": params.get("prediction_horizon", 5),
                "last_sequence": prices_series[-seq_length:].tolist(),
            }
            print("  ✓ CatBoost训练完成")
            del X, y
        except Exception as e:
            print(f"  ❌ CatBoost训练失败: {e}")
            self.models["catboost"] = None
        finally:
            gc.collect()

    # ==================== 量化因子模型 ====================

    def _train_quant_factor(self, df, close_prices, volume, high, low, open_prices):
        try:
            params = self._get_model_params("quant_factor")
            min_data_required = params.get("min_data_required", 120)
            n_factors = params.get("n_factors", 16)
            window_size = params.get("window_size", 60)
            prediction_horizon = params.get("prediction_horizon", 5)

            if df is None or close_prices is None or len(close_prices) < min_data_required:
                print(f"  ⚠️ 数据不足（最少需要{min_data_required}个点），跳过量化因子模型")
                self.models["quant_factor"] = None
                return

            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split

            min_len = min(
                len(close_prices),
                len(volume) if volume is not None else len(close_prices),
                len(high) if high is not None else len(close_prices),
                len(low) if low is not None else len(close_prices),
                len(open_prices) if open_prices is not None else len(close_prices),
            )
            data = pd.DataFrame({
                "close": close_prices[:min_len],
                "volume": volume[:min_len] if volume is not None else np.ones(min_len),
                "high": high[:min_len] if high is not None else close_prices[:min_len],
                "low": low[:min_len] if low is not None else close_prices[:min_len],
                "open": open_prices[:min_len] if open_prices is not None else close_prices[:min_len],
            })
            data = data.ffill().bfill().fillna(0)

            c = data["close"].values.astype(float)
            v = data["volume"].values.astype(float)
            h = data["high"].values.astype(float)
            l = data["low"].values.astype(float)
            ret = np.array(pd.Series(c).pct_change().fillna(0))

            f = pd.DataFrame(index=data.index)

            # 动量因子
            for w in [5, 10, 20, 60]:
                if len(c) > w:
                    mom = np.full(len(c), np.nan)
                    mom[w:] = c[w:] / c[:-w] - 1.0
                    f[f"momentum_{w}d"] = mom * 100
                else:
                    f[f"momentum_{w}d"] = 0.0

            # 短期/长期动量比
            if len(c) > 60:
                ratio = np.full(len(c), np.nan)
                short = c[20:] / c[:-20] - 1.0
                long_ = c[60:] / c[:-60] - 1.0
                min_len_mr = min(len(short), len(long_))
                ratio[60:60 + min_len_mr] = short[-min_len_mr:] / (long_[-min_len_mr:] + 1e-10)
                f["momentum_ratio"] = ratio
            else:
                f["momentum_ratio"] = 0.0

            # 波动率
            for w in [10, 20, 60]:
                f[f"volatility_{w}d"] = (
                    pd.Series(ret).rolling(w, min_periods=5).std().values * np.sqrt(252)
                )
            neg_ret = np.where(ret < 0, ret, 0)
            f["downside_volatility"] = (
                pd.Series(neg_ret).rolling(20, min_periods=5).std().values * np.sqrt(252)
            )

            # 偏度/峰度（向量化）
            skew_vals = np.zeros(len(c))
            kurt_vals = np.zeros(len(c))
            if len(ret) >= 20:
                ret_clean = np.nan_to_num(ret, nan=0.0)
                win = sliding_window_view(ret_clean, 20)
                std_win = win.std(axis=1)
                valid = std_win > 1e-10
                if valid.any():
                    sk = sp_stats.skew(win[valid], axis=1)
                    ku = sp_stats.kurtosis(win[valid], axis=1)
                    skew_vals[19:][valid] = np.nan_to_num(sk, nan=0.0)
                    kurt_vals[19:][valid] = np.nan_to_num(ku, nan=0.0)
            f["skewness_20d"] = skew_vals
            f["kurtosis_20d"] = kurt_vals

            # 趋势强度（向量化）
            f["trend_strength"] = _rolling_trend_r2(c, window=20)

            # 均线交叉
            def ma_series(arr, w):
                return pd.Series(arr).rolling(w, min_periods=1).mean().values
            ma5 = ma_series(c, 5)
            ma20 = ma_series(c, 20)
            ma60 = ma_series(c, 60)
            cross_5_20 = np.where(ma5 > ma20, 1.0, -1.0)
            cross_20_60 = np.where(ma20 > ma60, 1.0, -1.0)
            cross_5_20[:20] = 0.0
            cross_20_60[:60] = 0.0
            f["ma_cross_5_20"] = cross_5_20
            f["ma_cross_20_60"] = cross_20_60

            # 价格位置
            for w in [10, 20]:
                hh = pd.Series(h).rolling(w, min_periods=1).max().values
                ll = pd.Series(l).rolling(w, min_periods=1).min().values
                f[f"price_position_{w}d"] = (c - ll) / (hh - ll + 1e-10)

            # Amihud 流动性（向量化）
            daily_ret = np.abs(np.nan_to_num(ret, nan=0.0))
            amihud = np.zeros(len(c))
            if len(daily_ret) >= 20:
                win_ret = sliding_window_view(daily_ret, 20)
                win_vol = sliding_window_view(v, 20) + 1e-10
                amihud[19:] = (win_ret / win_vol).mean(axis=1)
            f["amihud_illiquidity"] = amihud * 1e6

            # 量比
            vol_ma = pd.Series(v).rolling(20, min_periods=1).mean().values
            f["volume_ratio"] = v / (vol_ma + 1e-10)

            # 量价背离（向量化）
            diff_arr = np.diff(c)
            pv_corr = np.zeros(len(c))
            if len(diff_arr) >= 20:
                d_win = sliding_window_view(diff_arr, 20)
                v_win = sliding_window_view(v, 20)[:len(d_win)]
                dc = d_win - d_win.mean(axis=1, keepdims=True)
                dv = v_win - v_win.mean(axis=1, keepdims=True)
                num = (dc * dv).sum(axis=1)
                den = np.sqrt((dc ** 2).sum(axis=1) * (dv ** 2).sum(axis=1)) + 1e-10
                pv_corr[20:] = num / den
            f["price_volume_divergence"] = pv_corr

            # MFI
            typical_price = (h + l + c) / 3.0
            money_flow = typical_price * v
            flow_signal = np.diff(np.concatenate([[c[0]], c]))
            pos_flow = pd.Series(np.where(flow_signal > 0, money_flow, 0))
            neg_flow = pd.Series(np.where(flow_signal < 0, money_flow, 0))
            pos_sum = pos_flow.rolling(14, min_periods=1).sum().values
            neg_sum = neg_flow.rolling(14, min_periods=1).sum().values + 1e-10
            f["mfi"] = 100 - (100 / (1 + pos_sum / neg_sum))

            f = f.ffill().bfill().fillna(0)
            if len(f.columns) > n_factors:
                top_cols = f.var().nlargest(n_factors).index
                f = f[top_cols]
            elif len(f.columns) < n_factors:
                for i in range(n_factors - len(f.columns)):
                    f[f"aux_factor_{i}"] = 0.0

            future_returns = (data["close"].shift(-prediction_horizon) / data["close"] - 1) * 100
            aligned_idx = f.index.intersection(future_returns.dropna().index)
            X = f.loc[aligned_idx]
            y = future_returns.loc[aligned_idx]

            if len(X) < 60:
                print(f"  ⚠️ 有效样本不足（{len(X)}），跳过量化因子模型")
                self.models["quant_factor"] = None
                return

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y.values, test_size=0.2, random_state=42, shuffle=False
            )

            model = RandomForestRegressor(
                n_estimators=params.get("n_estimators", 100),
                max_depth=params.get("max_depth", 8),
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            val_score = model.score(X_val, y_val)
            feature_imp = model.feature_importances_

            imp_df = pd.DataFrame({
                "factor": X.columns,
                "importance": feature_imp,
            }).sort_values("importance", ascending=False)
            top5 = imp_df.head(5)
            top5_str = ", ".join([f"{r['factor']}({r['importance']:.3f})"
                                  for _, r in top5.iterrows()])

            last_factors = X_scaled[-1:].copy()

            self.models["quant_factor"] = {
                "model": model,
                "scaler": scaler,
                "factor_names": X.columns.tolist(),
                "factor_importance": feature_imp.tolist(),
                "top_factors": top5,
                "train_score": train_score,
                "val_score": val_score,
                "window_size": window_size,
                "prediction_horizon": prediction_horizon,
                "n_factors": n_factors,
                "last_factors": last_factors,
                "last_close": float(data["close"].iloc[-1]),
                "training_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  ✓ 量化因子模型训练完成 R² train={train_score:.4f}, val={val_score:.4f}")
            print(f"    Top-5 因子: {top5_str}")

            del X, y, X_scaled, X_train, X_val, y_train, y_val
        except Exception as e:
            print(f"  ❌ 量化因子模型训练失败: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            self.models["quant_factor"] = None
        finally:
            gc.collect()

    def evaluate_models(self):
        try:
            scores = {}
            for model_name, model_info in self.models.items():
                if model_info is None:
                    scores[model_name] = 0.0
                    continue

                base_score = 0.5
                if model_name in ["arima", "sarima", "garch"]:
                    if "aic" in model_info:
                        base_score = 1.0 / (1.0 + abs(model_info["aic"]) / 1000)
                elif model_name in ["xgboost", "lightgbm", "catboost"]:
                    if "val_score" in model_info:
                        base_score = max(0, model_info["val_score"])
                    elif "train_score" in model_info:
                        base_score = max(0, model_info["train_score"])
                elif model_name == "multi_factor":
                    base_score = max(0, model_info.get("test_score", 0.5))
                elif model_name == "quant_factor":
                    base_score = max(0, model_info.get("val_score", 0.5))
                elif model_name == "lstm":
                    base_score = 0.6
                elif model_name == "ensemble":
                    base_score = 0.7

                market_adjustment = 1.0
                if self.market_regime == "high_volatility":
                    if model_name in ["garch", "multi_factor", "lightgbm", "quant_factor"]:
                        market_adjustment = 1.2
                elif self.market_regime == "trending":
                    if model_name in ["sarima", "arima", "lightgbm", "catboost", "quant_factor"]:
                        market_adjustment = 1.2

                scores[model_name] = min(1.0, base_score * market_adjustment)

            total = sum(scores.values())
            if total > 0:
                for k in scores:
                    scores[k] /= total
            print(f"模型评估分数: {scores}")
            return scores
        except Exception as e:
            print(f"模型评估出错: {e}")
            return {m: 0.2 for m in self.models.keys()}