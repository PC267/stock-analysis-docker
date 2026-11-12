# model_evaluator.py
import numpy as np


class ModelEvaluator:
    """选择可用预测模型，优先使用训练期间的时序验证分数。"""

    def __init__(self, config, models, simulations, processed_data):
        self.config = config
        self.models = models or {}
        self.simulations = simulations or {}
        self.processed_data = processed_data or {}

        if hasattr(config, 'simulation_days'):
            self.simulation_days = config.simulation_days
        elif hasattr(config, 'days_to_simulate'):
            self.simulation_days = config.days_to_simulate
        elif isinstance(config, dict):
            self.simulation_days = config.get('simulation_days', config.get('days_to_simulate', 30))
        else:
            self.simulation_days = 30

        self._best_model_cache = None

    @property
    def select_best_model(self):
        """选择最优模型；验证集分数优先，模拟路径合理性仅作后备。"""
        if self._best_model_cache is not None:
            return self._best_model_cache

        evaluated = self._evaluate_all_models()
        valid = {name: score for name, score in evaluated['scores'].items() if score > 0}
        if valid:
            best_name = max(valid, key=valid.get)
            result = (best_name, evaluated['arrays'][best_name])
            self._best_model_cache = result
            print(f"✅ 通过时序验证/合理性评分选择模型: {best_name}")
            return result

        for name in ('ensemble', 'xgboost', 'lightgbm', 'catboost', 'lstm', 'sarima', 'garch', 'multi_factor', 'arima'):
            if name in self.simulations:
                array = self._safe_process_data(self.simulations[name])
                if array is not None and array.size:
                    self._best_model_cache = (name, array)
                    return self._best_model_cache

        return self._get_fallback_model()

    def _evaluate_all_models(self):
        result = {'scores': {}, 'arrays': {}, 'valid_models': [], 'all_models': list(self.simulations)}
        for name, simulation in self.simulations.items():
            array = self._safe_process_data(simulation)
            result['arrays'][name] = array
            score = self._calculate_score(name, array) if array is not None else 0.0
            result['scores'][name] = score
            if score > 0:
                result['valid_models'].append(name)
        return result

    def _get_fallback_model(self, preferred_model='ensemble'):
        processed = self.processed_data.get('processed')
        if processed is not None and not processed.empty and 'close' in processed:
            last_price = float(processed['close'].iloc[-1])
        else:
            last_price = 1.0
        for name, simulation in self.simulations.items():
            array = self._safe_process_data(simulation)
            if array is not None and array.size:
                return name, array
        return preferred_model, np.full(max(1, int(self.simulation_days)), last_price, dtype=float)

    def _safe_process_data(self, sims):
        try:
            if isinstance(sims, dict):
                for key in ('predictions', 'values', 'simulation', 'close', 'price'):
                    if key in sims:
                        sims = sims[key]
                        break
                else:
                    return None
            array = np.asarray(sims, dtype=float).reshape(-1)
            if array.size < 2:
                return None
            # Invalid/nonpositive prices cannot be meaningfully evaluated as stock prices.
            if not np.isfinite(array).all() or np.any(array <= 0):
                return None
            return array
        except (TypeError, ValueError):
            return None

    def _calculate_score(self, model_name, sims_array):
        """Use chronological holdout score when present; otherwise score forecast sanity.

        The fallback deliberately does not claim predictive accuracy because no future
        observations are available to this evaluator.
        """
        if sims_array is None or sims_array.size < 2:
            return 0.0

        model_info = self.models.get(model_name)
        if isinstance(model_info, dict):
            validation_score = model_info.get('val_score', model_info.get('test_score'))
            if validation_score is not None:
                try:
                    validation_score = float(validation_score)
                    if np.isfinite(validation_score):
                        # Map R² to [0, 1]; negative R² is worse than the mean baseline.
                        return float(np.clip((validation_score + 1.0) / 2.0, 0.0, 1.0))
                except (TypeError, ValueError):
                    pass

        returns = np.diff(sims_array) / sims_array[:-1]
        if returns.size == 0 or not np.isfinite(returns).all():
            return 0.0

        # Reject implausible explosive paths. Keep bounds broad so legitimate volatile
        # instruments are not rejected solely by a fixed daily-return threshold.
        if np.max(np.abs(returns)) > 0.5:
            return 0.0

        historical = self.processed_data.get('processed')
        hist_returns = None
        if historical is not None and 'close' in historical and len(historical) > 2:
            close = np.asarray(historical['close'], dtype=float)
            hist_returns = np.diff(close) / close[:-1]
            hist_returns = hist_returns[np.isfinite(hist_returns)]

        if hist_returns is not None and hist_returns.size:
            # Score forecast volatility against observed training volatility; this is a
            # plausibility fallback, not an out-of-sample accuracy estimate.
            observed_vol = float(np.std(hist_returns))
            forecast_vol = float(np.std(returns))
            scale = max(observed_vol, 1e-6)
            volatility_score = float(np.exp(-abs(forecast_vol - observed_vol) / scale))
        else:
            volatility_score = 0.5

        return volatility_score

    # Keep these helpers for callers that may use the previous internal API.
    def _select_by_standard_method(self, evaluation_result):
        scores = evaluation_result.get('scores', {})
        valid = {name: score for name, score in scores.items() if score > 0}
        if not valid:
            return None
        name = max(valid, key=valid.get)
        array = evaluation_result.get('arrays', {}).get(name)
        return (name, array) if array is not None and len(array) >= 2 else None

    def _select_by_backup_method(self, evaluation_result):
        for name in evaluation_result.get('all_models', []):
            array = evaluation_result.get('arrays', {}).get(name)
            if array is not None and len(array) > 0:
                return name, array
        return None
