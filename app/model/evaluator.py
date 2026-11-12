import numpy as np

try:
    from .model_validation_guard import safe_numeric_array
except ImportError:
    try:
        from model_validation_guard import safe_numeric_array
    except ImportError:
        from app.model.model_validation_guard import safe_numeric_array


class ModelEvaluator:
    """Select models using genuine chronological holdout metrics only."""

    VALIDATION_SCORE_KEYS = ('val_score', 'validation_score', 'test_score')
    FALLBACK_PRIORITY = (
        'ensemble', 'xgboost', 'lightgbm', 'catboost', 'lstm',
        'sarima', 'garch', 'multi_factor', 'quant_factor', 'arima'
    )

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
        """Prefer positive chronological holdout R²; sanity is not accuracy."""
        if self._best_model_cache is not None:
            return self._best_model_cache

        evaluated = self._evaluate_all_models()
        scored_models = evaluated['validation_scores']
        if scored_models:
            best_name = max(scored_models, key=scored_models.get)
            result = (best_name, evaluated['arrays'][best_name])
            self._best_model_cache = result
            print(f"✅ 依据时序留出集评分选择模型: {best_name}")
            return result

        # No model has positive holdout skill. Select a usable path deterministically,
        # but do not claim that the fallback order represents forecast accuracy.
        for name in self.FALLBACK_PRIORITY:
            array = evaluated['arrays'].get(name)
            if array is not None:
                self._best_model_cache = (name, array)
                print(f"⚠️ 缺少正向时序验证证据，按可用性回退选择模型: {name}")
                return self._best_model_cache

        return self._get_fallback_model()

    def _get_validation_score(self, model_name):
        """Return finite positive chronological holdout R², else None.

        A non-positive R² indicates no improvement over a constant-mean baseline,
        so such a model is not used as evidence for selecting a predictive model.
        """
        model_info = self.models.get(model_name)
        if not isinstance(model_info, dict):
            return None

        for key in self.VALIDATION_SCORE_KEYS:
            value = model_info.get(key)
            if value is None:
                continue
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(score) and score > 0.0:
                return score
        return None

    def _evaluate_all_models(self):
        result = {
            'scores': {},
            'validation_scores': {},
            'arrays': {},
            'valid_models': [],
            'all_models': list(self.simulations)
        }
        for name, simulation in self.simulations.items():
            array = self._safe_process_data(simulation)
            result['arrays'][name] = array
            validation_score = self._get_validation_score(name) if array is not None else None
            if validation_score is not None:
                result['scores'][name] = validation_score
                result['validation_scores'][name] = validation_score
                result['valid_models'].append(name)
            else:
                result['scores'][name] = 0.0
        return result

    def _get_fallback_model(self, preferred_model='ensemble'):
        processed = self.processed_data.get('processed')
        if processed is not None and not processed.empty and 'close' in processed:
            last_price = float(processed['close'].iloc[-1])
        else:
            last_price = 1.0
        for name in self.FALLBACK_PRIORITY:
            array = self._safe_process_data(self.simulations.get(name))
            if array is not None:
                return name, array
        return preferred_model, np.full(max(1, int(self.simulation_days)), last_price, dtype=np.float32)

    def _safe_process_data(self, sims):
        try:
            if isinstance(sims, dict):
                for key in ('predictions', 'values', 'simulation', 'close', 'price'):
                    if key in sims:
                        sims = sims[key]
                        break
                else:
                    return None
            array = safe_numeric_array(sims)
            if array is None or array.size < 2 or np.any(array <= 0):
                return None
            return array
        except (TypeError, ValueError):
            return None

    def _calculate_score(self, model_name, sims_array):
        """Return only genuine validation evidence, never simulation sanity as accuracy."""
        if sims_array is None or sims_array.size < 2:
            return 0.0
        score = self._get_validation_score(model_name)
        return score if score is not None else 0.0

    def _select_by_standard_method(self, evaluation_result):
        scores = evaluation_result.get('validation_scores', evaluation_result.get('scores', {}))
        valid = {name: score for name, score in scores.items() if np.isfinite(score) and score > 0.0}
        if not valid:
            return None
        name = max(valid, key=valid.get)
        array = evaluation_result.get('arrays', {}).get(name)
        return (name, array) if array is not None and len(array) >= 2 else None

    def _select_by_backup_method(self, evaluation_result):
        arrays = evaluation_result.get('arrays', {})
        for name in self.FALLBACK_PRIORITY:
            array = arrays.get(name)
            if array is not None and len(array) >= 2:
                return name, array
        return None
