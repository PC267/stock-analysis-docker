# model_evaluator.py
import numpy as np


class ModelEvaluator:
    def __init__(self, config, models, simulations, processed_data):
        self.config = config
        self.models = models
        self.simulations = simulations
        self.processed_data = processed_data

        # 兼容模拟天数配置（支持 simulation_days 或 days_to_simulate）
        if hasattr(config, 'simulation_days'):
            self.simulation_days = config.simulation_days
        elif hasattr(config, 'days_to_simulate'):
            self.simulation_days = config.days_to_simulate
        elif isinstance(config, dict):
            self.simulation_days = config.get('simulation_days', config.get('days_to_simulate', 30))
        else:
            self.simulation_days = 30

        self._best_model_cache = None  # 缓存最优模型结果

    @property
    def select_best_model(self):
        """选择最优模型"""
        # 如果已经缓存了结果，直接返回
        if self._best_model_cache is not None:
            return self._best_model_cache

        print("开始模型评估...")

        # 记录评估结果
        evaluation_result = self._evaluate_all_models()

        # 尝试标准方法选择最优模型
        best_model_info = self._select_by_standard_method(evaluation_result)

        if best_model_info is not None:
            model_name, sims_array = best_model_info
            print(f"✅ 通过标准方法选择最优模型: {model_name}")
            self._best_model_cache = (model_name, sims_array)
            return self._best_model_cache

        # 如果标准方法失败，使用备选方法
        print("\n⚠️ 标准方法失败，使用备选方案...")
        backup_model_info = self._select_by_backup_method(evaluation_result)

        if backup_model_info is not None:
            model_name, sims_array = backup_model_info
            print(f"✅ 通过备选方案选择模型: {model_name}")
            self._best_model_cache = (model_name, sims_array)
            return self._best_model_cache

        # 如果所有方法都失败，使用硬编码备选
        print("⚠️ 所有方法都失败，使用硬编码备选方案")
        model_name = 'lstm' if 'lstm' in self.simulations else 'arima'
        return self._get_fallback_model(model_name)

    def _evaluate_all_models(self):
        """评估所有模型"""
        evaluation_result = {
            'scores': {},
            'arrays': {},
            'valid_models': [],
            'all_models': list(self.simulations.keys())
        }

        print("正在评估以下模型:", ', '.join(evaluation_result['all_models']))

        for model_name, sims in self.simulations.items():
            # 处理数据
            sims_array = self._safe_process_data(sims)
            evaluation_result['arrays'][model_name] = sims_array

            if sims_array is None:
                print(f"  {model_name:20s} ❌ 数据无效")
                evaluation_result['scores'][model_name] = 0
                continue

            # 计算得分
            score = self._calculate_score(model_name, sims_array)
            evaluation_result['scores'][model_name] = score

            if score > 0:
                evaluation_result['valid_models'].append(model_name)
                print(f"  {model_name:20s} ✅ 得分: {score:.2f}")
            else:
                print(f"  {model_name:20s} ❌ 得分: 0")

        print(f"\n评估完成，有效模型: {len(evaluation_result['valid_models'])}/{len(evaluation_result['all_models'])}")
        return evaluation_result

    def _select_by_standard_method(self, evaluation_result):
        """使用标准方法选择最优模型"""
        valid_scores = {k: v for k, v in evaluation_result['scores'].items() if v > 0}

        if not valid_scores:
            print("❌ 标准方法: 没有有效得分的模型")
            return None

        # 找到得分最高的模型
        best_model = max(valid_scores, key=valid_scores.get)
        best_score = valid_scores[best_model]

        # 获取对应的模拟数据
        best_array = evaluation_result['arrays'][best_model]

        if best_array is None or len(best_array) < 2:
            print(f"❌ 标准方法: 最优模型 {best_model} 数据无效")
            return None

        print(f"\n🏆 标准方法找到最优模型: {best_model} (得分: {best_score:.2f})")
        return best_model, best_array

    def _select_by_backup_method(self, evaluation_result):
        """使用备选方法选择模型（包含所有新增模型）"""
        print("\n正在使用备选方案...")

        available_models = list(self.simulations.keys())

        # 策略1: 选择数组长度合适的模型
        for model_name in available_models:
            array = evaluation_result['arrays'][model_name]
            if array is not None and len(array) >= self.simulation_days:
                print(f"备选策略1: 选择数据完整的模型 {model_name}")
                return model_name, array

        # 策略2: 选择任意有数据的模型
        for model_name in available_models:
            array = evaluation_result['arrays'][model_name]
            if array is not None and len(array) > 0:
                print(f"备选策略2: 选择有数据的模型 {model_name}")
                return model_name, array

        # 策略3: 按优先级选择（包含全部模型，新模型优先级较高）
        model_priority = [
            'ensemble', 'xgboost', 'lightgbm', 'catboost', 'lstm',
            'sarima', 'garch', 'multi_factor', 'arima'
        ]
        for model_name in model_priority:
            if model_name in available_models:
                print(f"备选策略3: 按优先级选择 {model_name}")
                sims = self.simulations[model_name]
                array = self._safe_process_data(sims)
                if array is not None:
                    return model_name, array

        return None

    def _get_fallback_model(self, preferred_model='xgboost'):
        """获取备选模型（硬编码）"""
        print(f"使用硬编码备选方案，选择模型: {preferred_model}")

        try:
            last_price = self.processed_data['processed']['close'].iloc[-1]
            default_sim = np.full(self.simulation_days, last_price)

            if preferred_model in self.simulations:
                # 尝试使用该模型的原始数据
                sims = self.simulations[preferred_model]
                array = self._safe_process_data(sims)
                if array is not None:
                    return preferred_model, array

            # 如果指定模型不可用，尝试从所有模型中任选一个
            for model_name in self.simulations.keys():
                array = self._safe_process_data(self.simulations[model_name])
                if array is not None:
                    return model_name, array

            return preferred_model, default_sim
        except:
            default_sim = np.full(self.simulation_days, 100.0)  # 默认值
            return 'arima', default_sim

    def _safe_process_data(self, sims):
        """安全地处理数据"""
        try:
            if isinstance(sims, np.ndarray):
                return sims.astype(float)
            elif isinstance(sims, dict):
                # 查找数值数据
                for key in ['predictions', 'values', 'simulation', 'close', 'price']:
                    if key in sims:
                        data = sims[key]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)
                # 尝试转换所有值
                return np.array(list(sims.values()), dtype=float)
            elif isinstance(sims, list):
                return np.array(sims, dtype=float)
            elif hasattr(sims, '__array__'):
                return np.array(sims, dtype=float)
            else:
                return None
        except Exception as e:
            print(f"数据处理失败: {e}")
            return None

    def _calculate_score(self, model_name, sims_array):
        """计算模型得分"""
        try:
            # 过滤无效值
            valid_mask = np.isfinite(sims_array)
            valid_sims = sims_array[valid_mask]

            if len(valid_sims) < 2:
                return 0

            # 计算稳定性得分
            returns = np.diff(valid_sims) / valid_sims[:-1]
            valid_returns = returns[np.isfinite(returns)]

            if len(valid_returns) == 0:
                stability = 0.5
            else:
                std_dev = np.std(valid_returns)
                stability = 1.0 / (std_dev + 1e-10) if std_dev > 0 else 10.0

            # 计算合理性得分
            if len(valid_returns) > 0:
                max_change = np.max(np.abs(valid_returns))
                reasonableness = 1.0 / (max_change + 0.05)
            else:
                reasonableness = 0.5

            # 计算趋势得分
            try:
                if 'close' in self.processed_data['processed'].columns:
                    hist_data = self.processed_data['processed']['close']
                    if len(hist_data) >= 20:
                        hist_trend = 1 if hist_data.iloc[-1] > hist_data.iloc[-20] else -1
                    else:
                        hist_trend = 1 if hist_data.iloc[-1] > hist_data.iloc[0] else -1
                else:
                    hist_trend = 1
            except:
                hist_trend = 1

            sim_trend = 1 if valid_sims[-1] > valid_sims[0] else -1
            trend_score = 1.0 if hist_trend == sim_trend else 0.5

            # 综合得分
            combined_score = (
                    min(stability, 10) * 0.4 +
                    reasonableness * 0.3 +
                    trend_score * 0.3
            )

            return combined_score

        except Exception as e:
            print(f"评分失败 ({model_name}): {e}")
            return 0