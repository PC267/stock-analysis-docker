# data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _ensure_numeric_columns(df):
        """统一数值字段类型，避免后续训练时出现字符串/NaN。"""
        if df is None or df.empty:
            return df

        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg',
            'returns', 'MA5', 'MA10', 'MA20', 'RSI', 'volatility'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def process_data(self, stock_data):
        """处理原始数据 - 添加完整的错误处理与时间顺序校验。"""
        print("处理数据...")

        if stock_data is None or len(stock_data) == 0:
            print("❌ 输入数据为空")
            return None

        try:
            if not isinstance(stock_data, pd.DataFrame):
                stock_data = pd.DataFrame(stock_data)

            stock_data = stock_data.copy()
            stock_data = self._ensure_numeric_columns(stock_data)

            if 'date' in stock_data.columns:
                stock_data['date'] = pd.to_datetime(stock_data['date'], errors='coerce')
                stock_data = stock_data.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

            if set(['open', 'high', 'low', 'close']).intersection(stock_data.columns):
                required = ['open', 'high', 'low', 'close']
                for col in required:
                    if col in stock_data.columns:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                stock_data = stock_data[(stock_data[['open', 'high', 'low', 'close']].notna().all(axis=1))].copy()

            if len(stock_data) < 5:
                print("❌ 数据长度不足，无法进行特征提取")
                return None

            if stock_data['close'].max() <= 0:
                print("❌ 收盘价异常：最大值 <= 0")
                return None

            print("  添加技术指标...")
            data_with_indicators = self.add_technical_indicators(stock_data)

            print("  数据清洗...")
            cleaned_data = self.clean_data(data_with_indicators)

            if cleaned_data is None or len(cleaned_data) == 0:
                print("❌ 数据清洗后为空")
                return None

            print("  数据标准化...")
            prepared_data = self.prepare_for_training(cleaned_data, fit_scaler=True)

            print(f"✓ 数据处理完成，数据量: {len(cleaned_data)}")

            return {
                'raw': stock_data,
                'processed': cleaned_data,
                'prepared': prepared_data
            }

        except Exception as e:
            print(f"❌ 数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def add_technical_indicators(self, df):
        """添加技术指标。注意：这里只使用当前及过去数据，不使用未来数据。"""
        try:
            if len(df) < 5:
                return df

            df = df.copy()

            # 收益率：只对当前数据及过去数据计算，避免未来 leak
            df['returns'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)

            # 移动平均线
            df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean()
            df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()

            # 相对强弱指标 (RSI) - 简化版
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = (100 - (100 / (1 + rs))).fillna(50)

            # 波动率
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std().fillna(0) * np.sqrt(252)

            return df

        except Exception as e:
            print(f"  添加技术指标失败: {e}")
            return df

    def clean_data(self, df):
        """数据清洗：删除无效值并保持时间顺序。"""
        try:
            if df is None:
                return None

            df = df.copy()
            if 'date' in df.columns:
                df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

            original_len = len(df)
            df = df.dropna()
            cleaned_len = len(df)

            if cleaned_len == 0:
                print("  ⚠️ 清洗后数据为空")
                return None

            if cleaned_len < 5:
                print("  ⚠️ 清洗后数据过少，无法继续训练")
                return None

            print(f"  数据清洗: {original_len} -> {cleaned_len} 条记录")
            return df

        except Exception as e:
            print(f"  数据清洗失败: {e}")
            return None

    @staticmethod
    def _safe_fit_scaler(values):
        """缩放器安全拟合：避免全空/常量数组导致失败。"""
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            return None
        if np.allclose(arr, arr[0], rtol=0, atol=1e-12):
            return np.zeros_like(arr)
        return arr

    def prepare_for_training(self, df, fit_scaler=True):
        """准备训练数据；默认在训练前先完成时间顺序一致性检查，并且不在这里做未来信息拟合。"""
        try:
            if df is None or len(df) == 0:
                print("  ⚠️ 准备训练数据时输入为空")
                return {}

            df = df.copy()
            df = self._ensure_numeric_columns(df)

            if not {'close', 'volume'}.issubset(df.columns):
                print("  ⚠️ 缺少 close/volume 字段，无法准备训练数据")
                return {}

            # 仅保留有效数值列
            valid_cols = [c for c in ['close', 'volume', 'open', 'high', 'low', 'returns', 'MA5', 'MA10', 'MA20', 'RSI', 'volatility'] if c in df.columns]
            df = df[valid_cols].dropna().copy()

            if len(df) == 0:
                print("  ⚠️ 过滤无效数据后为空")
                return {}

            # 训练前先做时序和数值校验，确保不在未来时刻反向修正过去数据
            if not np.isfinite(df['close'].to_numpy()).all():
                print("  ⚠️ close 列存在非有限值")
                return {}

            # 准备收盘价数据
            close_prices = df['close'].values.reshape(-1, 1)
            scaler_close = MinMaxScaler()
            if fit_scaler:
                close_scaled = scaler_close.fit_transform(close_prices)
            else:
                close_scaled = scaler_close.transform(close_prices)

            # 准备成交量数据
            volume = df['volume'].values.reshape(-1, 1)
            scaler_volume = MinMaxScaler()
            if fit_scaler:
                volume_scaled = scaler_volume.fit_transform(volume)
            else:
                volume_scaled = scaler_volume.transform(volume)

            # 准备多因子数据
            factor_data = {}
            factor_cols = ['MA5', 'MA10', 'MA20', 'RSI', 'volatility']
            factor_df = df[[c for c in factor_cols if c in df.columns]].dropna()

            if len(factor_df) > 0:
                factor_scaler = MinMaxScaler()
                factors_scaled = factor_scaler.fit_transform(factor_df.values)
                factor_data = {
                    'df': factor_df,
                    'scaled': factors_scaled,
                    'scaler': factor_scaler,
                    'columns': factor_df.columns.tolist()
                }
            else:
                print("  ⚠️ 因子数据为空")

            returns_data = df['returns'].values if 'returns' in df.columns else None

            return {
                'close_scaled': close_scaled,
                'volume_scaled': volume_scaled,
                'scaler_close': scaler_close,
                'scaler_volume': scaler_volume,
                'factor_data': factor_data,
                'returns': returns_data,
                'close_prices': df['close'].values,
                'index': df.index.to_numpy()
            }

        except Exception as e:
            print(f"  准备训练数据失败: {e}")
            return {}


# compatibility alias
prepare_for_training = DataProcessor().prepare_for_training
