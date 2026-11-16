# data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _ensure_numeric_columns(df):
        if df is None or df.empty:
            return df

        numeric_cols = [
            'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg',
            'returns', 'MA5', 'MA10', 'MA20', 'MA13', 'MA30', 'RSI', 'volatility',
            'break_ma30', 'is_strong_trend', 'MACD_DIF', 'MACD_DEA', 'KDJ_K', 'KDJ_D', 
            'X_div', 'Y_div', 'Z_div', 'XYZ_Resonance'  # 🌟 新增特征
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)
        return df

    def process_data(self, stock_data):
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

            print("  添加技术指标（含策略特征）...")
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
        try:
            if len(df) < 5:
                return df

            df = df.copy()

            # 基础特征
            df['returns'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan).astype(np.float32)
            df['MA5'] = df['close'].rolling(window=5, min_periods=1).mean().astype(np.float32)
            df['MA10'] = df['close'].rolling(window=10, min_periods=1).mean().astype(np.float32)
            df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean().astype(np.float32)

            # RSI
            delta = df['close'].diff().astype(np.float32)
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean().astype(np.float32)
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean().astype(np.float32)
            rs = gain / loss.replace(0, np.nan)
            df['RSI'] = (100 - (100 / (1 + rs))).fillna(50).astype(np.float32)

            df['volatility'] = (
                df['returns'].rolling(window=20, min_periods=1).std().fillna(0).astype(np.float32) * np.sqrt(252)
            )

            # ==========================================================
            # 🌟 新增1：图4 强势股二次启动策略指标
            # ==========================================================
            df['MA13'] = df['close'].rolling(window=13, min_periods=1).mean().astype(np.float32)
            df['MA30'] = df['close'].rolling(window=30, min_periods=1).mean().astype(np.float32)
            
            # 确认强势：股价在13日均线之上
            df['is_strong_trend'] = (df['close'] > df['MA13']).astype(np.float32)
            # 确认机会：今日收盘价突破30日均线
            df['break_ma30'] = ((df['close'] > df['MA30']) & (df['close'].shift(1) <= df['MA30'].shift(1))).astype(np.float32)

            # ==========================================================
            # 🌟 新增2：图1-3 多因子背离指标 (X/Y/Z)
            # ==========================================================
            # 计算MACD (Y因子基础)
            exp12 = df['close'].ewm(span=12, adjust=False).mean()
            exp26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD_DIF'] = (exp12 - exp26).astype(np.float32)
            df['MACD_DEA'] = df['MACD_DIF'].ewm(span=9, adjust=False).mean().astype(np.float32)

            # 计算KDJ (Y因子基础)
            low_9 = df['low'].rolling(window=9, min_periods=1).min()
            high_9 = df['high'].rolling(window=9, min_periods=1).max()
            rsv = (df['close'] - low_9) / (high_9 - low_9 + 1e-10) * 100
            df['KDJ_K'] = rsv.ewm(com=2, adjust=False).mean().fillna(50).astype(np.float32)
            df['KDJ_D'] = df['KDJ_K'].ewm(com=2, adjust=False).mean().fillna(50).astype(np.float32)

            # 背离检测 (使用简化逻辑：近14日价格创新低，但指标未创新低)
            # 寻找价格低点 (14天内最低点)
            price_low_14 = df['close'].rolling(window=14, min_periods=1).min()
            is_price_low = (df['close'] == price_low_14)
            
            # X因子：价量背离 (价创新低，量未创新低)
            vol_low_14 = df['volume'].rolling(window=14, min_periods=1).min()
            df['X_div'] = (is_price_low & (df['volume'] > vol_low_14)).astype(np.float32)
            
            # Y因子：MACD/KDJ背离 (价创新低，MACD/KDJ未创新低)
            macd_low_14 = df['MACD_DIF'].rolling(window=14, min_periods=1).min()
            kdj_low_14 = df['KDJ_K'].rolling(window=14, min_periods=1).min()
            df['Y_div'] = (is_price_low & (df['MACD_DIF'] > macd_low_14) & (df['KDJ_K'] > kdj_low_14)).astype(np.float32)
            
            # Z因子：RSI背离 (价创新低，RSI未创新低)
            rsi_low_14 = df['RSI'].rolling(window=14, min_periods=1).min()
            df['Z_div'] = (is_price_low & (df['RSI'] > rsi_low_14)).astype(np.float32)

            # 因子共振：5日内至少有2个信号出现
            div_sum = df['X_div'].rolling(5, min_periods=1).sum() + df['Y_div'].rolling(5, min_periods=1).sum() + df['Z_div'].rolling(5, min_periods=1).sum()
            df['XYZ_Resonance'] = (div_sum >= 2).astype(np.float32)

            return df

        except Exception as e:
            print(f"  添加技术指标失败: {e}")
            return df

    def clean_data(self, df):
        try:
            if df is None:
                return None

            df = df.copy()
            if 'date' in df.columns:
                df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

            original_len = len(df)
            # 🌟 已将新增的策略字段加入白名单
            required_cols = [
                'open', 'high', 'low', 'close', 'volume', 'returns', 'MA5', 'MA10', 'MA20', 'RSI', 'volatility',
                'MA13', 'MA30', 'is_strong_trend', 'break_ma30', 'MACD_DIF', 'MACD_DEA', 'KDJ_K', 'KDJ_D', 'X_div', 'Y_div', 'Z_div', 'XYZ_Resonance'
            ]
            existing_cols = [c for c in required_cols if c in df.columns]
            if not existing_cols:
                print("  ⚠️ 没有可用于训练的有效列")
                return None

            df = df.loc[:, list(dict.fromkeys(existing_cols))].dropna().copy()
            cleaned_len = len(df)

            if cleaned_len == 0 or cleaned_len < 5:
                print("  ⚠️ 清洗后数据为空或过少")
                return None

            print(f"  数据清洗: {original_len} -> {cleaned_len} 条记录")
            return df.astype({c: np.float32 for c in existing_cols if c != 'date'})

        except Exception as e:
            print(f"  数据清洗失败: {e}")
            return None

    @staticmethod
    def _safe_fit_scaler(values):
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0 or not np.isfinite(arr).any():
            return None
        if np.allclose(arr, arr[0], rtol=0, atol=1e-12):
            return np.zeros_like(arr, dtype=np.float32)
        return arr

    def prepare_for_training(self, df, fit_scaler=True):
        try:
            if df is None or len(df) == 0:
                print("  ⚠️ 准备训练数据时输入为空")
                return {}

            df = df.copy()
            df = self._ensure_numeric_columns(df)

            if not {'close', 'volume'}.issubset(df.columns):
                print("  ⚠️ 缺少 close/volume 字段，无法准备训练数据")
                return {}

            # 🌟 加入新字段作为模型学习的因子
            valid_cols = [
                c for c in ['close', 'volume', 'open', 'high', 'low', 'returns', 'MA5', 'MA10', 'MA20', 'RSI', 'volatility',
                            'MA13', 'MA30', 'is_strong_trend', 'break_ma30', 'XYZ_Resonance'] 
                if c in df.columns
            ]
            df = df[valid_cols].dropna().copy().astype(np.float32)

            if len(df) == 0:
                return {}

            if not np.isfinite(df['close'].to_numpy(dtype=np.float32, copy=False)).all():
                return {}

            close_prices = df['close'].to_numpy(dtype=np.float32).reshape(-1, 1)
            scaler_close = MinMaxScaler()
            if fit_scaler:
                close_scaled = scaler_close.fit_transform(close_prices)
            else:
                close_scaled = scaler_close.transform(close_prices)

            volume = df['volume'].to_numpy(dtype=np.float32).reshape(-1, 1)
            scaler_volume = MinMaxScaler()
            if fit_scaler:
                volume_scaled = scaler_volume.fit_transform(volume)
            else:
                volume_scaled = scaler_volume.transform(volume)

            factor_data = {}
            # 🌟 将技术策略指标加入因子池 (factor_cols)
            factor_cols = ['MA5', 'MA10', 'MA20', 'MA13', 'MA30', 'RSI', 'volatility', 'break_ma30', 'is_strong_trend', 'XYZ_Resonance']
            factor_df = df[[c for c in factor_cols if c in df.columns]].dropna().astype(np.float32)

            if len(factor_df) > 0:
                factor_scaler = MinMaxScaler()
                factors_scaled = factor_scaler.fit_transform(factor_df.to_numpy(dtype=np.float32, copy=False))
                factor_data = {
                    'df': factor_df,
                    'scaled': factors_scaled.astype(np.float32),
                    'scaler': factor_scaler,
                    'columns': factor_df.columns.tolist()
                }
            else:
                print("  ⚠️ 因子数据为空")

            returns_data = df['returns'].to_numpy(dtype=np.float32, copy=False) if 'returns' in df.columns else None

            return {
                'close_scaled': close_scaled.astype(np.float32),
                'volume_scaled': volume_scaled.astype(np.float32),
                'scaler_close': scaler_close,
                'scaler_volume': scaler_volume,
                'factor_data': factor_data,
                'returns': returns_data,
                'close_prices': df['close'].to_numpy(dtype=np.float32, copy=False),
                'index': df.index.to_numpy()
            }

        except Exception as e:
            print(f"  准备训练数据失败: {e}")
            return {}


# compatibility alias
prepare_for_training = DataProcessor().prepare_for_training