# data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self):
        pass

    def process_data(self, stock_data):
        """处理原始数据 - 添加完整的错误处理"""
        print("处理数据...")

        if stock_data is None or len(stock_data) == 0:
            print("❌ 输入数据为空")
            return None

        try:
            # 添加技术指标
            print("  添加技术指标...")
            data_with_indicators = self.add_technical_indicators(stock_data.copy())

            # 数据清洗
            print("  数据清洗...")
            cleaned_data = self.clean_data(data_with_indicators)

            if cleaned_data is None or len(cleaned_data) == 0:
                print("❌ 数据清洗后为空")
                return None

            # 数据标准化准备
            print("  数据标准化...")
            prepared_data = self.prepare_for_training(cleaned_data)

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
        """添加技术指标"""
        try:
            # 确保有足够的数据
            if len(df) < 5:
                return df

            # 收益率
            df['returns'] = df['close'].pct_change()

            # 移动平均线
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()

            # 相对强弱指标 (RSI) - 简化版
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # 波动率
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

            return df

        except Exception as e:
            print(f"  添加技术指标失败: {e}")
            return df

    def clean_data(self, df):
        """数据清洗"""
        try:
            if df is None:
                return None

            # 删除缺失值
            original_len = len(df)
            df = df.dropna()
            cleaned_len = len(df)

            if cleaned_len == 0:
                print("  ⚠️ 清洗后数据为空")
                return None

            print(f"  数据清洗: {original_len} -> {cleaned_len} 条记录")

            return df

        except Exception as e:
            print(f"  数据清洗失败: {e}")
            return None

    def prepare_for_training(self, df):
        """准备训练数据"""
        try:
            if df is None or len(df) == 0:
                print("  ⚠️ 准备训练数据时输入为空")
                return {}

            # 准备收盘价数据
            close_prices = df['close'].values.reshape(-1, 1)
            scaler_close = MinMaxScaler()
            close_scaled = scaler_close.fit_transform(close_prices)

            # 准备成交量数据
            volume = df['volume'].values.reshape(-1, 1)
            scaler_volume = MinMaxScaler()
            volume_scaled = scaler_volume.fit_transform(volume)

            # 准备多因子数据
            factor_data = {}
            if 'MA5' in df.columns:
                factor_cols = ['MA5', 'MA10', 'MA20', 'RSI', 'volatility']
                factor_df = df[factor_cols].dropna()

                if len(factor_df) > 0:
                    factor_scaler = MinMaxScaler()
                    factors_scaled = factor_scaler.fit_transform(factor_df.values)

                    factor_data = {
                        'df': factor_df,
                        'scaled': factors_scaled,
                        'scaler': factor_scaler,
                        'columns': factor_cols
                    }
                else:
                    print("  ⚠️ 因子数据为空")

            returns_data = None
            if 'returns' in df.columns:
                returns_data = df['returns'].values

            return {
                'close_scaled': close_scaled,
                'volume_scaled': volume_scaled,
                'scaler_close': scaler_close,
                'scaler_volume': scaler_volume,
                'factor_data': factor_data,
                'returns': returns_data,
                'close_prices': df['close'].values
            }

        except Exception as e:
            print(f"  准备训练数据失败: {e}")
            return {}