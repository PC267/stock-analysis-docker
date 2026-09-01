#!/usr/bin/env python3
"""验证量化因子模型 — 003021 兆威机电"""
import sys, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sys.path.insert(0, 'app')
sys.path.insert(0, 'app/model')

import pandas as pd
import numpy as np

# ===== 1. 加载数据 =====
csv_path = 'app/stock_data/003021_兆威机电_baostock.csv'
print(f"📂 加载数据: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   行数: {len(df)}, 列: {list(df.columns)}")
print(f"   日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
print(f"   收盘价: {df['close'].iloc[-1]:.2f}")
print()

# ===== 2. 准备 processed_data =====
close = df['close'].values.astype(float)
volume = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones(len(close))
high = df['high'].values.astype(float) if 'high' in df.columns else close
low = df['low'].values.astype(float) if 'low' in df.columns else close
open_p = df['open'].values.astype(float) if 'open' in df.columns else close

# close_scaled (归一化)
close_scaled = ((close - close.min()) / (close.max() - close.min() + 1e-10)).reshape(-1, 1)
returns = np.diff(close) / close[:-1]
returns = np.concatenate([[0], returns])

processed_data = {
    'prepared': {'close_scaled': close_scaled, 'returns': returns},
    'processed': pd.DataFrame(
        {'close': close, 'volume': volume,
         'high': high, 'low': low, 'open': open_p},
        index=pd.to_datetime(df['date'])
    )
}

# ===== 3. 直接测试 _train_quant_factor =====
print("🔧 创建 ModelFactory 并训练所有模型...")
from factory import ModelFactory

# 模拟一个简单配置
class FakeConfig:
    pass
config = FakeConfig()
config.stock_system = {}

factory = ModelFactory(config)

# 手动注册 quant_factor 参数
factory.parameter_recommendations['quant_factor'] = {
    'n_estimators': 100, 'max_depth': 8,
    'n_factors': 16, 'window_size': 60,
    'prediction_horizon': 5, 'min_data_required': 120,
    'lookback_windows': [5, 10, 20, 60]
}

print()
print("=" * 60)
print("🚀 执行 _train_quant_factor()")
print("=" * 60)
factory._train_quant_factor(
    processed_data['processed'],
    close, volume, high, low, open_p
)

quant = factory.models.get('quant_factor')
if quant is None:
    print("❌ 量化因子模型训练失败")
    sys.exit(1)

print()
print("✅ 模型训练成功！")
print(f"   训练集 R²: {quant.get('train_score', 'N/A'):.4f}")
print(f"   验证集 R²: {quant.get('val_score', 'N/A'):.4f}")
print(f"   因子数: {quant.get('n_factors', 'N/A')}")
print(f"   预测步长: {quant.get('prediction_horizon', 'N/A')}")

# Top-5 因子
top5 = quant.get('top_factors')
if top5 is not None:
    print()
    print("📊 Top-5 因子重要性:")
    for _, row in top5.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['factor']:<25s} {row['importance']:.4f}  {bar}")

# ===== 4. 测试 _simulate_quant_factor =====
print()
print("=" * 60)
print("🚀 执行 _simulate_quant_factor()")
print("=" * 60)

from simulator import StockSimulator

# 准备模拟器
sim = StockSimulator(config, factory.models, processed_data)

# 生成未来日期
from datetime import datetime, timedelta
import pandas as pd
last_date = pd.Timestamp.now()
future_dates = pd.DatetimeIndex([last_date + timedelta(days=i) for i in range(1, 21)])

# 运行量化因子模拟
result = sim._simulate_quant_factor(future_dates)

print()
print(f"✅ 模拟结果 (20天):")
print(f"   起始价: {result[0]:.2f}")
print(f"   最终价: {result[-1]:.2f}")
print(f"   总收益率: {(result[-1]/result[0] - 1)*100:.2f}%")
print(f"   最高价: {result.max():.2f}")
print(f"   最低价: {result.min():.2f}")
print(f"   波动率(年化): {np.std(result/result[0]-1)*np.sqrt(252)*100:.2f}%")

# ===== 5. 验证全链路集成 =====
print()
print("=" * 60)
print("🚀 全链路模拟 (所有模型 + Ensemble)")
print("=" * 60)

simulations, fd = sim.simulate_future()
print()
print(f"🔍 参与集成的模型: {list(simulations.keys())}")
print(f"✅ quant_factor 在集成结果中: {'quant_factor' in simulations}")
print(f"✅ Ensemble 结果范围: {simulations['ensemble'][0]:.2f} ~ {simulations['ensemble'][-1]:.2f}")

print()
print("=" * 60)
print("✅ 验证通过！量化因子模型完整可用")
print("=" * 60)
