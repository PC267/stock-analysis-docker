# data_fetcher.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Optional, Dict, Any, Tuple
import time

warnings.filterwarnings('ignore')


class DataFetcher:
    def __init__(self, config):
        """初始化数据获取器

        Args:
            config: 配置对象

        Raises:
            ValueError: 配置无效
        """
        if config is None:
            raise ValueError("Config不能为None")
        if not hasattr(config, 'stock_code') or not hasattr(config, 'stock_name'):
            raise ValueError("Config缺少必要的股票信息属性")

        self.config = config
        self._baostock_logged_in = False
        print(f"数据获取器初始化，股票: {config.stock_name}({config.stock_code})")

    def __del__(self):
        """析构函数，确保退出时注销baostock"""
        self._safe_logout_baostock()

    def fetch_stock_data(self) -> pd.DataFrame:
        """获取股票数据 - 优先使用baostock，失败则使用本地缓存

        Returns:
            pd.DataFrame: 处理后的股票数据

        Raises:
            ValueError: 获取数据失败
        """
        print("\n" + "=" * 48)
        print(f"开始获取 {self.config.stock_name}({self.config.stock_code}) 数据")
        print("=" * 48)

        # 动态设置 end_date，确保尝试获取最新数据
        self._adjust_end_date_for_fresh_data()

        cache_file = self.config.cache_file_path

        # 策略1: 尝试从baostock获取（带重试）
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"\n尝试从baostock获取数据 (尝试 {attempt + 1}/{max_retries})...")
                df = self._fetch_from_baostock_with_retry()

                if df is not None and len(df) > 0:
                    if self._validate_data(df, source='baostock'):
                        print(f"✓ 成功从baostock获取 {len(df)} 个交易日真实数据")
                        print(f"  数据时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")
                        print(f"  最新价格: {df['close'].iloc[-1]:.2f}元")

                        # 检查数据的新鲜度
                        self._check_data_freshness(df)

                        # 保存到缓存
                        self._save_to_cache(df, cache_file)
                        return df
                    else:
                        print(f"⚠ baostock数据验证失败，尝试 {attempt + 1}/{max_retries}")
            except Exception as e:
                print(f"baostock获取失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试

        # 策略2: 检查本地缓存
        print("\nbaostock获取失败，检查本地缓存...")
        df = self._try_load_from_cache(cache_file)
        if df is not None:
            return df

        # 所有方法都失败
        error_msg = f"无法获取股票 {self.config.stock_name}({self.config.stock_code}) 数据"
        error_msg += "\n可能的原因："
        error_msg += "\n1. 网络连接问题"
        error_msg += "\n2. 股票代码错误或停牌"
        error_msg += "\n3. 数据源暂时不可用"
        error_msg += f"\n缓存文件位置: {cache_file}"
        raise ValueError(error_msg)

    def _adjust_end_date_for_fresh_data(self):
        """调整 end_date 以确保获取最新数据"""
        try:
            from datetime import datetime, timedelta

            today = datetime.now()
            today_str = today.strftime('%Y-%m-%d')

            # 如果 end_date 不存在，设置为今天
            if not hasattr(self.config, 'end_date'):
                self.config.end_date = today_str
                print(f"  设置数据截止日期为: {today_str}")
                return

            # 如果 end_date 早于昨天，则更新为今天
            config_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
            yesterday = today - timedelta(days=1)

            if config_date < yesterday:
                print(f"  更新数据截止日期: {self.config.end_date} -> {today_str}")
                self.config.end_date = today_str
            else:
                print(f"  使用配置的截止日期: {self.config.end_date}")

        except Exception as e:
            print(f"  日期调整警告: {e}")
            # 确保有 end_date
            if not hasattr(self.config, 'end_date'):
                self.config.end_date = datetime.now().strftime('%Y-%m-%d')

    def _check_data_freshness(self, df):
        """检查数据的新鲜度"""
        if df.empty:
            return

        latest_date = df.index[-1]
        today = datetime.now().date()

        # 计算数据最后日期与今天的差距
        days_diff = (today - latest_date.date()).days

        if days_diff == 0:
            print(f"  ✓ 数据是最新的（今天的数据）")
        elif days_diff == 1:
            print(f"  ✓ 数据是新鲜的（昨天的数据）")
        elif days_diff <= 3:
            print(f"  ⚠ 数据较新，但不是最新（{days_diff} 天前）")
        elif days_diff <= 7:
            print(f"  ⚠ 数据已有一周未更新（{days_diff} 天前）")
        else:
            print(f"  ⚠ 数据较旧（{days_diff} 天前），可能需要更新")

    def _fetch_from_baostock_with_retry(self, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """从baostock获取数据，带重试机制

        Args:
            max_retries: 最大重试次数

        Returns:
            DataFrame 或 None
        """
        for attempt in range(max_retries):
            try:
                df = self._fetch_from_baostock()
                if df is not None and len(df) > 100:  # 确保数据量足够
                    return df
                elif df is not None and len(df) <= 100:
                    print(f"⚠ 获取的数据量较少，仅 {len(df)} 条记录")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  重试中... ({attempt + 1}/{max_retries})")
                    time.sleep(1)  # 等待1秒
                else:
                    raise e
        return None

    def _fetch_from_baostock(self) -> Optional[pd.DataFrame]:
        """从baostock获取数据"""
        try:
            import baostock as bs

            # 登录baostock
            if not self._baostock_logged_in:
                lg = bs.login()
                if lg.error_code != '0':
                    raise Exception(f"baostock登录失败: {lg.error_msg}")
                self._baostock_logged_in = True
                print("  ✓ baostock登录成功")

            # 设置股票代码格式
            stock_code = self.config.stock_code
            bs_code = self._format_stock_code_for_baostock(stock_code)

            print(f"  股票代码: {bs_code} ({self.config.stock_name})")
            print(f"  时间范围: {self.config.start_date} 至 {self.config.end_date}")

            # 查询历史K线数据
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg",
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                frequency="d",
                adjustflag="3"  # 前复权
            )

            if rs.error_code != '0':
                raise Exception(f"查询数据失败: {rs.error_msg}")

            # 获取数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                raise Exception("未获取到任何数据")

            print(f"  获取到 {len(data_list)} 条原始记录")

            # 转换为DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)

            # 数据清洗和验证
            df = self._clean_baostock_data(df)

            # 确保注销
            self._safe_logout_baostock()

            return df

        except ImportError:
            raise ImportError("请安装baostock库: pip install baostock")
        except Exception as e:
            # 发生错误时尝试注销
            self._safe_logout_baostock()
            raise e

    def _safe_logout_baostock(self):
        """安全注销baostock"""
        if self._baostock_logged_in:
            try:
                import baostock as bs
                bs.logout()
                self._baostock_logged_in = False
                # print("  ✓ baostock已注销")
            except:
                pass  # 忽略注销错误

    def _format_stock_code_for_baostock(self, stock_code: str) -> str:
        """格式化股票代码为baostock格式"""
        stock_code = str(stock_code).strip()

        # 移除可能的交易所前缀
        if stock_code.startswith(('sh', 'sz')):
            return stock_code

        # 根据代码规则判断
        if stock_code.startswith('6'):
            return f"sh.{stock_code}"
        elif stock_code.startswith(('0', '3')):
            return f"sz.{stock_code}"
        else:
            print(f"⚠ 无法识别的股票代码格式: {stock_code}，默认使用深市")
            return f"sz.{stock_code}"

    def _clean_baostock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗baostock数据"""
        if df.empty:
            return df

        # 转换日期
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df.set_index('date', inplace=True)

        # 过滤掉非交易日数据
        if 'tradestatus' in df.columns:
            df = df[df['tradestatus'] == '1']  # 只保留交易状态为1的数据

        # 转换数据类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 只保留需要的列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        optional_cols = ['pctChg']  # 可选列

        available_cols = [col for col in required_cols if col in df.columns]
        for col in optional_cols:
            if col in df.columns:
                available_cols.append(col)

        df = df[available_cols]

        # 删除缺失值
        df = df.dropna()

        # 按日期排序
        df = df.sort_index()

        # 确保成交量是整数
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype(int)

        # 确保价格合理
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]

        # 移除异常值（价格<=0或异常高）
        for col in price_cols:
            if col in df.columns:
                # 移除价格为0或负数的数据
                df = df[df[col] > 0]
                # 移除价格异常高的数据（假设单股价格不超过10000元）
                df = df[df[col] < 10000]

        # 确保价格关系正确
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            df = df[df['high'] >= df['low']]
            df['high'] = np.maximum(df['high'], df[['open', 'close']].max(axis=1))
            df['low'] = np.minimum(df['low'], df[['open', 'close']].min(axis=1))

        # 移除重复的日期
        df = df[~df.index.duplicated(keep='last')]

        return df

    def _validate_data(self, df: pd.DataFrame, source: str = 'unknown') -> bool:
        """验证数据质量

        Args:
            df: 待验证的DataFrame
            source: 数据来源

        Returns:
            bool: 数据是否有效
        """
        if df.empty:
            print(f"  ⚠ {source}数据为空")
            return False

        # 检查数据量
        if len(df) < 100:
            print(f"  ⚠ {source}数据量不足，仅 {len(df)} 条记录")
            return False

        # 检查缺失值
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"  ⚠ {source}数据包含缺失值: {missing_cols}")
            return False

        # 检查价格合理性
        if 'close' in df.columns:
            if df['close'].max() / df['close'].min() > 100:  # 价格波动过大
                print(f"  ⚠ {source}价格波动异常")
                return False

        # 检查日期连续性
        date_diff = (df.index[1:] - df.index[:-1]).days
        if date_diff.max() > 30:  # 有超过30天的间隔
            print(f"  ⚠ {source}日期不连续，最大间隔 {date_diff.max()} 天")

        print(f"  ✓ {source}数据验证通过")
        return True

    def _save_to_cache(self, df: pd.DataFrame, cache_file: str) -> None:
        """保存数据到缓存文件"""
        try:
            # 创建目录（如果不存在）
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            # 保存数据
            df.to_csv(cache_file)
            print(f"  ✓ 数据已保存到缓存: {cache_file}")
            print(f"    文件大小: {os.path.getsize(cache_file) / 1024:.1f} KB")

            # 不再生成元数据文件
            print(f"  ✓ 数据保存完成")

        except Exception as e:
            print(f"  ⚠ 保存缓存失败: {e}")

    def _try_load_from_cache(self, cache_file: str) -> Optional[pd.DataFrame]:
        """尝试从缓存加载数据"""
        if not os.path.exists(cache_file):
            print(f"  ✗ 缓存文件不存在: {cache_file}")
            return None

        try:
            # 检查缓存文件时间
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            file_age = (datetime.now() - file_time).days

            if file_age > 7:  # 缓存超过7天
                print(f"  ⚠ 缓存文件已过期 ({file_age} 天前)")

            print(f"找到缓存文件，更新时间: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # 读取数据
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

            if df.empty:
                print(f"  ✗ 缓存文件为空")
                return None

            # 验证缓存数据
            if self._validate_data(df, source='cache'):
                print(f"✓ 从缓存读取到 {len(df)} 个交易日数据")
                print(f"  数据时间范围: {df.index[0].date()} 至 {df.index[-1].date()}")
                print(f"  最新价格: {df['close'].iloc[-1]:.2f}元")
                return df
            else:
                print(f"  ✗ 缓存数据验证失败")
                return None

        except Exception as e:
            print(f"读取缓存失败: {e}")
            return None

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        if df.empty:
            return {}

        info = {
            '数据来源': '缓存' if 'cache' in str(self.config.cache_file_path) else 'baostock',
            '数据条数': len(df),
            '时间范围': f"{df.index[0].date()} 至 {df.index[-1].date()}",
            '交易天数': len(df),
            '最新价格': float(df['close'].iloc[-1]),
            '价格统计': {
                '最高价': float(df['high'].max()),
                '最低价': float(df['low'].min()),
                '平均价': float(df['close'].mean()),
            },
            '成交量统计': {
                '最大成交量': int(df['volume'].max()),
                '平均成交量': int(df['volume'].mean()),
            }
        }

        if 'pctChg' in df.columns:
            info['涨跌幅统计'] = {
                '最大涨幅': float(df['pctChg'].max()),
                '最大跌幅': float(df['pctChg'].min()),
                '平均涨幅': float(df['pctChg'].mean()),
            }

        return info