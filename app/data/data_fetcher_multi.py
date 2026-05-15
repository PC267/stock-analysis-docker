# data_fetcher_multi.py
# 多数据源股票数据获取器
# 优先级: baostock > akshare > 本地缓存

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Optional, Dict, Any, List

warnings.filterwarnings('ignore')


class DataSourceResult:
    """数据源结果封装"""
    def __init__(self, df: pd.DataFrame, source: str, rows: int, date_range: str, fresh: bool):
        self.df = df
        self.source = source
        self.rows = rows
        self.date_range = date_range
        self.fresh = fresh
        self.price = float(df['close'].iloc[-1]) if not df.empty and 'close' in df.columns else 0

    def __repr__(self):
        return f"<DataSourceResult [{self.source}] {self.rows}行 {self.date_range} 最新价:{self.price}>"


class MultiSourceDataFetcher:
    """
    多数据源股票数据获取器
    优先使用 baostock，失败则降级到 akshare，最后用本地缓存
    """

    # 数据源优先级（按顺序尝试）
    SOURCE_PRIORITY = ['baostock', 'akshare', 'cache']

    def __init__(self, config):
        if config is None:
            raise ValueError("Config不能为None")
        if not hasattr(config, 'stock_code') or not hasattr(config, 'stock_name'):
            raise ValueError("Config缺少必要的股票信息属性")

        self.config = config
        self.stock_code = config.stock_code
        self.stock_name = config.stock_name
        self.start_date = getattr(config, 'start_date', None)
        self.end_date = getattr(config, 'end_date', None)
        self.cache_file = getattr(config, 'cache_file_path', None)

        # 动态设置 end_date 确保获取最新数据
        self._adjust_end_date()

        # 记录各数据源状态
        self.source_status = {}
        self.failed_sources = []

        print(f"\n{'='*50}")
        print(f"  多数据源数据获取器初始化")
        print(f"  股票: {self.stock_name}({self.stock_code})")
        print(f"  目标日期: {self.start_date} ~ {self.end_date}")
        print(f"{'='*50}")

    def _adjust_end_date(self):
        """调整 end_date 以确保获取最新数据"""
        today = datetime.now().strftime('%Y-%m-%d')
        if self.end_date:
            config_date = datetime.strptime(self.end_date, '%Y-%m-%d')
            yesterday = datetime.now() - timedelta(days=1)
            if config_date < yesterday:
                print(f"  [日期调整] {self.end_date} -> {today}")
                self.end_date = today
        else:
            self.end_date = today

    # ========== 数据源1: baostock ==========

    def _fetch_from_baostock(self) -> Optional[pd.DataFrame]:
        """从 baostock 获取数据"""
        source = 'baostock'
        print(f"\n  >> 尝试数据源: {source}")

        try:
            import baostock as bs

            lg = bs.login()
            if lg.error_code != '0':
                print(f"  !! baostock 登录失败: {lg.error_msg}")
                return None

            bs_code = self._format_code_baostock(self.stock_code)

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,tradestatus,pctChg",
                start_date=self.start_date,
                end_date=self.end_date,
                frequency="d",
                adjustflag="3"
            )

            if rs.error_code != '0':
                print(f"  !! 查询失败: {rs.error_msg}")
                bs.logout()
                return None

            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())

            bs.logout()

            if not data_list:
                print(f"  !! 未获取到任何数据")
                return None

            df = pd.DataFrame(data_list, columns=rs.fields)
            df = self._clean_baostock_data(df)

            if df is None or len(df) < 100:
                print(f"  !! 数据量不足: {len(df) if df is not None else 0} 行")
                return None

            latest = df.index[-1].date()
            today = datetime.now().date()
            fresh = (today - latest).days <= 3

            print(f"  [OK] {source} 成功, {len(df)} 行, 范围 {df.index[0].date()} ~ {latest}, 新鲜度: {'✓' if fresh else '⚠'+str((today-latest).days)+'天前'}")
            return df

        except ImportError:
            print(f"  !! baostock 未安装")
            return None
        except Exception as e:
            print(f"  !! {source} 异常: {str(e)[:60]}")
            return None

    def _format_code_baostock(self, code: str) -> str:
        """格式化股票代码为 baostock 格式"""
        code = str(code).strip()
        if code.startswith(('sh.', 'sz.')):
            return code
        if code.startswith('6'):
            return f"sh.{code}"
        return f"sz.{code}"

    def _clean_baostock_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """清洗 baostock 数据"""
        if df.empty:
            return None
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)

            if 'tradestatus' in df.columns:
                df = df[df['tradestatus'] == '1']

            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'pctChg']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            required = ['open', 'high', 'low', 'close', 'volume']
            cols = [c for c in required if c in df.columns]
            if 'pctChg' in df.columns:
                cols.append('pctChg')
            df = df[cols].dropna()
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]

            # 价格合理性
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df = df[df[col] > 0]
                    df = df[df[col] < 10000]

            return df
        except Exception as e:
            print(f"  !! 数据清洗失败: {e}")
            return None

    # ========== 数据源2: akshare ==========

    def _fetch_from_akshare(self) -> Optional[pd.DataFrame]:
        """从 akshare 获取数据（baostock 失败时的备用）"""
        source = 'akshare'
        print(f"\n  >> 尝试数据源: {source}")

        try:
            import akshare as ak

            # 格式化代码: 000001 -> 000001.XSHE (深市) 或 000001.XSHG (沪市)
            code = str(self.stock_code).strip()
            if code.startswith('6'):
                symbol = f"{code}.XSHG"
            else:
                symbol = f"{code}.XSHE"

            df = ak.stock_zh_a_hist(
                symbol=code,
                period='daily',
                start_date=self.start_date.replace('-', ''),
                end_date=self.end_date.replace('-', ''),
                adjust='qfq'
            )

            if df is None or len(df) < 100:
                print(f"  !! 数据量不足: {len(df) if df is not None else 0} 行")
                return None

            # 重命名字段以匹配 baostock 格式
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '涨跌幅': 'pctChg'
            })

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            df.set_index('date', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume', 'pctChg']]
            df = df.dropna()
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]

            latest = df.index[-1].date()
            today = datetime.now().date()
            fresh = (today - latest).days <= 3

            print(f"  [OK] {source} 成功, {len(df)} 行, 范围 {df.index[0].date()} ~ {latest}, 新鲜度: {'✓' if fresh else '⚠'+str((today-latest).days)+'天前'}")
            return df

        except ImportError:
            print(f"  !! akshare 未安装")
            return None
        except Exception as e:
            print(f"  !! {source} 异常: {str(e)[:80]}")
            return None

    # ========== 数据源3: 本地缓存 ==========

    # 数据缓存 TTL: 7天（股票数据以日为单位，缓存一天足够了）
    CACHE_TTL_DAYS = 7

    def _fetch_from_cache(self) -> Optional[pd.DataFrame]:
        """从本地缓存加载数据（带TTL验证）"""
        source = 'cache'
        print(f"\n  >> 尝试数据源: {source}")

        cache_file = self.cache_file
        if not cache_file:
            # 尝试默认位置
            cache_file = f"/tmp/stock_data/{self.stock_code}_baostock.csv"

        if not os.path.exists(cache_file):
            print(f"  !! 缓存文件不存在: {cache_file}")
            return None

        # 检查缓存是否过期
        meta_file = cache_file + ".meta"
        cache_valid = False
        if os.path.exists(meta_file):
            try:
                import json
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                saved_at = datetime.fromisoformat(meta['saved_at'])
                age_days = (datetime.now() - saved_at).total_seconds() / 86400
                cache_valid = age_days < self.CACHE_TTL_DAYS
                if not cache_valid:
                    print(f"  !! 缓存已过期（{age_days:.1f}天前 > {self.CACHE_TTL_DAYS}天）")
            except Exception:
                pass

        if not cache_valid:
            # 尝试降级使用：文件存在但过期，看数据本身是否足够新
            pass  # 继续尝试读取，数据新鲜度由数据内日期判断

        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if df.empty:
                print(f"  !! 缓存为空")
                return None

            latest = df.index[-1].date()
            today = datetime.now().date()
            age = (today - latest).days
            data_fresh = age <= self.CACHE_TTL_DAYS

            print(f"  [OK] {source} 成功, {len(df)} 行, {df.index[0].date()} ~ {latest}, "
                  f"数据{data_fresh}天前 {'✓' if data_fresh else '⚠过期'} "
                  f"[meta:{'有效' if cache_valid else '缺失'}]")
            return df

        except Exception as e:
            print(f"  !! {source} 读取失败: {e}")
            return None

    def _validate_data(self, df: pd.DataFrame, source: str = 'unknown') -> bool:
        """验证数据质量"""
        if df.empty:
            return False
        if len(df) < 100:
            return False
        if df.isnull().any().any():
            return False
        if 'close' in df.columns:
            if df['close'].max() / df['close'].min() > 100:
                return False
        return True

    # ========== 主获取方法 ==========

    def fetch(self) -> pd.DataFrame:
        """
        获取股票数据，按优先级尝试各数据源

        Returns:
            pd.DataFrame: 清洗后的股票数据

        Raises:
            ValueError: 所有数据源都失败
        """
        results: Dict[str, DataSourceResult] = {}

        # 1. 尝试 baostock
        df = self._fetch_from_baostock()
        if df is not None and self._validate_data(df, 'baostock'):
            self.source_status['baostock'] = 'SUCCESS'
            return df
        else:
            self.source_status['baostock'] = 'FAILED'
            self.failed_sources.append('baostock')

        # 2. 尝试 akshare
        df = self._fetch_from_akshare()
        if df is not None and self._validate_data(df, 'akshare'):
            self.source_status['akshare'] = 'SUCCESS'
            # 自动保存为缓存
            self._save_cache(df)
            return df
        else:
            self.source_status['akshare'] = 'FAILED'
            self.failed_sources.append('akshare')

        # 3. 尝试本地缓存
        df = self._fetch_from_cache()
        if df is not None and self._validate_data(df, 'cache'):
            self.source_status['cache'] = 'SUCCESS'
            return df
        else:
            self.source_status['cache'] = 'FAILED'
            self.failed_sources.append('cache')

        # 所有数据源都失败
        raise ValueError(
            f"无法获取股票 {self.stock_name}({self.stock_code}) 数据。"
            f"尝试过的数据源: {', '.join(self.failed_sources)}。"
            f"请检查网络连接或更新本地缓存。"
        )

    def _save_cache(self, df: pd.DataFrame):
        """保存数据到缓存（同时保存元数据）"""
        if not self.cache_file:
            return
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            df.to_csv(self.cache_file)
            # 保存元数据（保存时间 + 数据摘要）
            import json
            meta_file = self.cache_file + ".meta"
            meta = {
                'saved_at': datetime.now().isoformat(),
                'stock_code': self.stock_code,
                'stock_name': self.stock_name,
                'rows': len(df),
                'date_range': f"{df.index[0].date()} ~ {df.index[-1].date()}",
                'latest_close': float(df['close'].iloc[-1]),
            }
            with open(meta_file, 'w') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"  [缓存] 已保存: {self.cache_file} + {meta_file}")
        except Exception as e:
            print(f"  [缓存] 保存失败: {e}")

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据统计信息"""
        if df.empty:
            return {}

        primary_source = self.failed_sources[0] if self.failed_sources else 'baostock'

        info = {
            '数据来源': primary_source if self.source_status.get('baostock') == 'SUCCESS' else
                       'akshare' if self.source_status.get('akshare') == 'SUCCESS' else 'cache',
            '可用数据源': [s for s in self.SOURCE_PRIORITY if self.source_status.get(s) == 'SUCCESS'],
            '失败数据源': self.failed_sources,
            '数据条数': len(df),
            '时间范围': f"{df.index[0].date()} 至 {df.index[-1].date()}",
            '最新价格': float(df['close'].iloc[-1]),
        }

        if 'pctChg' in df.columns:
            info['涨跌幅'] = float(df['pctChg'].iloc[-1])

        return info
