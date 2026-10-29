# 市场环境特征
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

class MarketFeatureExtractor:
    """市场特征提取器"""

    def __init__(self, market_data: Optional[pd.DataFrame] = None,
                 macro_data: Optional[pd.DataFrame] = None):
        """
        初始化市场特征提取器

        Args:
            market_data: 市场数据DataFrame，包含价格、成交量等
            macro_data: 宏观经济数据DataFrame，包含各项经济指标
        """
        self.market_data = market_data
        self.macro_data = macro_data
        self.extracted_features = {}  # 存储提取的特征

    def extract_gdp_features(self) -> Dict[str, Optional[Union[pd.Series, float]]]:
        """
        提取GDP相关特征

        Returns:
            包含GDP特征的字典
        """
        if self.macro_data is None:
            return {}

        gdp_features = {}

        # 1. GDP增长率
        if 'gdp_growth_rate' in self.macro_data.columns:
            gdp_growth = self.macro_data['gdp_growth_rate']
            gdp_features['gdp_growth_rate'] = gdp_growth.iloc[-1] if len(gdp_growth) > 0 else None

            # 计算GDP增长趋势
            if len(gdp_growth) >= 4:
                gdp_features['gdp_growth_trend'] = gdp_growth.iloc[-4:].mean() - gdp_growth.iloc[-8:-4].mean()

        # 2. 名义GDP
        if 'nominal_gdp' in self.macro_data.columns:
            nominal_gdp = self.macro_data['nominal_gdp']
            gdp_features['nominal_gdp'] = nominal_gdp.iloc[-1] if len(nominal_gdp) > 0 else None

        # 3. 实际GDP
        if 'real_gdp' in self.macro_data.columns:
            real_gdp = self.macro_data['real_gdp']
            gdp_features['real_gdp'] = real_gdp.iloc[-1] if len(real_gdp) > 0 else None

        # 4. GDP缺口（实际GDP与潜在GDP的差距）
        if 'real_gdp' in self.macro_data.columns and 'potential_gdp' in self.macro_data.columns:
            gdp_gap = (self.macro_data['real_gdp'] - self.macro_data['potential_gdp']) / self.macro_data['potential_gdp'] * 100
            gdp_features['gdp_gap_percent'] = gdp_gap.iloc[-1] if len(gdp_gap) > 0 else None

        self.extracted_features['gdp_features'] = gdp_features
        return gdp_features

    def extract_cpi_features(self) -> Dict[str, Optional[Union[pd.Series, float]]]:
        """
        提取CPI（消费者价格指数）相关特征

        Returns:
            包含CPI特征的字典
        """
        if self.macro_data is None:
            return {}

        cpi_features = {}

        # 1. CPI同比
        if 'cpi_yoy' in self.macro_data.columns:
            cpi_yoy = self.macro_data['cpi_yoy']
            cpi_features['cpi_yoy'] = cpi_yoy.iloc[-1] if len(cpi_yoy) > 0 else None

            # 通胀水平分类
            last_cpi = cpi_yoy.iloc[-1] if len(cpi_yoy) > 0 else None
            if last_cpi is not None:
                if last_cpi < 2:
                    cpi_features['inflation_level'] = '低通胀'
                elif 2 <= last_cpi < 5:
                    cpi_features['inflation_level'] = '温和通胀'
                elif 5 <= last_cpi < 10:
                    cpi_features['inflation_level'] = '高通胀'
                else:
                    cpi_features['inflation_level'] = '恶性通胀'

        # 2. CPI环比
        if 'cpi_mom' in self.macro_data.columns:
            cpi_mom = self.macro_data['cpi_mom']
            cpi_features['cpi_mom'] = cpi_mom.iloc[-1] if len(cpi_mom) > 0 else None

        # 3. 核心CPI（排除食品和能源）
        if 'core_cpi_yoy' in self.macro_data.columns:
            core_cpi = self.macro_data['core_cpi_yoy']
            cpi_features['core_cpi_yoy'] = core_cpi.iloc[-1] if len(core_cpi) > 0 else None

        # 4. 通胀预期
        if 'inflation_expectation' in self.macro_data.columns:
            inflation_exp = self.macro_data['inflation_expectation']
            cpi_features['inflation_expectation'] = inflation_exp.iloc[-1] if len(inflation_exp) > 0 else None

        # 5. 计算实际利率（名义利率 - 通胀率）
        if 'cpi_yoy' in self.macro_data.columns and 'policy_rate' in self.macro_data.columns:
            real_rate = self.macro_data['policy_rate'] - self.macro_data['cpi_yoy']
            cpi_features['real_interest_rate'] = real_rate.iloc[-1] if len(real_rate) > 0 else None

        self.extracted_features['cpi_features'] = cpi_features
        return cpi_features

    def extract_unemployment_features(self) -> Dict[str, Optional[Union[pd.Series, float]]]:
        """
        提取失业率相关特征

        Returns:
            包含失业率特征的字典
        """
        if self.macro_data is None:
            return {}

        unemployment_features = {}

        # 1. 整体失业率
        if 'unemployment_rate' in self.macro_data.columns:
            unemployment = self.macro_data['unemployment_rate']
            unemployment_features['unemployment_rate'] = unemployment.iloc[-1] if len(unemployment) > 0 else None

            # 失业率趋势
            if len(unemployment) >= 4:
                unemployment_features['unemployment_trend'] = unemployment.iloc[-4:].mean() - unemployment.iloc[-8:-4].mean()

        # 2. 青年失业率
        if 'youth_unemployment_rate' in self.macro_data.columns:
            youth_unemployment = self.macro_data['youth_unemployment_rate']
            unemployment_features['youth_unemployment_rate'] = youth_unemployment.iloc[-1] if len(youth_unemployment) > 0 else None

        # 3. 自然失业率
        if 'natural_unemployment_rate' in self.macro_data.columns:
            natural_unemployment = self.macro_data['natural_unemployment_rate']
            unemployment_features['natural_unemployment_rate'] = natural_unemployment.iloc[-1] if len(natural_unemployment) > 0 else None

            # 计算失业缺口
            if 'unemployment_rate' in self.macro_data.columns:
                unemployment_gap = self.macro_data['unemployment_rate'] - self.macro_data['natural_unemployment_rate']
                unemployment_features['unemployment_gap'] = unemployment_gap.iloc[-1] if len(unemployment_gap) > 0 else None

        # 4. 就业人口变化
        if 'employment_change' in self.macro_data.columns:
            employment_change = self.macro_data['employment_change']
            unemployment_features['employment_change'] = employment_change.iloc[-1] if len(employment_change) > 0 else None

        # 5. 劳动参与率
        if 'labor_participation_rate' in self.macro_data.columns:
            participation_rate = self.macro_data['labor_participation_rate']
            unemployment_features['labor_participation_rate'] = participation_rate.iloc[-1] if len(participation_rate) > 0 else None

        self.extracted_features['unemployment_features'] = unemployment_features
        return unemployment_features

    def extract_interest_rate_features(self) -> Dict[str, Optional[Union[pd.Series, float]]]:
        """
        提取利率相关特征

        Returns:
            包含利率特征的字典
        """
        if self.macro_data is None:
            return {}

        interest_features = {}

        # 1. 政策利率（央行基准利率）
        if 'policy_rate' in self.macro_data.columns:
            policy_rate = self.macro_data['policy_rate']
            interest_features['policy_rate'] = policy_rate.iloc[-1] if len(policy_rate) > 0 else None

            # 利率变化方向
            if len(policy_rate) >= 2:
                rate_change = policy_rate.iloc[-1] - policy_rate.iloc[-2]
                interest_features['policy_rate_change'] = rate_change
                interest_features['policy_rate_direction'] = '加息' if rate_change > 0 else '降息' if rate_change < 0 else '不变'

        # 2. 国债收益率
        treasury_yields = {}
        for maturity in ['3m', '6m', '1y', '2y', '5y', '10y']:
            col_name = f'treasury_yield_{maturity}'
            if col_name in self.macro_data.columns:
                yield_data = self.macro_data[col_name]
                treasury_yields[maturity] = yield_data.iloc[-1] if len(yield_data) > 0 else None

        if treasury_yields:
            interest_features['treasury_yields'] = treasury_yields

            # 收益率曲线（10年-2年利差）
            if '10y' in treasury_yields and '2y' in treasury_yields:
                if treasury_yields['10y'] is not None and treasury_yields['2y'] is not None:
                    interest_features['yield_curve_spread'] = treasury_yields['10y'] - treasury_yields['2y']

                    # 收益率曲线形态判断
                    spread = interest_features['yield_curve_spread']
                    if spread > 0:
                        interest_features['yield_curve_shape'] = '正常（陡峭）'
                    elif spread < 0:
                        interest_features['yield_curve_shape'] = '倒挂（预示衰退）'
                    else:
                        interest_features['yield_curve_shape'] = '平坦'

        # 3. 市场利率（LIBOR/SHIBOR等）
        if 'market_rate' in self.macro_data.columns:
            market_rate = self.macro_data['market_rate']
            interest_features['market_rate'] = market_rate.iloc[-1] if len(market_rate) > 0 else None

        # 4. 存款准备金率
        if 'reserve_requirement_ratio' in self.macro_data.columns:
            rrr = self.macro_data['reserve_requirement_ratio']
            interest_features['reserve_requirement_ratio'] = rrr.iloc[-1] if len(rrr) > 0 else None

        # 5. 实际利率（已在上面的CPI函数中计算）
        # 如果这里需要也可以独立计算

        self.extracted_features['interest_rate_features'] = interest_features
        return interest_features

    def extract_additional_macro_features(self) -> Dict[str, Dict]:
        """
        提取其他重要的宏观经济特征

        Returns:
            包含其他宏观经济特征的字典
        """
        if self.macro_data is None:
            return {}

        additional_features = {}

        # 1. PMI（采购经理人指数）
        pmi_features = {}
        for pmi_type in ['manufacturing_pmi', 'services_pmi', 'composite_pmi']:
            if pmi_type in self.macro_data.columns:
                pmi_data = self.macro_data[pmi_type]
                pmi_features[pmi_type] = pmi_data.iloc[-1] if len(pmi_data) > 0 else None

                # PMI荣枯线判断
                if pmi_features[pmi_type] is not None:
                    pmi_features[f'{pmi_type}_status'] = '扩张' if pmi_features[pmi_type] > 50 else '收缩'

        if pmi_features:
            additional_features['pmi_features'] = pmi_features

        # 2. 货币供应量
        money_supply_features = {}
        for m_type in ['m1_growth', 'm2_growth', 'total_social_financing']:
            if m_type in self.macro_data.columns:
                money_data = self.macro_data[m_type]
                money_supply_features[m_type] = money_data.iloc[-1] if len(money_data) > 0 else None

        if money_supply_features:
            additional_features['money_supply_features'] = money_supply_features

        # 3. 贸易数据
        trade_features = {}
        for trade_item in ['trade_balance', 'exports_growth', 'imports_growth']:
            if trade_item in self.macro_data.columns:
                trade_data = self.macro_data[trade_item]
                trade_features[trade_item] = trade_data.iloc[-1] if len(trade_data) > 0 else None

        if trade_features:
            additional_features['trade_features'] = trade_features

        # 4. 财政指标
        fiscal_features = {}
        for fiscal_item in ['fiscal_deficit_ratio', 'government_debt_gdp_ratio']:
            if fiscal_item in self.macro_data.columns:
                fiscal_data = self.macro_data[fiscal_item]
                fiscal_features[fiscal_item] = fiscal_data.iloc[-1] if len(fiscal_data) > 0 else None

        if fiscal_features:
            additional_features['fiscal_features'] = fiscal_features

        # 5. 汇率
        if 'exchange_rate' in self.macro_data.columns:
            exchange_rate = self.macro_data['exchange_rate']
            additional_features['exchange_rate'] = exchange_rate.iloc[-1] if len(exchange_rate) > 0 else None

        self.extracted_features['additional_macro_features'] = additional_features
        return additional_features

    def extract_macro_indicators_summary(self) -> pd.DataFrame:
        """
        提取所有宏观经济指标摘要

        Returns:
            包含所有宏观经济指标的DataFrame
        """
        # 提取所有宏观经济特征
        gdp_features = self.extract_gdp_features()
        cpi_features = self.extract_cpi_features()
        unemployment_features = self.extract_unemployment_features()
        interest_features = self.extract_interest_rate_features()
        additional_features = self.extract_additional_macro_features()

        # 创建摘要字典
        summary_dict = {}

        # 合并所有特征
        all_features = {
            **gdp_features,
            **cpi_features,
            **unemployment_features,
            **interest_features
        }

        # 添加嵌套字典的特征
        if 'additional_macro_features' in self.extracted_features:
            for category, features in self.extracted_features['additional_macro_features'].items():
                if isinstance(features, dict):
                    for key, value in features.items():
                        all_features[f"{category}_{key}"] = value
                else:
                    all_features[category] = features

        # 创建DataFrame
        summary_df = pd.DataFrame([all_features])

        # 计算宏观经济景气指数（简化版）
        macro_score = self._calculate_macro_score(all_features)
        summary_df['macro_score'] = macro_score

        return summary_df

    def _calculate_macro_score(self, features: Dict) -> float:
        """
        计算宏观经济综合评分

        Args:
            features: 宏观经济特征字典

        Returns:
            宏观经济评分（0-100）
        """
        score = 50  # 基准分

        # 根据GDP增长评分
        if 'gdp_growth_rate' in features and features['gdp_growth_rate'] is not None:
            gdp_growth = features['gdp_growth_rate']
            if gdp_growth > 6:
                score += 15
            elif gdp_growth > 4:
                score += 10
            elif gdp_growth > 2:
                score += 5
            elif gdp_growth <= 0:
                score -= 10

        # 根据通胀评分
        if 'cpi_yoy' in features and features['cpi_yoy'] is not None:
            cpi = features['cpi_yoy']
            if 1.5 <= cpi <= 3.5:  # 理想通胀区间
                score += 10
            elif cpi < 0:  # 通缩
                score -= 10
            elif cpi > 5:  # 高通胀
                score -= 15

        # 根据失业率评分
        if 'unemployment_rate' in features and features['unemployment_rate'] is not None:
            unemployment = features['unemployment_rate']
            if unemployment < 4:
                score += 10
            elif unemployment < 6:
                score += 5
            elif unemployment > 8:
                score -= 10

        # 确保分数在0-100之间
        return max(0, min(100, score))

    def get_macro_health_status(self) -> Dict[str, str]:
        """
        获取宏观经济健康状况评估

        Returns:
            包含各项评估的字典
        """
        summary = self.extract_macro_indicators_summary()
        if summary.empty:
            return {}

        health_status = {}

        # GDP增长评估
        gdp_growth = summary.get('gdp_growth_rate', [None])[0]
        if gdp_growth is not None:
            if gdp_growth > 6:
                health_status['gdp'] = '强劲增长'
            elif gdp_growth > 4:
                health_status['gdp'] = '稳定增长'
            elif gdp_growth > 2:
                health_status['gdp'] = '缓慢增长'
            elif gdp_growth > 0:
                health_status['gdp'] = '接近停滞'
            else:
                health_status['gdp'] = '衰退'

        # 通胀评估
        cpi = summary.get('cpi_yoy', [None])[0]
        if cpi is not None:
            if 1.5 <= cpi <= 3.5:
                health_status['inflation'] = '健康'
            elif cpi < 1.5:
                health_status['inflation'] = '偏低'
            elif cpi > 3.5:
                health_status['inflation'] = '偏高'
            if cpi < 0:
                health_status['inflation'] = '通缩'
            if cpi > 5:
                health_status['inflation'] = '高通胀'

        # 失业率评估
        unemployment = summary.get('unemployment_rate', [None])[0]
        if unemployment is not None:
            if unemployment < 4:
                health_status['employment'] = '充分就业'
            elif unemployment < 6:
                health_status['employment'] = '正常'
            elif unemployment < 8:
                health_status['employment'] = '偏高'
            else:
                health_status['employment'] = '高失业'

        # 总体评估
        macro_score = summary.get('macro_score', [50])[0]
        if macro_score >= 70:
            health_status['overall'] = '健康'
        elif macro_score >= 50:
            health_status['overall'] = '稳定'
        elif macro_score >= 30:
            health_status['overall'] = '疲弱'
        else:
            health_status['overall'] = '衰退'

        return health_status

# 使用示例
if __name__ == "__main__":
    # 创建示例宏观经济数据
    macro_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=12, freq='ME'),
        'gdp_growth_rate': [4.5, 4.7, 4.6, 4.8, 5.0, 5.2, 5.1, 5.3, 5.2, 5.4, 5.3, 5.5],
        'cpi_yoy': [2.1, 2.3, 2.2, 2.4, 2.5, 2.6, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9],
        'core_cpi_yoy': [1.8, 1.9, 1.8, 2.0, 2.1, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5],
        'unemployment_rate': [5.2, 5.1, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1],
        'policy_rate': [3.5, 3.5, 3.75, 3.75, 4.0, 4.0, 4.25, 4.25, 4.5, 4.5, 4.75, 4.75],
        'treasury_yield_2y': [3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3],
        'treasury_yield_10y': [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6],
        'manufacturing_pmi': [51.2, 51.5, 51.8, 52.0, 52.3, 52.5, 52.7, 52.9, 53.1, 53.3, 53.5, 53.7],
        'm2_growth': [8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6],
        'trade_balance': [45, 47, 49, 50, 52, 53, 55, 56, 58, 59, 60, 62]
    })

    # 创建市场数据（简化）
    market_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=252, freq='D'),
        'close': np.random.normal(100, 10, 252).cumsum() + 1000,
        'volume': np.random.randint(1000000, 10000000, 252),
        'advances': np.random.randint(100, 300, 252),
        'declines': np.random.randint(100, 300, 252)
    })

    # 创建特征提取器
    extractor = MarketFeatureExtractor(market_data=market_data, macro_data=macro_data)

    # 提取GDP特征
    print("GDP特征:")
    gdp_features = extractor.extract_gdp_features()
    for key, value in gdp_features.items():
        print(f"  {key}: {value}")

    # 提取CPI特征
    print("\nCPI特征:")
    cpi_features = extractor.extract_cpi_features()
    for key, value in cpi_features.items():
        print(f"  {key}: {value}")

    # 提取失业率特征
    print("\n失业率特征:")
    unemployment_features = extractor.extract_unemployment_features()
    for key, value in unemployment_features.items():
        print(f"  {key}: {value}")

    # 提取利率特征
    print("\n利率特征:")
    interest_features = extractor.extract_interest_rate_features()
    for key, value in interest_features.items():
        print(f"  {key}: {value}")

    # 获取宏观经济摘要
    print("\n宏观经济摘要:")
    macro_summary = extractor.extract_macro_indicators_summary()
    print(macro_summary)

    # 获取宏观经济健康状况
    print("\n宏观经济健康状况:")
    health_status = extractor.get_macro_health_status()
    for key, value in health_status.items():
        print(f"  {key}: {value}")