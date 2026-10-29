import pandas as pd
import os


def analyze_stock_data():
    """
    分析股票数据并保存简化版结果到文件
    """
    # 读取CSV文件
    csv_file = "../results/eastmoney_stock_data.csv"

    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在！")
        return

    try:
        # 读取CSV文件时指定股票代码列为字符串类型，防止前面的0被删除
        df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'股票代码': str})
        print(f"成功读取数据，共 {len(df)} 条记录")

        # 检查必要的列是否存在 - 增加筛选所需的列
        required_columns = ['股票代码', '股票名称', '涨跌幅', '各项总和减主力流入净额',
                            '主力流入净额', '主力流入净额占比%', '超大单流入占比%', '涨跌幅%']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"错误：缺少必要的列: {missing_columns}")
            return

        # 数据处理：确保数值列是数值类型
        df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
        df['各项总和减主力流入净额'] = pd.to_numeric(df['各项总和减主力流入净额'], errors='coerce')
        df['主力流入净额'] = pd.to_numeric(df['主力流入净额'], errors='coerce')
        df['主力流入净额占比%'] = pd.to_numeric(df['主力流入净额占比%'], errors='coerce')
        df['超大单流入占比%'] = pd.to_numeric(df['超大单流入占比%'], errors='coerce')
        df['涨跌幅%'] = pd.to_numeric(df['涨跌幅%'], errors='coerce')

        # 确保股票代码保持字符串格式，保留前面的0
        df['股票代码'] = df['股票代码'].astype(str)
        df['股票代码'] = df['股票代码'].apply(lambda x: x.zfill(6) if isinstance(x, str) else str(x).zfill(6))

        # ==================== 优质股票筛选逻辑 ====================

        # 第一层：主力资金强势（核心指标）
        # 条件：主力净流入 > 1亿 且 占比 > 5%
        df_filtered_1 = df[
            (df['主力流入净额'] > 100000000) &
            (df['主力流入净额占比%'] > 5)
            ].copy()

        # 第二层：机构资金认可（超大单占比高）
        # 条件：超大单占比 > 3%
        df_filtered_2 = df_filtered_1[df_filtered_1['超大单流入占比%'] > 3].copy()

        # 第三层：股价表现良好
        # 条件：涨跌幅在0-10%之间（避免暴涨暴跌）
        df_filtered_3 = df_filtered_2[
            (df_filtered_2['涨跌幅%'] > 0) &
            (df_filtered_2['涨跌幅%'] < 10)
            ].copy()

        # 第四层：综合评分排序
        if len(df_filtered_3) > 0:
            # 使用过滤后的数据进行综合评分
            work_df = df_filtered_3.copy()
        else:
            # 如果没有符合前三层筛选的数据，使用原始数据
            print("警告：前三层筛选无结果，使用原始数据进行综合评分")
            work_df = df.copy()

        # 归一化函数
        def normalize_series(series):
            max_val = series.abs().max()
            if max_val == 0:
                return pd.Series([0] * len(series), index=series.index)
            return series / max_val

        # 归一化各个指标
        work_df['score_主力'] = normalize_series(work_df['主力流入净额'])
        work_df['score_主力占比'] = normalize_series(work_df['主力流入净额占比%'])
        work_df['score_超大单'] = normalize_series(work_df['超大单流入占比%'])
        work_df['score_涨幅'] = normalize_series(work_df['涨跌幅%'])

        # 计算综合评分（权重：主力资金40%，主力占比30%，超大单20%，涨幅10%）
        work_df['综合评分'] = (
                work_df['score_主力'] * 0.4 +
                work_df['score_主力占比'] * 0.3 +
                work_df['score_超大单'] * 0.2 +
                work_df['score_涨幅'] * 0.1
        )

        # 优先按综合评分排序，再按主力流入净额排序
        df_sorted = work_df.sort_values(
            by=['综合评分', '主力流入净额'],
            ascending=[False, False]  # 综合评分降序，主力净流入降序
        )

        print(f"\n筛选统计：")
        print(f"原始数据：{len(df)} 条")
        print(f"第一层筛选（主力流入>1亿且占比>5%）：{len(df_filtered_1)} 条")
        print(f"第二层筛选（超大单占比>3%）：{len(df_filtered_2)} 条")
        print(f"第三层筛选（涨幅0-10%）：{len(df_filtered_3)} 条")

        # 获取前20条数据
        top_10 = df_sorted.head(20)

        print(f"\n排序后的前20条数据:")
        print("=" * 80)
        for idx, row in top_10.iterrows():
            stock_code = row['股票代码']
            stock_name = row['股票名称']
            three_day_change = row['涨跌幅']
            diff_value = row['主力流入净额']

            # 显示更多信息
            if '综合评分' in row:
                score = row['综合评分']
                main_inflow = row['主力流入净额']
                main_ratio = row['主力流入净额占比%']
                super_ratio = row['超大单流入占比%']
                daily_change = row['涨跌幅%']

                print(f"{stock_code},{stock_name} (综合评分={score:.4f}, 主力流入={main_inflow:,.0f})")
                print(
                    f"  主力占比={main_ratio:.2f}%, 超大单占比={super_ratio:.2f}%, 日涨幅={daily_change:.2f}%, 三日涨幅={three_day_change:.2f}")
                print("-" * 40)
            else:
                print(f"{stock_code},{stock_name} (涨跌幅={three_day_change:.2f}, 差额={diff_value:,.0f})")

        # 保存简化版结果到文件
        output_file = "../results/Stock_Data.txt"

        # 确保results目录存在
        output_dir = "../results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n创建目录: {output_dir}")

        # 写入文件，只保存"股票代码,股票名称"格式，确保代码前面的0保留
        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in top_10.iterrows():
                stock_code = str(row['股票代码'])
                stock_name = str(row['股票名称'])
                # 确保股票代码有6位，不足的补0
                if len(stock_code) < 6:
                    stock_code = stock_code.zfill(6)
                f.write(f"{stock_code},{stock_name}\n")

        print(f"\n✓ 简化版结果已保存到: {output_file}")

        # 显示保存的文件内容
        print(f"\n保存的文件内容:")
        print("=" * 30)
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)

        return top_10

    except Exception as e:
        print(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    主函数
    """
    print("开始分析股票数据...")
    print("=" * 60)

    # 分析数据
    result = analyze_stock_data()

    if result is not None:
        print(f"\n✓ 成功处理 {len(result)} 条数据")

    print("\n" + "=" * 60)
    print("分析完成！")


if __name__ == "__main__":
    # 检查pandas是否安装
    try:
        import pandas as pd

        print("✓ pandas 已安装")
    except ImportError:
        print("错误: pandas 未安装")
        print("请运行: pip install pandas")
        exit(1)

    main()