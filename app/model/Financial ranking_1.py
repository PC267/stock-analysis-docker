"""
优化要点：
- 4 层 .copy() 合并为一次布尔掩码 + 单份工作副本
- 数值列一次性转换，不做重复 to_numeric
- 输出用写入流，逐行生成，不再构造大字符串
"""
import pandas as pd
import os
import gc


def analyze_stock_data():
    csv_file = "../results/eastmoney_stock_data.csv"
    if not os.path.exists(csv_file):
        print(f"错误：文件 {csv_file} 不存在！")
        return None

    try:
        df = pd.read_csv(csv_file, encoding="utf-8-sig", dtype={"股票代码": str})
        print(f"成功读取数据，共 {len(df)} 条记录")

        required_columns = [
            "股票代码", "股票名称", "涨跌幅", "各项总和减主力流入净额",
            "主力流入净额", "主力流入净额占比%", "超大单流入占比%", "涨跌幅%",
        ]
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            print(f"错误：缺少必要的列: {missing_columns}")
            return None

        # 数值列一次性转换
        numeric_cols = ["涨跌幅", "各项总和减主力流入净额", "主力流入净额",
                        "主力流入净额占比%", "超大单流入占比%", "涨跌幅%"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)

        # 三层筛选 -> 合并为布尔掩码
        mask1 = (df["主力流入净额"] > 100_000_000) & (df["主力流入净额占比%"] > 5)
        mask2 = mask1 & (df["超大单流入占比%"] > 3)
        mask3 = mask2 & (df["涨跌幅%"] > 0) & (df["涨跌幅%"] < 10)

        print("\n筛选统计：")
        print(f"原始数据：{len(df)} 条")
        print(f"第一层筛选（主力流入>1亿且占比>5%）：{int(mask1.sum())} 条")
        print(f"第二层筛选（超大单占比>3%）：{int(mask2.sum())} 条")
        print(f"第三层筛选（涨幅0-10%）：{int(mask3.sum())} 条")

        if mask3.any():
            work_df = df[mask3].copy()
        else:
            print("警告：前三层筛选无结果，使用原始数据进行综合评分")
            work_df = df.copy()

        def normalize_series(series):
            max_val = series.abs().max()
            if max_val == 0:
                return pd.Series(0.0, index=series.index)
            return series / max_val

        work_df["综合评分"] = (
            normalize_series(work_df["主力流入净额"]) * 0.4
            + normalize_series(work_df["主力流入净额占比%"]) * 0.3
            + normalize_series(work_df["超大单流入占比%"]) * 0.2
            + normalize_series(work_df["涨跌幅%"]) * 0.1
        )

        df_sorted = work_df.sort_values(
            by=["综合评分", "主力流入净额"], ascending=[False, False]
        )
        top_10 = df_sorted.head(20)

        print("\n排序后的前20条数据:")
        print("=" * 80)
        for _, row in top_10.iterrows():
            print(f"{row['股票代码']},{row['股票名称']} "
                  f"(综合评分={row['综合评分']:.4f}, "
                  f"主力流入={row['主力流入净额']:,.0f}, "
                  f"主力占比={row['主力流入净额占比%']:.2f}%, "
                  f"超大单占比={row['超大单流入占比%']:.2f}%, "
                  f"日涨幅={row['涨跌幅%']:.2f}%, "
                  f"三日涨幅={row['涨跌幅']:.2f})")

        # 保存精简版
        output_dir = "../results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "Stock_Data.txt")

        with open(output_file, "w", encoding="utf-8") as f:
            for _, row in top_10.iterrows():
                stock_code = str(row["股票代码"]).zfill(6)
                stock_name = str(row["股票名称"])
                f.write(f"{stock_code},{stock_name}\n")

        print(f"\n✓ 简化版结果已保存到: {output_file}")
        print("\n保存的文件内容:")
        with open(output_file, "r", encoding="utf-8") as fh:
            print(fh.read())

        # 释放大对象
        del df, work_df, df_sorted, top_10
        gc.collect()
        return True

    except Exception as e:
        print(f"处理数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("开始分析股票数据...")
    print("=" * 60)
    result = analyze_stock_data()
    if result is not None:
        print("\n✓ 分析完成！")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()