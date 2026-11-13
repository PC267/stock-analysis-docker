#!/bin/bash

# 强制加载 bashrc 确保拿到 GITHUB_TOKEN
source /root/.bashrc

REPO="PC267/daily_stock_analysis"
WORKFLOW_FILE="daily_analysis.yml"
EXCEL_FILE="/vol1/1000/docker/ZWJ_stock-analysis/app/results/reports/风险等级汇总.xlsx"
VAR_NAME="STOCK_LIST"

# 路径更新为新的 Stock_list 目录
SAVE_DIR="/vol1/1000/docker/ZWJ_stock-analysis/app/results/Stock_list"
TODAY=$(date +%Y-%m-%d)
SAVE_FILE="$SAVE_DIR/$TODAY.txt"

# 检查环境变量
if [ -z "$GITHUB_TOKEN" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 错误: 未检测到 GITHUB_TOKEN。"
    exit 1
fi

# 检查目录和文件
mkdir -p "$SAVE_DIR"
if [ ! -f "$EXCEL_FILE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ 错误: 找不到 Excel 文件: $EXCEL_FILE"
    # 即使没找到文件，也强行触发一次 Workflow，保证 GitHub 能跑
    /root/trigger_workflow.sh
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔍 开始从 Excel 筛选数据..."

# ==========================================
# 1. Python 读 Excel 并筛选
# ==========================================
STOCK_CODES=$(python3 - << 'EOF'
import pandas as pd
import sys

file_path = "/vol1/1000/docker/ZWJ_stock-analysis/app/results/reports/风险等级汇总.xlsx"
try:
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    
    code_col = next((c for c in df.columns if '代码' in c or 'Code' in c or c == 'code'), None)
    return_col = next((c for c in df.columns if 'Total Return' in c or '总收益' in c or '收益' in c or '模拟价格差' in c), None)
    win_col = next((c for c in df.columns if '胜率' in c or 'Win' in c), None)

    if not code_col or not return_col or not win_col:
        print(f"ERROR: 列名匹配失败! 现有列: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    def clean_numeric(val):
        if pd.isna(val): return 0.0
        if isinstance(val, str):
            val = val.replace('%', '').strip()
        try: return float(val)
        except: return 0.0

    df['clean_return'] = df[return_col].apply(clean_numeric)
    df['clean_win'] = df[win_col].apply(clean_numeric)
    
    # 筛选条件：总收益 > 70% 且 胜率 > 60%
    filtered = df[(df['clean_return'] > 70) & (df['clean_win'] > 60)]
    
    codes = filtered[code_col].astype(str).str.zfill(6).str.strip().tolist()
    print(",".join(codes))
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Python 读取 Excel 失败。跳过变量更新。"
    # 即使失败也触发
    /root/trigger_workflow.sh
    exit 1
fi

# ==========================================
# 2. 判断是否有结果并更新
# ==========================================
if [ -z "$STOCK_CODES" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 今日无符合条件（收益>70% 且 胜率>60%）的股票。"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 保持 GitHub 现有 STOCK_LIST 不变。"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ 今日筛选出股票: $STOCK_CODES"
    
    # 存档当天的结果
    echo "$STOCK_CODES" > "$SAVE_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 📁 已保存当天的股票池到: $SAVE_FILE"

    # 更新 GitHub 变量
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 正在更新 GitHub 变量 $VAR_NAME ..."
    VAR_URL="https://api.github.com/repos/$REPO/actions/variables/$VAR_NAME"
    
    HTTP_CODE=$(curl -s -o /tmp/var_resp.json -w "%{http_code}" -X PATCH \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Bearer $GITHUB_TOKEN" \
      -H "X-GitHub-Api-Version: 2022-11-28" \
      "$VAR_URL" \
      -d "{\"name\":\"$VAR_NAME\",\"value\":\"$STOCK_CODES\"}")
      
    if [ "$HTTP_CODE" -eq 404 ]; then
        HTTP_CODE=$(curl -s -o /tmp/var_resp.json -w "%{http_code}" -X POST \
          -H "Accept: application/vnd.github+json" \
          -H "Authorization: Bearer $GITHUB_TOKEN" \
          -H "X-GitHub-Api-Version: 2022-11-28" \
          "https://api.github.com/repos/$REPO/actions/variables" \
          -d "{\"name\":\"$VAR_NAME\",\"value\":\"$STOCK_CODES\"}")
    fi

    if [ "$HTTP_CODE" -eq 204 ] || [ "$HTTP_CODE" -eq 201 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ GitHub 变量更新成功！"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ GitHub 变量更新失败，状态码: $HTTP_CODE"
        cat /tmp/var_resp.json
    fi
fi

# ==========================================
# 3. 无论如何都触发 GitHub Actions
# ==========================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 准备触发工作流..."
/root/trigger_workflow.sh

rm -f /tmp/var_resp.json
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🎉 今日流程执行完毕！"
