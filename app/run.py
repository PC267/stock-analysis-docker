# run.py
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 保留 WARNING，但禁止 GPU 相关警告
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ========== 修改导入路径 ==========
# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 添加report_chart目录到路径
report_chart_dir = os.path.join(current_dir, "report_chart")
sys.path.insert(0, report_chart_dir)

# 现在可以导入report_config了
import report_config


# 设置保存路径 - 使用report_config模块中的配置
SAVE_PATH = report_config.REPORT_DIR

print("\n========== 开始查找和删除风险等级文件 ==========")
# 目标目录 - 确保是列表格式
if isinstance(SAVE_PATH, str):
    target_dirs = [SAVE_PATH]
elif isinstance(SAVE_PATH, list):
    target_dirs = SAVE_PATH
else:
    print(f"⚠️  REPORT_DIR 类型异常 ({type(SAVE_PATH)})，使用 report_chart_dir 作为默认")
    target_dirs = [report_chart_dir]

for dir_path in target_dirs:   
    if os.path.exists(dir_path): 
        # 列出所有Excel文件
        excel_files = []
        # 修复：使用 os.listdir() 获取目录内容
        for filename in os.listdir(dir_path):
            if filename.lower().endswith(('.xlsx', '.xls')):
                full_path = os.path.join(dir_path, filename)
                excel_files.append((filename, full_path))
        # 查找包含特定关键词的文件
        risk_files = []
        for filename, full_path in excel_files:
            lower_name = filename.lower()
            if '风险' in filename or '等级' in filename or '汇总' in filename:
                risk_files.append((filename, full_path))
        
        if risk_files:
            print(f"\n✅ 找到 {len(risk_files)} 个风险相关文件:")
            for risk_file, full_path in risk_files:
                print(f"  🗑️ 准备删除: {risk_file}")
                try:
                    os.remove(full_path)
                    print(f"  ✅ 已删除: {risk_file}")
                except Exception as e:
                    print(f"  ❌ 删除失败 {risk_file}: {e}")
        else:
            print("⚠️ 未找到风险等级相关的Excel文件")
    else:
        print("❌ 目录不存在，跳过检查")

# 添加当前目录到Python路径
sys.path.insert(0, current_dir)

print("\n" + "=" * 48)
print("股票智能模拟分析系统 - baostock真实数据版")
print("=" * 48)

try:
    from main import main
    if __name__ == "__main__":
        print("🚀 启动主程序...")
        main()
except ImportError as e:
    print(f"❌ 无法导入 main 模块: {e}")
    print(f"📁 当前 Python 路径:")
    for path in sys.path:
        print(f"  - {path}")
except Exception as e:
    print(f"❌ 运行主程序时出错: {e}")