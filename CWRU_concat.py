from pathlib import Path
import pandas as pd

# ==================== 配置路径 ====================
# 1. 你的主文件夹路径（包含多个子文件夹）
root_dir = Path(r"D:\KTH\online\CWRU_Bearing_NumPy-main\CWRU_Data\data\O")

# 2. 合并后输出的文件路径
output_file = Path(r"D:\KTH\online\CWRU_Bearing_NumPy-main\CWRU_Data\data/O.csv")
# ==================================================

# 递归查找主文件夹及所有子文件夹下的 .csv 文件
csv_files = list(root_dir.rglob("*.csv"))

if not csv_files:
    print("未在指定目录下找到任何 CSV 文件，请检查路径。")
else:
    df_list = []

    # 逐个读取 CSV
    for file in csv_files:
        df = pd.read_csv(file)

        # ---------------- 修改部分开始 ----------------
        n_rows = len(df)

        # 计算 1024 的最大整数倍行数
        # 例如：n_rows = 2050 -> 2050 // 1024 = 1 -> valid_rows = 1024
        # 例如：n_rows = 3000 -> 3000 // 1024 = 2 -> valid_rows = 2048
        valid_rows = (n_rows // 1024) * 1024

        # 使用 iloc 截取有效数据，舍弃多余的尾部数据
        df_cropped = df.iloc[:valid_rows]

        # 为了防止某些文件数据量小于 1024 导致截断后为空，这里加个判断
        if not df_cropped.empty:
            df_list.append(df_cropped)
        else:
            print(f"提示: 文件 {file.name} 的数据少于 1024 行，已跳过。")
        # ---------------- 修改部分结束 ----------------

    # 确保合并列表不为空（避免所有文件都小于 1024 行的情况）
    if not df_list:
        print("所有文件均不足 1024 行，无数据可合并。")
    else:
        # 纵向拼接（按行合并）
        merged_df = pd.concat(df_list, axis=0, ignore_index=True)

        # 保存合并后的 CSV（index=False 表示不保存 DataFrame 的行索引）
        merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print(f"\n合并完成！共处理了 {len(csv_files)} 个文件。")
        print(f"总数据行数（不含表头）: {len(merged_df)}")
        print(f"合并后的总行数是否为 1024 的倍数: {len(merged_df) % 1024 == 0}")
        print(f"输出路径: {output_file}")
