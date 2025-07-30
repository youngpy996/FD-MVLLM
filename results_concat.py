import pandas as pd
import os

# 配置路径参数
base_dir = r'D:\KTH\online\LLM results\results without repro\results JNU without repro Linux'  # 父文件夹路径
csv_name = 'max_values_results.csv'  # 所有子文件夹中CSV的统一文件名（根据实际情况修改）
output_name = base_dir + r'/merged_result.csv'  # 输出文件名

# 自动获取所有子文件夹（仅目录）
subdirs = [d for d in os.listdir(base_dir)
           if os.path.isdir(os.path.join(base_dir, d))]

# 验证找到的文件夹数量
print(f"发现 {len(subdirs)} 个子文件夹：{subdirs}")

# 初始化合并容器
merged_df = pd.DataFrame()

# 遍历每个子文件夹
for folder in subdirs:
    # 构建完整文件路径
    file_path = os.path.join(base_dir, folder, csv_name)

    # 读取CSV文件（假设索引在第一列）
    df = pd.read_csv(file_path, index_col=0)

    # 提取value列并用文件夹名重命名
    merged_df[folder] = df['min_value']

# 保存合并结果
merged_df.to_csv(output_name)
print(f"合并完成！结果已保存为 {output_name}")
