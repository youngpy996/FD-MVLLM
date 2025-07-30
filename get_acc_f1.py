import os
import pandas as pd


def main(folder_path):
    # 设置你的文件夹路径
    # folder_path = r"D:\KTH\online\LLM results\results without pic\results CWRU without pic Linux\results Qwen2.5-7B-Instruct without pic"  # 替换为你的实际文件夹路径

    # 获取所有CSV文件并按文件名排序
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

    # 选择前6个CSV文件
    selected_files = all_files[:6]

    # 存储结果的列表
    result_list = []

    # 处理每个选中的文件
    for filename in selected_files:
        file_path = os.path.join(folder_path, filename)

        # 读取CSV文件（假设没有标题行）
        df = pd.read_csv(file_path)

        # 获取最小值（假设数据在第一列）
        if not df.empty:
            max_value = df.iloc[:, 0].max()
        else:
            max_value = None  # 处理空文件情况

        # 添加到结果列表
        result_list.append({
            "filename": filename,
            "min_value": max_value
        })

    # 创建DataFrame并保存为新的CSV
    result_df = pd.DataFrame(result_list)
    result_df.to_csv(os.path.join(folder_path, r"max_values_results.csv"), index=False)

    print("处理完成！结果已保存为 max_values_results.csv")


if __name__ == '__main__':
    root_path = r'D:\KTH\online\LLM results\results without repro\results JNU without repro Linux'
    path_list = os.listdir(root_path)
    for p in path_list:

        main(os.path.join(root_path, p))
