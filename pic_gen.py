import pandas as pd
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import warnings
from multiprocessing import Pool, cpu_count

warnings.filterwarnings('ignore')


# 定义窗口分割函数
def sliding_window(signal, window_size, overlap):
    step = window_size - overlap
    return [signal[i:i + window_size] for i in range(0, len(signal) - window_size + 1, step)]


def process_window(args):
    """处理单个窗口的核心函数"""
    idx, window, window_size, output_dir, sf, wavelet = args
    # 计算尺度参数
    min_freq, max_freq = 1, 500
    cf = pywt.central_frequency(wavelet)
    scales = cf / (np.array([max_freq, min_freq]) * (1 / sf))
    scales = np.linspace(scales[0], scales[1], 128)

    # 小波变换
    coeffs, _ = pywt.cwt(window, scales, wavelet, sampling_period=1 / sf)

    # 绘图保存
    plt.figure(figsize=(6.4, 6.4), dpi=100)
    plt.imshow(np.abs(coeffs), cmap='jet', aspect='auto', origin='lower')
    plt.axis('off')
    plt.savefig(
        os.path.join(output_dir, f'time_frequency_window_{idx + 1}.png'),
        bbox_inches='tight', pad_inches=0
    )
    plt.close()


def process_file(file_info):
    """处理单个文件（并行单元）"""
    fname, input_dir, output_root, ws, ol, sf, wavelet = file_info
    output_dir = os.path.join(output_root, fname.replace('.csv', ''))
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    data = pd.read_csv(os.path.join(input_dir, fname))
    signal = data.iloc[:, 0].values

    # 顺序处理窗口（不再嵌套并行）
    for idx, window in enumerate(sliding_window(signal, ws, ol)):
        process_window((idx, window, ws, output_dir, sf, wavelet))


if __name__ == '__main__':
    # 配置参数
    config = {
        'window_size': 1000,
        'overlap': 400,
        'sampling_frequency': 50000,
        'wavelet': 'cmor',
        'output_root': r'E:\young\JNU-Bearing-Dataset-main\results test1',
        'input_dir': r'E:\young\JNU-Bearing-Dataset-main\input'
    }

    # 创建任务列表
    tasks = [
        (f, config['input_dir'], config['output_root'],
         config['window_size'], config['overlap'],
         config['sampling_frequency'], config['wavelet'])
        for f in os.listdir(config['input_dir'])
    ]

    # 启动单层进程池
    with Pool(processes=cpu_count()) as pool:
        pool.map(process_file, tasks)

    print(f'处理完成：{config["output_root"]}')
