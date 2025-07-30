import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt


# 1. 数据读取
data = pd.read_csv(r'D:\KTH\online\JNU-Bearing-Dataset-main\JNU-Bearing-Dataset-main/ib600_2.csv', header=None)
x = data.iloc[:2048, 0].values  # 提取第一列前2048个数据

# 2. 参数设置
Fs = 50000                      # 采样频率 (50 kHz)
N = len(x)                      # 数据长度
t = np.arange(N) / Fs           # 时间向量

# 3. 频率范围设置（线性分布）
f_max = Fs / 2                  # Nyquist频率 (25 kHz)
f_min = 1                       # 最低分析频率
frequencies = np.linspace(f_min, f_max, 100)  # 100个线性间隔频率点

# 4. 小波参数计算
wavelet_name = 'morl'           # Morlet小波
center_freq = pywt.central_frequency(wavelet_name)  # 获取小波中心频率
scales = center_freq * Fs / frequencies

# 5. 执行连续小波变换
coefficients, _ = pywt.cwt(x, scales, wavelet_name, sampling_period=1/Fs)

# 6. 绘制时频图
plt.figure(figsize=(10, 5))

# 主时频图
# plt.pcolormesh(t, frequencies, np.abs(coefficients),
#              shading='gouraud', cmap='jet')
plt.imshow(np.abs(coefficients), cmap='jet', aspect='auto', origin='lower')

plt.colorbar(label='Magnitude')
plt.clim(0, np.percentile(np.abs(coefficients), 95))  # 设置颜色范围

# plt.axis('xy')                  # 低频在底部
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('CWT Time-Frequency Analysis (Linear Frequency Axis)')

# 7. 可选：叠加原始信号曲线（取消注释使用）
# ax2 = plt.gca().twinx()
# ax2.plot(t, x, 'w', linewidth=0.5)
# ax2.set_ylabel('Amplitude', color='white')
# ax2.tick_params(axis='y', colors='white')

plt.tight_layout()
plt.show()
