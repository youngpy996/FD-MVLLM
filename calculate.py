# 示例：合并两个不同维度的数据集
import torch
from torch.utils.data import Dataset, DataLoader

# 假设两个数据集
data1 = torch.randn(50, 10)  # 50条数据，每条10维
data2 = torch.randn(62, 20)  # 62条数据，每条20维


# 自定义 Dataset 合并两个数据集
class CombinedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.total_len = len(dataset1) + len(dataset2)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx], 0  # 0表示数据来源于第一个数据集
        else:
            return self.dataset2[idx - len(self.dataset1)], 1  # 1表示数据来源于第二个数据集


# 合并数据集
combined_dataset = CombinedDataset(data1, data2)

# 随机划分数据集
train_size = int(len(combined_dataset) * 0.7)
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

# 提取 train_dataset 中属于第一个数据集的数据
first_dataset_indices = [i for i, (_, source) in enumerate(train_dataset) if source == 0]
first_dataset = [train_dataset[i][0] for i in first_dataset_indices]
first_dataset = torch.stack(first_dataset)  # 转为 torch.Tensor

print("第一数据集形状：", first_dataset.shape)  # 输出: torch.Size([N, 10])

import torch
import random
from torch.utils.data import DataLoader, TensorDataset

# 定义数据
data = torch.randn(112, 10, 10)  # 假设原始数据是一个形状为 (112, 10, 10) 的张量

# 定义索引范围
indices = list(range(data.size(0)))  # 数据的第一维是样本数量
random.seed(42)  # 为了结果可复现
random.shuffle(indices)  # 随机打乱索引

# 计算划分索引
total = len(indices)
train_end = int(0.7 * total)
test_end = train_end + int(0.2 * total)

# 生成索引列表
train_list = indices[:train_end]
test_list = indices[train_end:test_end]
val_list = indices[test_end:]

# 使用索引列表划分数据
train_data = data[train_list]
test_data = data[test_list]
val_data = data[val_list]

# 创建 TensorDataset
train_dataset = TensorDataset(train_data)
test_dataset = TensorDataset(test_data)
val_dataset = TensorDataset(val_data)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 检查数据加载器
for batch in train_loader:
    print(f"Train batch shape: {batch[0].shape}")
    break
for batch in test_loader:
    print(f"Test batch shape: {batch[0].shape}")
    break
for batch in val_loader:
    print(f"Val batch shape: {batch[0].shape}")
    break

import torch
import numpy as np
import matplotlib.pyplot as plt

# 示例振动数据，长度为600
data = torch.randn(600)  # 使用随机数据，实际应用中请替换为你的振动数据

# 1. 计算FFT
fft_data = torch.fft.fft(data)

# 2. 获取频率轴
n = data.size(0)
sampling_rate = 1000  # 假设采样率为1000Hz
frequencies = torch.fft.fftfreq(n, d=1 / sampling_rate)

# 3. 计算频谱密度（PSD）
psd = torch.abs(fft_data) ** 2 / n

# 4. 计算总功率
total_power = torch.sum(psd)

# 5. 计算峰值频率
peak_frequency = frequencies[torch.argmax(psd)]

# 6. 计算均方根频率（RMS Frequency）
rms_frequency = torch.sqrt(torch.sum(frequencies ** 2 * psd) / torch.sum(psd))

# 7. 计算谱峭度（Spectral Kurtosis）
mean_psd = torch.mean(psd)
skewness = torch.mean(((psd - mean_psd) ** 4)) / (torch.mean(((psd - mean_psd) ** 2)) ** 2)

# 8. 计算中心频率
center_frequency = torch.sum(frequencies * psd) / torch.sum(psd)

# 打印结果
print(f"总功率: {total_power:.4f}")
print(f"峰值频率: {peak_frequency:.2f} Hz")
print(f"均方根频率: {rms_frequency:.2f} Hz")
print(f"谱峭度: {skewness:.4f}")
print(f"中心频率: {center_frequency:.2f} Hz")

# 可视化频谱
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:n // 2], psd[:n // 2].numpy())  # 只显示正频率部分
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
# plt.show()
plt.close()
print('*' * 50)
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# 示例振动数据，长度为600
data = torch.randn(600)  # 使用随机数据，实际应用中请替换为你的振动数据

# 1. 计算均值
mean_value = torch.mean(data)

# 2. 计算方差
variance = torch.var(data)

# 3. 计算标准差
std_deviation = torch.std(data)

# 4. 获取最大值和最小值
max_value = torch.max(data)
min_value = torch.min(data)

# 5. 计算峰值（Peak）
peak_value = torch.max(torch.abs(data))  # 绝对值最大值

# 6. 计算峭度（Kurtosis）
# 使用scipy库计算Kurtosis
kurt = kurtosis(data.numpy())

# 7. 计算偏度（Skewness）
# 使用scipy库计算Skewness
skewness = skew(data.numpy())

# 8. 计算均方根（RMS）
rms_value = torch.sqrt(torch.mean(data ** 2))

# 9. 计算峭度（Crest Factor）
crest_factor = peak_value / rms_value

# 打印结果
print(f"均值: {mean_value:.4f}")
print(f"方差: {variance:.4f}")
print(f"标准差: {std_deviation:.4f}")
print(f"最大值: {max_value:.4f}")
print(f"最小值: {min_value:.4f}")
print(f"峰值: {peak_value:.4f}")
print(f"峭度: {kurt:.4f}")
print(f"偏度: {skewness:.4f}")
print(f"均方根值: {rms_value:.4f}")
print(f"峭度系数: {crest_factor:.4f}")

# 可视化时域信号
plt.figure(figsize=(10, 6))
plt.plot(data.numpy())
plt.title('Time Domain Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.close()

print('*' * 50)
print([1] * 1000)

import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Define linear layers for query, key, value
        self.query_proj = nn.Linear(600, 600)
        self.key_proj = nn.Linear(4096, 4096)
        self.value_proj = nn.Linear(4096, 4096)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention

    def forward(self, target_embedding, source_embedding, value_embedding):
        batch_size, target_len, embed_dim = target_embedding.size()
        source_len = source_embedding.size(1)  # 4096

        # Linear projections
        query = self.query_proj(target_embedding)  # (batch_size, target_len, embed_dim)
        key = self.key_proj(source_embedding)  # (batch_size, source_len, embed_dim)
        value = self.value_proj(value_embedding)  # (batch_size, source_len, embed_dim)

        # Reshape for multi-head attention
        query = query.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # (batch_size, num_heads, target_len, head_dim)
        key = key.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1,
                                                                                        2)  # (batch_size, num_heads, source_len, head_dim)
        value = value.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # (batch_size, num_heads, source_len, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2,
                                                             -1)) * self.scale  # (batch_size, num_heads, target_len, source_len)
        attention_probs = torch.softmax(attention_scores, dim=-1)  # Normalize scores
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of value
        attention_output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, target_len, head_dim)

        # Combine heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, target_len, embed_dim)

        # Final output projection
        output = self.out_proj(attention_output)  # (batch_size, target_len, embed_dim)
        return output


# Example usage
batch_size = 16
target_len = 2
source_len = 1000
embed_dim = 4096  # You can adjust this based on your task
num_heads = 8

target_embedding = torch.randn(batch_size, target_len, 600).cuda()
source_embedding = torch.randn(batch_size, source_len, embed_dim).cuda()
value_embedding = source_embedding  # Typically the same as source_embedding

mha = MultiHeadAttention(embed_dim, num_heads).cuda()
output = mha(target_embedding, source_embedding, value_embedding)
print(output.shape)  # Should be (batch_size, target_len, embed_dim)

