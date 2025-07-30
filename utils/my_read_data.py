import pandas as pd
import os
import random
import torchvision
from PIL import Image
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import torch
from torchvision.transforms import ToTensor, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset

plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, message="Glyph 8722")


def time_feature(data, device):
    """
    获取时域特征
    """
    # 1. 计算均值
    mean_value = torch.mean(data, dim=1)

    # 2. 计算方差
    variance = torch.var(data, dim=1)

    # 3. 计算标准差
    std_deviation = torch.std(data, dim=1)

    # 4. 获取最大值和最小值
    max_value = torch.max(data, dim=1)[0]
    min_value = torch.min(data, dim=1)[0]

    # 5. 计算峰值（Peak）
    peak_value = torch.max(torch.abs(data), dim=1)[0]  # 绝对值最大值

    # 8. 计算均方根（RMS）
    rms_value = torch.sqrt(torch.mean(data ** 2, dim=1))

    # # 6. 计算峭度（Kurtosis）
    # # 使用scipy库计算Kurtosis
    # kurt = kurtosis(data.numpy())
    #
    # # 7. 计算偏度（Skewness）
    # # 使用scipy库计算Skewness
    # skewness = skew(data.numpy())
    #
    # # 9. 计算峭度（Crest Factor）
    # crest_factor = peak_value / rms_value

    kurt = torch.tensor(
        [kurtosis(sample.cpu().numpy()) for sample in data]
    ).to(device)
    skewness = torch.tensor(
        [skew(sample.cpu().numpy()) for sample in data]
    ).to(device)

    # Crest factor = Peak value / RMS
    crest_factor = peak_value / rms_value

    return mean_value, variance, std_deviation, max_value, min_value, peak_value, kurt, skewness, rms_value, crest_factor


def fft_feature(data, sampling_rate, device, batch_y=None):
    """
    获取频域特征
    """

    # def frequency_domain_analysis(batch_data, sampling_rate=2560):
    """
    Perform frequency-domain analysis on a batch of vibration data.

    Args:
        batch_data (torch.Tensor): Input tensor with shape (batch_size, seq_length).
        sampling_rate (int): Sampling rate of the vibration data.

    Returns:
        dict: A dictionary containing frequency-domain features for each batch.
    """
    # 确保 batch_y 存在且为张量
    # assert batch_y is not None, "batch_y 未正确加载！"
    #
    # # 将 batch_y 转换为浮点张量，并确保它是标量或单元素张量
    # batch_y_float = batch_y.float()
    #
    # # 直接获取标量值（无需计算均值）
    # value = batch_y_float.item()  # 或直接使用 batch_y_float.squeeze().item()
    #
    # # 根据标量值判断采样率
    # if value == 0:
    #     sampling_rate = 97656
    # elif value in [1, 2]:
    #     sampling_rate = 48828
    # else:
    #     print(f"值 {value} 无法匹配采样率")
    #     raise ValueError("采样率无法对齐")
    # # 若 data 形状为 [600]，添加批次维度（变为 [1, 600]）
    # if len(data.shape) == 1:
    #     data = data.unsqueeze(0)  # 或 data = data.view(1, -1)
    # Perform FFT
    fft_data = torch.fft.fft(data)
    n = data.size(1)
    frequencies = torch.fft.fftfreq(n, d=1 / sampling_rate).to(device)

    # Only consider the positive half of the spectrum
    half_n = n
    fft_data = fft_data[:, :half_n]
    frequencies = frequencies[:half_n]
    psd = torch.abs(fft_data) ** 2 / n  # Power Spectral Density (PSD)

    # Calculate frequency-domain features
    total_power = torch.sum(psd, dim=1)[0]
    peak_indices = torch.argmax(psd, dim=1)
    peak_frequency = frequencies[peak_indices]
    rms_frequency = torch.sqrt(torch.sum(frequencies ** 2 * psd, dim=1) / total_power)
    center_frequency = torch.sum(frequencies * psd, dim=1) / total_power

    # return {
    #     "psd": psd,
    #     "total_power": total_power,
    #     "peak_frequency": peak_frequency,
    #     "rms_frequency": rms_frequency,
    #     "center_frequency": center_frequency,
    # }

    # # 1. 计算FFT
    # fft_data = torch.fft.fft(data)
    #
    # # 2. 获取频率轴
    # n = data.size(1)
    # sampling_rate = 1000  # 假设采样率为1000Hz
    # frequencies = torch.fft.fftfreq(n, d=1 / sampling_rate)
    #
    # # 3. 计算频谱密度（PSD）
    # psd = torch.abs(fft_data) ** 2 / n
    #
    # # 4. 计算总功率
    # total_power = torch.sum(psd)
    #
    # # 5. 计算峰值频率
    # peak_frequency = frequencies[torch.argmax(psd)]
    #
    # # 6. 计算均方根频率（RMS Frequency）
    # rms_frequency = torch.sqrt(torch.sum(frequencies ** 2 * psd) / torch.sum(psd))
    #
    # # 7. 计算谱峭度（Spectral Kurtosis）
    # mean_psd = torch.mean(psd)
    # skewness = torch.mean(((psd - mean_psd) ** 4)) / (torch.mean(((psd - mean_psd) ** 2)) ** 2)
    #
    # # 8. 计算中心频率
    # center_frequency = torch.sum(frequencies * psd) / torch.sum(psd)

    return psd, total_power, peak_frequency, rms_frequency, center_frequency


class CombinedDatasetThree(Dataset):
    def __init__(self, csv_data, img_data, labels):
        """
        初始化数据集
        :param csv_data: Tensor 或 list，CSV数据
        :param img_data: Tensor 或 list，图像数据
        :param labels: Tensor 或 list，标签数据
        """
        self.csv_data = csv_data
        self.img_data = img_data
        self.labels = labels

        # 确保三个数据长度一致
        assert len(self.csv_data) == len(self.img_data) == len(self.labels), "Dataset lengths do not match!"

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        # 从 csv_data 中提取每一列
        # print(self.csv_data)
        csv_item = self.csv_data[index][0]
        img_item = self.img_data[index][0]
        label_item = self.labels[index]

        # 转换为 tensor
        csv_item = torch.tensor(csv_item, dtype=torch.float32)
        img_item = torch.tensor(img_item, dtype=torch.float32)
        label_item = torch.tensor(label_item, dtype=torch.long)

        return csv_item, img_item, label_item


def collate_fn(batch):
    try:
        # CSV 数据处理：拼接成 (batch_size, csv_length) 的二维张量
        csv_batch = torch.stack([item[0] for item in batch], dim=0)
        # 图像数据处理：拼接成 (batch_size, channels, height, width)
        img_batch = torch.stack([item[1] for item in batch], dim=0)
        # 标签数据处理：拼接成 (batch_size,)
        label_batch = torch.stack([item[2] for item in batch], dim=0)
    except Exception as e:
        print(f"Collate Function Error: {e}")
        print(f"Batch: {batch}")
        raise
    return csv_batch, img_batch, label_batch


def data_indices(data_csv, image_data, labels, args):
    if len(data_csv) != len(image_data):
        raise NotImplementedError
    indices = list(range(len(data_csv)))
    random.shuffle(indices)  # 随机打乱索引

    # 计算划分索引
    # num = int(len(indices) / 16)
    total = len(indices)
    train_end = int(0.8 * total)
    test_end = train_end + int(0.1 * total)
    # train_end = int(16 * 5)
    # test_end = train_end + int(16 * 1)
    # 根据随机索引划分数据
    train_list = indices[:train_end]
    test_list = indices[train_end:test_end]
    val_list = indices[test_end:]

    def tensor2dataloader(data):
        train = data[train_list]
        test = data[test_list]
        val = data[val_list]
        train_data = TensorDataset(train)
        test_data = TensorDataset(test)
        val_data = TensorDataset(val)

        # train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
        # test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        # val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
        # return train_loader, test_loader, val_loader
        return train_data, test_data, val_data

    train_csv_loader, test_csv_loader, val_csv_loader = tensor2dataloader(data_csv)
    train_img_loader, test_img_loader, val_img_loader = tensor2dataloader(image_data)
    train_labels_loader, test_labels_loader, val_labels_loader = tensor2dataloader(labels)
    train_loader = CombinedDatasetThree(train_csv_loader, train_img_loader, train_labels_loader)
    test_loader = CombinedDatasetThree(test_csv_loader, test_img_loader, test_labels_loader)
    val_loader = CombinedDatasetThree(val_csv_loader, val_img_loader, val_labels_loader)
    train_loader1 = DataLoader(train_loader, batch_size=args.batch_size, shuffle=False, drop_last=True)
    test_loader1 = DataLoader(test_loader, batch_size=args.batch_size, shuffle=False, drop_last=True)
    val_loader1 = DataLoader(val_loader, batch_size=args.batch_size, shuffle=False, drop_last=True)

    return train_loader1, test_loader1, val_loader1
    # return train_csv_loader, test_csv_loader, val_csv_loader, train_img_loader, test_img_loader, val_img_loader, train_labels_loader, test_labels_loader, val_labels_loader


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


# 加载时频图数据
def load_time_frequency_images(folder_path):
    images = []
    labels = []
    # 遍历每个类别文件夹
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                img = Image.open(image_path).convert('RGB')  # 转为RGB格式
                # print(img)
                img_array = np.array(img).flatten()  # 转为数组
                # print(img_array)
                # print(img_array.shape)
                images.append(img_array)
                labels.append(class_folder)  # 类别来自文件夹名称
    return torch.tensor(images), labels


# def sliding_window(data, window_size, overlap):
#     """
#     将数据按窗口大小和重叠量划分成多行。
#
#     :param data: 输入数据（列表或一维数组）
#     :param window_size: 每个窗口的大小
#     :param overlap: 窗口的重叠数量
#     :return: 分块后的二维数组
#     """
#     step = window_size - overlap  # 每次移动的步长
#     num_windows = (len(data) - window_size) // step + 1  # 计算窗口数量
#     windows = [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]
#     return np.array(windows)
# return np.array(windows).reshape(num_windows, -1)
def sliding_window(data, window_size, overlap):
    """
    将数据按窗口大小和重叠量划分成多行。

    :param data: 输入数据（列表或一维数组）
    :param window_size: 每个窗口的大小
    :param overlap: 窗口的重叠数量
    :return: 分块后的二维数组
    """
    step = window_size - overlap  # 每次移动的步长
    num_windows = (len(data) - window_size) // step + 1  # 计算窗口数量

    windows = [data[i:i + window_size] for i in range(0, len(data) - window_size + 1, step)]
    windows = np.array(windows)
    # 确保返回的结果是一个 NumPy 数组，且每个窗口的数据类型一致
    return windows.reshape(windows.shape[0], windows.shape[1])


def my_read_csv(input_csv, output_csv, window_size, overlap, i, nrows=10000, usecols=0):
    # 读取数据
    df = pd.read_csv(input_csv, usecols=[usecols])
    # df = pd.read_csv(input_csv, usecols=[usecols], nrows=nrows)
    data = df.values  # 提取指定列数据

    # 应用滑动窗口
    windowed_data = sliding_window(data, window_size, overlap)

    # 保存结果
    windowed_df = pd.DataFrame(windowed_data)
    # windowed_df.to_csv(output_csv, index=False)
    # windowed_df = windowed_df.assign(new_column=1)  # 在数据右侧增加一列

    # 生成指定长度的一列数据，并设置所有值为2
    label = pd.DataFrame.from_dict({'label': [i] * len(windowed_df)})
    # 返回数据与标签转换为张量
    # windowed_tensor = torch.tensor(windowed_data, dtype=torch.float32)
    # label_tensor = torch.tensor(label.values, dtype=torch.long)

    return windowed_df, label


def read_data(csv_root, image_root, output_csv, window_size=600, overlap=200):
    """
    csv_root (28, 600)
    image_root ([4, 28, 4, 128, 128])
    """
    df = pd.DataFrame()
    label = pd.DataFrame()
    image_list = []
    for i, csv_file in enumerate(os.listdir(csv_root)):
        csv_name = os.path.join(csv_root, csv_file)
        # print(csv_name)
        # 获取csv acc数据
        windowed_df, lab = my_read_csv(csv_name, output_csv, window_size, overlap, i)
        # print(windowed_df.shape)
        df = pd.concat([df, windowed_df], axis=0)
        label = pd.concat([label, lab], axis=0)

        # 获取image数据
        img_file_name = os.path.join(image_root, csv_file.replace('.csv', ''))
        # images = []
        # labels = []
        for image_file in os.listdir(img_file_name):
            image_path = os.path.join(img_file_name, image_file)
            # img = torchvision.io.read_image(image_path)  # 转为RGB格式
            image_pil = Image.open(image_path).resize((128, 128))
            image_tensor = ToTensor()(image_pil)  # 转换为 [C, H, W] 格式，且归一化到 [0, 1]

            # img = Image.open(image_path).convert('RGB')  # 转为RGB格式
            # img_array = np.array(img).flatten()  # 转为数组
            # print(img_array)
            # print(img_array.shape)
            image_list.append(image_tensor)
            # labels.append(class_folder)  # 类别来自文件夹名称
        # image_data = torch.tensor(np.array(images))
        # print(image_data.shape)
        #     image_list.append(image_data)
    # image_list = torch.tensor(np.array(image_list))
    image_list = torch.stack(image_list)  # 使用 stack 保证 shape 统一

    # print(image_list.shape)
    # return torch.tensor(np.array(df)), image_list, torch.tensor(np.array(label))
    return torch.tensor(df.values, dtype=torch.float32, requires_grad=True), torch.tensor(image_list,
                                                                                          requires_grad=True), torch.tensor(
        label.values, dtype=torch.long)


if __name__ == '__main__':
    # 参数设置
    input_csv = r"D:\KTH\online\matlab output\input\内圈-acc.csv"  # 输入CSV文件路径
    output_csv = "output_windowed_data.csv"  # 输出CSV文件路径
    window_size = 600
    overlap = 256

    csv_root = r'E:\dayoung\matlab output\input'
    image_root = r"E:\dayoung\matlab output\results"

    df, image_data = read_data(csv_root, image_root, output_csv)  # image_root ([4, 28, 4, 128, 128])
    # print(df)

    # # 加载时频图数据
    # folder_path = r'D:\KTH\online\matlab output\results'  # 替换为时频图存储的文件夹路径
    # time_frequency_images, image_labels = load_time_frequency_images(folder_path)  # [112, 566, 548, 3]
    # print(time_frequency_images)
    # print(time_frequency_images.shape)
    # # 检查数据
    # print(f"Loaded {len(time_frequency_images)} time-frequency images.")
    # print(f"Labels: {set(image_labels)}")
