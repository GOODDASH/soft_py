import torch
import numpy as np
from torch.utils.data import Dataset


class TempRpmDataset(Dataset):
    def __init__(self, csv_file, t=3, m=5):
        """
        Args:
            csv_file (str): CSV文件的路径。
            t (int): 使用的温升时间步数。
            m (int): 后续的平均转速和温升的步数。
        """
        # 读取CSV文件到numpy数组中
        data = np.loadtxt(csv_file, delimiter=",")
        self.average_speeds = data[:, 0]  # 第一列：平均转速
        self.temps = data[:, 1:]  # 后面的列：温度传感器数据

        # 温升计算：每个温度值减去初始温度值
        self.first_temps = self.temps[0]  # 第一行温度值
        self.temp_rises = self.temps - self.first_temps  # 温升

        self.t = t  # 温升的时间步数
        self.m = m  # 后续平均转速和温升的步数

        self.features = []
        self.labels = []
        N = len(self.temp_rises)
        for i in range(N):
            # 构建t次温升，前面不足的填充0
            temp_rises_seq = []
            for j in range(self.t):
                idx = i - (self.t - 1 - j)
                if idx < 0:
                    # 前面不足的填充0
                    temp_rises_seq.append(np.zeros_like(self.temp_rises[0]))
                else:
                    temp_rises_seq.append(self.temp_rises[idx])
            temp_rises_seq = np.stack(temp_rises_seq)

            # 获取后面m个平均转速
            next_avg_speeds = self.average_speeds[i + 1 : i + 1 + self.m]
            # 如果不足m个，用最后一个值填充
            if len(next_avg_speeds) < self.m:
                if len(next_avg_speeds) == 0:
                    last_value = np.array([self.average_speeds[-1]])
                else:
                    last_value = next_avg_speeds[-1]
                padding = np.full((self.m - len(next_avg_speeds),), last_value)
                next_avg_speeds = np.concatenate([next_avg_speeds, padding])

            # 特征
            self.features.append((temp_rises_seq, next_avg_speeds))

            # 获取后面m个温升作为标签
            next_temp_rises = self.temp_rises[i + 1 : i + 1 + self.m]
            # 如果不足m个，用最后一个值填充
            if len(next_temp_rises) < self.m:
                if len(next_temp_rises) == 0:
                    last_temp = np.zeros_like(self.temp_rises[0])
                else:
                    last_temp = next_temp_rises[-1]
                padding = np.tile(last_temp, (self.m - len(next_temp_rises), 1))
                next_temp_rises = np.concatenate([next_temp_rises, padding], axis=0)

            self.labels.append(next_temp_rises)

        # 截断末尾无法构建完整标签的数据
        self.features = self.features[: -self.m] if self.m > 1 else self.features
        self.labels = self.labels[: -self.m] if self.m > 1 else self.labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        temp_rises_seq, next_avg_speeds = self.features[idx]
        label = self.labels[idx]

        # 转换为Tensor
        temp_rises_seq = torch.tensor(temp_rises_seq, dtype=torch.float32)
        next_avg_speeds = torch.tensor(next_avg_speeds, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return (temp_rises_seq, next_avg_speeds), label


if __name__ == "__main__":
    from torch.utils.data import ConcatDataset

    csv_files = [r"data\tsp_1.csv", r"data\tsp_2.csv", r"data\tsp_3.csv"]
    # 对每个CSV文件创建独立的数据集
    datasets = [TempRpmDataset(csv_file, t=3, m=5) for csv_file in csv_files]
    # 使用ConcatDataset将这些数据集合并
    combined_dataset = ConcatDataset(datasets)
    # 获取第二个数据样本
    (features, next_avg_speeds), label = combined_dataset[1]
    print("特征（历史温升序列）：", features)
    print("特征（后面平均转速）：", next_avg_speeds)
    print("标签（后面温升）：", label)
