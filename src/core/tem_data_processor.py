import torch
import numpy as np
from torch.utils.data import Dataset
from src.core.file2numpy import read_datafile_to_numpy


class TempDataProcessor(Dataset):
    def __init__(self, file_paths: list[str], seq_len: int):
        """
        初始化数据处理器。
        :param file_paths: 文件路径列表，每个路径指向一个热误差数据文件。
        :param seq_len: 历史时间步长，用于生成特征和标签。
        """
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.features = []
        self.labels = []
        self.load_and_process_data()

    def load_and_process_data(self):
        """
        从文件加载数据，进行处理和特征标签生成。
        """
        for file_path in self.file_paths:
            data_array = read_datafile_to_numpy(file_path)
            t_data = data_array[:, :-1]
            self.process_single_file(t_data)

    def process_single_file(self, t_data):
        # 在数据前填充 L-1 行零值
        pad_t_data = np.pad(t_data, ((self.seq_len - 1, 0), (0, 0)))
        # 生成特征和标签
        for i in range(self.seq_len, len(pad_t_data) - self.seq_len + 1):
            feature = pad_t_data[i - self.seq_len : i, :]
            label = pad_t_data[i : i + self.seq_len, :]
            self.features.append(feature)
            self.labels.append(label)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    paths = ["../../data/tsp_1.csv"]
    dataset = TempDataProcessor(paths, seq_len=5)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=8)
    for features, labels in data_loader:
        print(features.shape, labels.shape)
        # print(features)
