import torch
import numpy as np
from torch_geometric.data import Dataset, Data


class GraphData(Dataset):
    def __init__(
        self,
        X: np.array,
        edge_index: torch.Tensor,
        y: np.array,
        seq_len: int,
        X_his: np.array = None,
        edge_weight: torch.FloatTensor = None,
    ):
        """
        从完整的温度数据中提取出指定序列长度的温度数据和对应最后一个时刻的热误差.

        返回torch_geometric.data.Data格式数据(图).

        如果有历史温度数据, 则对应填充到数据最上方, 没有或不够则填充0.

        参数:
        ----------
        X : np.array
            (温度长度, 传感器个数) - 温度数据
        edge_index : torch.FloatTensor
            (2, 节点之间的指向边数) - 节点的连接列表
        y : np.array
            (温度长度, 热误差自由度)
        seq_len : int
            指定的时间序列长度
        X_his : np.array, optional
            (温度长度, 传感器个数) - 历史温度数据, 默认值: None
        edge_weight: torch.FloatTensor, optional
            (每条有向边的权重) - 边权重, 默认值: None
        """
        super(GraphData, self).__init__()

        if X_his is not None:
            # 如果有历史数据(中间时刻), 在数据上方填充历史数据
            if X.shape[0] < seq_len - 1:
                # 历史数据长度不够则仍需填充0
                sta_X = np.vstack((X_his, X))
                pad_X = np.pad(sta_X, ((seq_len - 1 - X.shape[0], 0), (0, 0)))
            else:
                # 历史数据长度够则直接取最后(序列长度-1)行和seq_X拼接
                pad_X = np.vstack((X_his[-(seq_len - 1) :, :], X))
        else:
            # 如果没有历史数据(开始时刻), 在数据上方填充(序列长度-1)行的0
            pad_X = np.pad(X, ((seq_len - 1, 0), (0, 0)))

        self.X = pad_X  # [X.shape[0]+seq_len-1, X.shape[1]]
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.y = y
        self.seq_len = seq_len
        self.num_nodes = self.X.shape[1]

        self.X_seq = self.__get_X_seq()
        self.graph_list = self.__create_graph_list()

    def __get_X_seq(self):
        # 合并所有温度序列在三维数组中
        X_seq = np.empty((len(self.y), self.seq_len, self.num_nodes))
        for idx in range(len(self.y)):
            X_seq[idx] = self.X[idx : idx + self.seq_len, :]

        return X_seq

    def __create_graph_list(self):
        graph_list = []
        for i in range(len(self.y)):
            # 将一个时间序列的图合并在一个Data数据里
            x = torch.tensor(
                self.X_seq[i].reshape(-1, 1), dtype=torch.float
            )  # [seq_len*num_nodes, 1]
            # 要注意对edge_index偏移
            all_edges = [self.edge_index + j * self.num_nodes for j in range(self.seq_len)]
            edge_index_all = torch.cat(all_edges, dim=1)
            if self.edge_weight is not None:
                edge_weight_all = self.edge_weight.repeat(self.seq_len)
            else:
                edge_weight_all = None
            data = Data(
                x=x,
                edge_index=edge_index_all,
                edge_attr=edge_weight_all,
                y=torch.tensor(self.y[i], dtype=torch.float32),
            )
            graph_list.append(data)

        return graph_list

    def len(self):
        return len(self.y)

    def get(self, idx):
        return self.graph_list[idx]
