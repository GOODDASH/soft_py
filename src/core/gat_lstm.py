from torch import nn
from torch_geometric.nn import GATConv


class GATLSTM(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        gat_hidden_dim: int,
        lstm_hidden_dim: int,
        num_nodes: int,
        heads: int,
    ):
        """结合图注意力网络和长短时记忆神经网络的时空预测热误差模型

        参数
        ----------
        in_dim : int
            输入维度, 应该为1, 因为每个传感器只有一个温度值
        out_dim : int
            输出维度, 也即需要预测的热误差自由度
        gcn_hidden_dim : int
            图注意力层的隐藏维度
        lstm_hidden_dim : int
            LSTM的隐藏维度
        num_nodes : int
            节点数, 即温度传感器个数
        heads : int
            GATConv中的参数heads, 注意力头数
        """
        super(GATLSTM, self).__init__()
        self.num_nodes = num_nodes
        # 两层GAT层
        self.gat1 = GATConv(in_dim, gat_hidden_dim, heads)
        self.gat2 = GATConv(gat_hidden_dim * heads, lstm_hidden_dim, heads)
        # 一层LSTM层
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim * num_nodes * heads,
            hidden_size=lstm_hidden_dim * num_nodes * heads,
            batch_first=True,
        )

        # 全连接线性层
        self.fc = nn.Linear(lstm_hidden_dim * num_nodes * heads, out_dim)

    def forward(self, data):
        """前向传播

        参数
        ----------
        data : torch_geometric.data.Data
            包含节点、节点特征、边权重...的图格式数据

        Returns
        -------
        torch.FloatTensor
            (batch_size) 预测的热误差
        """

        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # x = x.view(data.batch_size,self.seq_len,self.num_nodes, 1) 验证x的形状

        # [batch_size*seq_len*num_nodes, 1]
        x = self.gat1(x, edge_index, edge_weight).relu()
        # [batch_size*seq_len*num_nodes, gcn_hidden_dim*heads]
        x = self.gat2(x, edge_index, edge_weight).relu()
        # [batch_size*seq_len*num_nodes, lstm_hidden_dim*heads]
        x = x.view(-1, self.num_nodes * x.size(1))
        # [batch_size*seq_len, lstm_hidden_dim*heads*num_nodes]
        x = (
            x.view(data.batch_size, -1, x.size(1))
            if hasattr(data, "batch_size")
            else x.view(1, -1, x.size(1))
        )
        # [batch_size, seq_len, lstm_hidden_dim*heads*num_nodes] - 要将时间步的维度分离出来
        x, _ = self.lstm(x)
        # [batch_size, seq_len, lstm_hidden_dim*heads*num_nodes]
        x = x[:, -1, :]
        # [batch_size, lstm_hidden_dim*heads*num_nodes]
        x = self.fc(x)
        # [batch_size, out_dim]
        # if not hasattr(data, "batch_size"):
        #     x = x.squeeze(0)

        return x
