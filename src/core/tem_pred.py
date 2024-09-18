import torch
from torch import nn



class TemPred(nn.Module):
    def __init__(self, temp_nums, hidden_dim, num_layers=1):
        super(TemPred, self).__init__()
        self.encoder = Encoder(temp_nums, hidden_dim, num_layers)
        self.decoder = Decoder(temp_nums, hidden_dim, num_layers)

    def forward(self, temp_seq, avg_speeds):
        # temp_seq: (batch_size, seq_len, temp_input_size)
        # avg_speeds: (batch_size, m)
        m = avg_speeds.size(1)
        outputs = []

        # 编码器
        h_n, c_n = self.encoder(temp_seq)

        # 初始解码器输入：最后一个温度
        input_temp = temp_seq[:, -1, :].unsqueeze(1)  # (batch_size, 1, temp_input_size)

        hidden = h_n
        cell = c_n

        for i in range(m):
            # 获取第 i 个平均转速
            input_speed = avg_speeds[:, i].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1)
            # 解码器一步预测
            pred_temp, hidden, cell = self.decoder(input_temp, input_speed, hidden, cell)
            outputs.append(pred_temp.unsqueeze(1))  # (batch_size, 1, output_size)

            # 更新输入温度为预测的温度
            input_temp = pred_temp.unsqueeze(1)  # (batch_size, 1, temp_input_size)

        outputs = torch.cat(outputs, dim=1)  # (batch_size, m, output_size)
        return outputs
    

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        # 使输入和输出张量的形状为 (batch_size, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, input_size)
        _, (h_n, c_n) = self.lstm(inputs)
        return h_n, c_n  # 返回最后的隐藏状态和记忆状态

class Decoder(nn.Module):
    def __init__(self, tem_nums, hidden_size,  num_layers=1):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(tem_nums + 1, hidden_size, num_layers, batch_first=True)  # 输入温度和平均转速
        self.fc = nn.Linear(hidden_size, tem_nums)

    def forward(self, input_temp, input_speed, hidden, cell):
        # input_temp: (batch_size, 1, input_size)
        # input_speed: (batch_size, 1, 1)
        input_combined = torch.cat((input_temp, input_speed), dim=2)  # (batch_size, 1, input_size + 1)
        output, (h_n, c_n) = self.lstm(input_combined, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # (batch_size, output_size)
        return prediction, h_n, c_n



if __name__ == "__main__":
    import torch

    temp_input_size = 6   # Number of features in the input temperature sequence
    hidden_size = 32       # Number of hidden units in LSTM layers
    num_layers = 2         # Number of layers in the LSTM

    # Create model
    model = TemPred(temp_input_size, hidden_size, num_layers)

    # Sample input data
    batch_size = 32         # Number of sequences in a batch
    seq_len = 3           # Length of the temperature sequence
    m = 10                # Number of future time steps to predict

    # Temperature sequences for the encoder (batch_size, seq_len, temp_input_size)
    temp_seq = torch.randn(batch_size, seq_len, temp_input_size)

    # Average speeds for future time steps (batch_size, m)
    avg_speeds = torch.randn(batch_size, m)

    # Forward pass through the model
    predicted_temps = model(temp_seq, avg_speeds)

    print(predicted_temps.shape)  # Should be (batch_size, m, output_size)
