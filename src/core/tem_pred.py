import torch
from torch import nn


class TemPred(nn.Module):
    """
    温度预测模型, 有编码解码器组成, 工况参数可选
    """

    def __init__(
        self, num_sensors=6, hidden_dim=50, future_seq_length=5, num_layers=1, use_speed=False
    ):
        super(TemPred, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.future_seq_length = future_seq_length
        self.num_layers = num_layers
        self.use_speed = use_speed

        # 编码器：LSTM
        self.encoder = nn.LSTM(
            input_size=num_sensors, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True
        )

        # 解码器的输入维度取决于是否使用转速
        decoder_input_size = num_sensors + 1 if use_speed else num_sensors
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # 线性层：从隐藏状态映射到预测的温度
        self.linear = nn.Linear(hidden_dim, num_sensors)

    def forward(self, temp_seq, future_speed_seq=None):
        # temp_seq: (batch_size, seq_length, num_sensors)
        # future_speed_seq: (batch_size, future_seq_length, 1) or None

        # 编码器部分
        _, (hidden, cell) = self.encoder(temp_seq)

        # 解码器的初始输入
        last_temp = temp_seq[:, -1, :].unsqueeze(1)  # (batch_size, 1, num_sensors)
        if self.use_speed and future_speed_seq is not None:
            first_speed = future_speed_seq[:, 0, :].unsqueeze(1)  # (batch_size, 1, 1)
            decoder_input = torch.cat(
                (last_temp, first_speed), dim=2
            )  # (batch_size, 1, num_sensors + 1)
        else:
            decoder_input = last_temp  # (batch_size, 1, num_sensors)

        predictions = []

        for i in range(self.future_seq_length):
            # 解码器部分
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            decoder_output = self.linear(output[:, 0, :])
            predictions.append(decoder_output.unsqueeze(1))

            # 更新解码器输入
            if self.use_speed and future_speed_seq is not None and i < self.future_seq_length - 1:
                next_speed = future_speed_seq[:, i + 1, :].unsqueeze(1)
                decoder_input = torch.cat((decoder_output.unsqueeze(1), next_speed), dim=2)
            else:
                decoder_input = decoder_output.unsqueeze(1)

        # predictions: (batch_size, future_seq_length, num_sensors)
        predictions = torch.cat(predictions, dim=1)

        return predictions
