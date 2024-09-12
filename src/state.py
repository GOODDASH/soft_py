import os
import csv
import datetime
from collections import deque
from typing import Optional
from src.core.nc_link import NCLink
from src.core.modbus_tcp import ModbusTCP
from src.core.yaml_handler import YamlHandler
from src.core.serial_port_reader import SerialPortReader
from src.thread.data_collector_thread import DataCollectorThread
from src.core.func import extract_numbers, extract_numbers
from PyQt5.QtCore import pyqtSignal as Signal, QObject


MAX_DATA_LEN = 3600 * 2  # 原始数据最大长度
DEFAULT_SAMPLE_PATH = "./data"  # 默认采集文件夹路径
RPM_AVG_INTER = 60  # 统计RPM平均值间隔(s)


class State(QObject):
    signal_connect_nc_status = Signal(list)
    signal_connect_tem_card_status = Signal(list)
    signal_open_port_status = Signal(bool)

    error_assert_temp_card_not_none = Signal(str)
    error_assert_nc_client_not_none = Signal(str)
    signal_start_sample_success = Signal()
    signal_show_orin_data = Signal()
    signal_show_sample_data = Signal()

    signal_get_datasets = Signal()
    signal_start_train = Signal()
    signal_train_val_loss = Signal(tuple)
    signal_train_finished = Signal(tuple)

    def __init__(self):
        super().__init__()
        self.nc_client: Optional[NCLink] = None
        self.tem_modbus_client: Optional[ModbusTCP] = None
        self.serial_port_client: Optional[SerialPortReader] = None

        self.sample_path = None
        self.data_collector_thread: Optional[DataCollectorThread] = None
        self.orin_data = dict()
        self.rule_data = dict()
        self.orin_count = 0
        self.rule_count = 0
        self.orin_data_filepath = None
        self.rule_data_filepath = None
        self.rpm_temp_data_filepath = None
        self.avg_rpm = []
        self.total_rpm = 0
        self.show_orin = True
        self.tem_from_nc = True
        self.nc_reg_num = None

        self.data = None
        self.cluster_res = None
        self.tsp_res = None
        # 存储不同模型是否需要重新创建数据集的布尔值字典
        self.flag_need_create_new_datasets = dict()
        self.datasets = None

        self.kf = None
        self.model = None
        self.best_model = None

        self.config = YamlHandler("res/config.yml")
        self.config.read()

    def on_close_window(self, config):
        self.config.data = config
        self.config.write()
        self.disconnect_nc()
        self.disconnect_tem_card()
        self.close_port()

    def connect_nc(self, para):
        from src.core.nc_link import NCLink

        self.nc_client = NCLink(mqtt_ip=para[0], mqtt_port=para[1], sn=para[2], client_id="DASH")
        success, info_str = self.nc_client.connect()
        if not success:
            self.nc_client = None
        self.signal_connect_nc_status.emit([success, info_str])

    def disconnect_nc(self):
        if self.nc_client:
            self.nc_client.disconnect()
            self.nc_client = None

    def connect_tem_card(self, para):
        from src.core.modbus_tcp import ModbusTCP

        self.tem_modbus_client = ModbusTCP(para)
        success, info_str = self.tem_modbus_client.connect_all()
        if not success:
            self.tem_modbus_client = None
        self.signal_connect_tem_card_status.emit([success, info_str])

    def disconnect_tem_card(self):
        if self.tem_modbus_client:
            self.tem_modbus_client.close_all()
            self.tem_modbus_client = None

    def open_port(self, para):
        from src.core.serial_port_reader import SerialPortReader

        if self.serial_port_client:
            self.serial_port_client.stop()
        try:
            self.serial_port_client = SerialPortReader(port=para[0], baud_rate=para[1])
            self.signal_open_port_status.emit(True)
        except:
            self.serial_port_client = None
            self.signal_open_port_status.emit(False)

    def close_port(self):
        if self.serial_port_client:
            self.serial_port_client.stop()

    def set_sample_save_path(self, path):
        self.sample_path = path

    def start_sample(self, para):
        # 前置条件判断
        if para["tem_from"] == "采集卡":
            self.tem_from_nc = False
            try:
                assert self.tem_modbus_client is not None
            except AssertionError:
                self.error_assert_temp_card_not_none.emit(
                    "选择从采集卡采集, 但是采集卡没有连接成功"
                )
                return
        else:
            self.tem_from_nc = True
            self.nc_reg_num = para["reg_num"]
            try:
                assert self.nc_client is not None
            except AssertionError:
                self.error_assert_nc_client_not_none.emit("选择从NC采集, 但是NC没有连接成功")
                return
        # 开始采集前根据连接的设备插入对应的空数据队列
        if self.nc_client:
            self.orin_data["nc_reg_g"] = deque(maxlen=MAX_DATA_LEN)
            self.orin_data["nc_axis_x"] = deque(maxlen=MAX_DATA_LEN)
            self.orin_data["nc_axis_y"] = deque(maxlen=MAX_DATA_LEN)
            self.orin_data["nc_axis_z"] = deque(maxlen=MAX_DATA_LEN)
            self.orin_data["nc_chan"] = deque(maxlen=MAX_DATA_LEN)
            self.rule_data["nc_reg_g"] = deque()
            self.rule_data["nc_axis_x"] = deque()
            self.rule_data["nc_axis_y"] = deque()
            self.rule_data["nc_axis_z"] = deque()
            self.rule_data["nc_chan"] = deque()
        if self.tem_modbus_client:
            self.orin_data["card_temp"] = deque(maxlen=MAX_DATA_LEN)
            self.rule_data["card_temp"] = deque()
        if self.serial_port_client:
            self.orin_data["error"] = deque(maxlen=MAX_DATA_LEN)
            self.rule_data["error"] = deque()
        # 创建csv文件路径
        if self.sample_path is None:
            self.sample_path = DEFAULT_SAMPLE_PATH
        file_name = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.orin_data_filepath = os.path.join(self.sample_path, f"{file_name}_orin.csv")
        self.rule_data_filepath = os.path.join(self.sample_path, f"{file_name}_rule.csv")
        self.rpm_temp_data_filepath = os.path.join(self.sample_path, f"{file_name}_rpm_temp.csv")
        # 实例化数据采集线程
        self.data_collector_thread = DataCollectorThread(
            para=para,
            nc_client=self.nc_client,
            tem_modbus_client=self.tem_modbus_client,
            serial_port_client=self.serial_port_client,
        )
        # 连接采集槽函数，开始采集
        self.data_collector_thread.signal_data_collected.connect(self.append_orin_data)
        self.data_collector_thread.signal_sample_collected.connect(self.append_rule_data)
        self.data_collector_thread.start()
        # 采集线程启动成功，通知界面切换成停止采集按钮
        self.signal_start_sample_success.emit()

    def stop_sample(self):
        self.data_collector_thread.stop()

    def append_orin_data(self, data):
        # data: [counter, {key: value}], key: "nc_data", "card_temp", "error"

        # FIXME: 没有处理可能出现采集到None或者空列表的情况
        # 保存采集的所有数据作为原始数据
        with open(self.orin_data_filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(extract_numbers(data[1]))
        self.orin_count += 1
        # 分类并提取数据
        for key, value in data[1].items():
            if key == "nc_data":
                # 机床G寄存器除以10才是温度
                self.orin_data["nc_reg_g"].append([num / 10 for num in value[0][: self.nc_reg_num]])
                self.orin_data["nc_axis_x"].append(value[1])
                self.orin_data["nc_axis_y"].append(value[2])
                self.orin_data["nc_axis_z"].append(value[3])
                self.orin_data["nc_chan"].append(extract_numbers(value[4]))
            else:
                self.orin_data[key].append(extract_numbers(value))
        # 统计一分钟平均转速和温度变化
        if "nc_chan" in self.orin_data:
            self.total_rpm += self.orin_data["nc_chan"][-1][2]
            if self.orin_count % RPM_AVG_INTER == 0:
                self.avg_rpm.append(self.total_rpm / RPM_AVG_INTER)
                if self.tem_from_nc:
                    temp = self.orin_data["nc_reg_g"][-1][: self.nc_reg_num]
                elif "card_temp" in self.orin_data:
                    temp = self.orin_data["card_temp"][-1]
                with open(self.rpm_temp_data_filepath, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(extract_numbers([self.avg_rpm[-1], temp]))
                self.total_rpm = 0
        # 发送显示原始数据信号
        if self.show_orin:
            self.signal_show_orin_data.emit()

    def append_rule_data(self):
        rule_record = []
        self.rule_count += 1
        for key, value in self.orin_data.items():
            self.rule_data[key].append(value[-1])
            if key == "nc_reg_g" and self.tem_from_nc:
                rule_record.append(value[-1][: self.nc_reg_num])
            if key == "card_temp" and not self.tem_from_nc:
                rule_record.append(value[-1])
            if key == "error":
                rule_record.append(value[-1])
        with open(self.rule_data_filepath, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(extract_numbers(rule_record))
        if not self.show_orin:
            self.signal_show_sample_data.emit()

    def get_data(self, para):
        from src.core.my_data import MyData

        self.data = MyData(para)
        # 只要导入新数据, 里面所有类型的模型训练都需要重新创建数据集
        if self.flag_need_create_new_datasets:
            for key in self.flag_need_create_new_datasets.keys():
                self.flag_need_create_new_datasets[key] = True

    def get_cluster_res(self, method_name, tsp_num):
        match method_name:
            case "FCM算法":
                from src.core.func import get_fcm_cluster_res

                self.cluster_res = get_fcm_cluster_res(self.data.Tdata, tsp_num)
            case "K均值算法":
                from src.core.func import get_kmeans_cluster_res

                self.cluster_res = get_kmeans_cluster_res(self.data.Tdata, tsp_num)

    def get_tra_tsp_res(self, method_name):
        match method_name:
            case "灰色关联度":
                from src.core.func import gra

                self.tsp_res = gra(self.data.Xdata, self.data.Tdata, self.cluster_res)
            case "相关系数":
                from src.core.func import cor

                self.tsp_res = cor(self.data.Xdata, self.data.Tdata, self.cluster_res)

    def get_ga_tsp_res(self, pop_size, iters):
        from functools import partial
        from src.core.ga import GA
        from src.core.func import get_kfold_cv
        from src.core.func import loss_func

        cv = get_kfold_cv(5, self.data.Tdata)
        # 设置的损失函数
        loss = partial(
            loss_func,
            Tdata=self.data.Tdata,
            Xdata=self.data.Xdata,
            idx=self.cluster_res,
            cv=cv,
        )
        optimizer = GA(loss, self.data.Tdata.shape[1], pop_size, iters, self.cluster_res)

        self.tsp_res = optimizer.opt()

    def check_edited_tsp_text(self, tsp_res_text):
        is_valid, error_message = self.validate_tsp_input(tsp_res_text)
        if not is_valid:
            return False, error_message
        tsp_list = list(map(int, tsp_res_text.split(",")))
        return True, tsp_list

    def save_chosen_data(self, file_path, tsp_res_text):
        if self.data is None:
            return False, "请先导入数据"
        is_valid, message = self.check_edited_tsp_text(tsp_res_text)
        if not is_valid:
            return False, message
        else:
            tsp_list = message
            self.data.write_file(file_path, tsp_list)
            return True, f"已保存至{file_path}"

    def mlr_fit(self, tsp_res_text):
        if self.data is None:
            return False, ["请先导入数据"]
        is_valid, message = self.check_edited_tsp_text(tsp_res_text)
        if not is_valid:
            return False, [message]
        else:
            tsp_list = message
            from src.core.func import check_tsp_per

            pred_list, rmse, coef, intercept = check_tsp_per(
                datas=self.data.data_arrays, data_cat=self.data.data_cat, tsp=tsp_list
            )
            return True, [pred_list, tsp_list, intercept, coef, rmse]

    @staticmethod
    def validate_tsp_input(input_text):
        if not input_text.strip():
            return False, "输入不能为空。"

        # 使用正则表达式检查输入格式是否正确（逗号分隔的整数）
        import re

        if not re.match(r"^(\d+,)*\d+$", input_text.strip()):
            return False, "输入格式不正确，请确保是逗号分隔的整数列表。"

        return True, ""

    def send_coef(self, coef):
        if self.nc_client is not None:
            items = {"values": []}
            items["values"].append(
                {
                    "id": "01035407",
                    "params": {
                        "index": 302137,
                        "value": coef[0],
                    },
                }
            )
            for index, val in enumerate(coef[1:]):
                items["values"].append(
                    {
                        "id": "01035407",
                        "params": {
                            "index": 700100 + index,
                            "value": val,
                        },
                    }
                )
            print(items)
            response = self.nc_client.set(items, need_parse=False)
            print(response)
            if response is not None:
                if all(response):
                    print("设值成功")
                else:
                    print("设值失败")
            else:
                print("设值失败")

    def load_model(self, para):
        """
        para["model_type"]  - 模型类型
            对于"GAT-LSTM":
                - para["num_heads"] - num_heads
                - para["gnn_dim"] - gnn_dim
                - para["lstm_dim"] - lstm_dim
                - para["num_nodes"] - num_nodes
                - para["seq_len"] - seq_len
            其他模型:
                -
        para["file_path"] - 模型文件路径
        """
        match para["model_type"]:
            case "GAT-LSTM":
                import torch
                from src.core.gat_lstm import GATLSTM

                self.model = GATLSTM(
                    in_dim=1,
                    out_dim=1,
                    gat_hidden_dim=para["gnn_dim"],
                    lstm_hidden_dim=para["lstm_dim"],
                    num_nodes=para["num_nodes"],
                    heads=para["num_heads"],
                )
                self.model.load_state_dict(torch.load(para["file_path"]))
            case "BPNN":
                import torch
                from src.core.bpnn import BPNN

                self.model = BPNN(hidden_units=para["hidden_dim"], input_shape=para["input_shape"], output_shape=1)
                self.model.load_state_dict(torch.load(para["file_path"]))

    def start_train(self, para, tsp_res_text):
        # 获取选取的温度测点列表
        tsp_list = list(map(int, tsp_res_text.split(",")))
        model_type = para["model_type"]
        model_para = para["model_para"]
        if (
            model_type not in self.flag_need_create_new_datasets
            or self.flag_need_create_new_datasets[model_type]
        ):
            self.datasets = self.get_datasets(model_type, model_para, tsp_list)

        self.reset_model(model_type, model_para)
        self.train_thread_start(model_type, para["train_para"])

    # 获取训练数据集
    def get_datasets(self, model_type, model_para, tsp_list):
        from sklearn.model_selection import KFold

        self.kf = KFold(n_splits=5, shuffle=True)
        self.signal_get_datasets.emit()  # 提示正在准备数据集
        datasets = None
        match model_type:
            case "GAT-LSTM":
                import torch
                from src.core.graph_data import GraphData

                edge_index = torch.tensor(model_para["edge_list"], dtype=torch.long)
                for idx, array in enumerate(self.data.data_arrays):
                    dataset = GraphData(
                        array[:, tsp_list],
                        edge_index,
                        array[:, -1],
                        model_para["seq_len"],
                    )
                    if idx == 0:
                        datasets = dataset
                    else:
                        datasets += dataset
            case "BPNN":
                import torch
                from torch.utils.data import TensorDataset

                torch_x = torch.from_numpy(self.data.Tdata[:, tsp_list]).float()
                torch_y = torch.from_numpy(self.data.Xdata.reshape((-1, 1))).float()
                datasets = TensorDataset(torch_x, torch_y)

        # 记录当前模型训练不需要重新创建数据集
        self.flag_need_create_new_datasets[model_type] = False

        return datasets

    # 重置模型
    def reset_model(self, model_type, model_para):
        match model_type:
            case "GAT-LSTM":
                from src.core.gat_lstm import GATLSTM

                self.model = GATLSTM(
                    in_dim=1,
                    out_dim=1,
                    gat_hidden_dim=model_para["gnn_dim"],
                    lstm_hidden_dim=model_para["lstm_dim"],
                    num_nodes=model_para["num_nodes"],
                    heads=model_para["num_heads"],
                )
            case _:
                from src.core.bpnn import BPNN

                self.model = BPNN(
                    input_shape=model_para["input_shape"], hidden_units=model_para["hidden_dim"], output_shape=1
                )

    def train_thread_start(self, model_type, train_para):
        self.signal_start_train.emit()
        match model_type:
            case "GAT-LSTM":
                from torch_geometric.loader import DataLoader as GeometricDataLoader
                dataloader = GeometricDataLoader
            case "BPNN":
                from torch.utils.data import DataLoader as TorchDataLoader
                dataloader = TorchDataLoader
                
        from src.thread.model_train_thread import ModelTrainThread

        self.thread_train = ModelTrainThread(
            model=self.model,
            datasets=self.datasets,
            dataloader=dataloader,
            kfold=self.kf,
            train_para=train_para,
        )

        self.thread_train.signal_train_val_loss.connect(self.signal_train_val_loss)
        self.thread_train.signal_train_finished.connect(self.signal_train_finished)
        self.thread_train.start()

    def increase_train(self, para, tsp_res_text):
        tsp_list = list(map(int, tsp_res_text.split(",")))
        model_type = para["model_type"]
        model_para = para["model_para"]
        if (
            model_type not in self.flag_need_create_new_datasets
            or self.flag_need_create_new_datasets[model_type]
        ):
            self.datasets = self.get_datasets(model_type, model_para, tsp_list)

        self.train_thread_start(model_type, para["train_para"])

    def save_model(self, file_path):
        import torch

        torch.save(self.model.state_dict(), file_path)

    # def import_tem_model(self, )
