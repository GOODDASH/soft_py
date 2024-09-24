import os
import csv
import datetime
from typing import Optional
from collections import deque
from src.core.nc_link import NCLink
from src.core.modbus_tcp import ModbusTCP
from src.core.yaml_handler import YamlHandler
from src.core.serial_port_reader import SerialPortReader
from src.thread.load_module_thread import ModuleLoaderThread
from src.thread.data_collector_thread import DataCollectorThread
from PyQt5.QtCore import pyqtSignal as Signal, QObject, QTimer


MAX_DATA_LEN = 1800  # 默认原始数据最大长度, 只显示最近半小时
DEFAULT_SAMPLE_PATH = "./data"  # 默认采集文件夹路径
RPM_AVG_INTER = 60  # 默认统计RPM平均值间隔(s)


class State(QObject):
    signal_module_loaded = Signal()

    signal_connect_nc_status = Signal(list)
    signal_connect_tem_card_status = Signal(list)
    signal_open_port_status = Signal(bool)

    error_assert_temp_card_not_none = Signal(str)
    error_assert_nc_client_not_none = Signal(str)
    error_assert_serial_port_client_not_none = Signal(str)
    signal_start_sample_success = Signal()
    signal_show_orin_data = Signal()
    signal_update_time = Signal()
    signal_show_sample_data = Signal()

    signal_ga_tsp_done = Signal()

    signal_get_datasets = Signal()
    signal_start_train = Signal()
    signal_train_val_loss = Signal(tuple)
    signal_train_finished = Signal(tuple)

    signal_fit_coef = Signal(list)

    def __init__(self):
        super().__init__()
        self.nc_client: Optional[NCLink] = None
        self.tem_modbus_client: Optional[ModbusTCP] = None
        self.serial_port_client: Optional[SerialPortReader] = None

        # 采集文件夹路径
        self.sample_path = None
        # 采集线程
        self.data_collector_thread: Optional[DataCollectorThread] = None
        # 采集的原始数据
        self.orin_data = dict()
        # 采集的规则数据
        self.rule_data = dict()
        # 采集原始数据计数
        self.orin_count = 0
        # 采集规则数据计数
        self.rule_count = 0
        # 采集错误数据计数
        self.err_count = 0
        # 采集错误数据索引
        self.err_idx = []
        # 采集原始数据保存路径
        self.orin_data_filepath = None
        # 采集规则数据保存路径
        self.rule_data_filepath = None
        # 采集温升数据保存路径
        self.rpm_temp_data_filepath = None
        # 采集的平均转速
        self.avg_rpm = []
        # 一分钟内统计的60秒总转速
        self.total_rpm = 0
        # 是否显示原始采集数据
        self.show_orin = True
        # 是否从NC采集温度
        self.tem_from_nc = True
        # 从NC采集温升的寄存器个数
        self.nc_reg_num = None

        # 导入的数据
        self.data = None
        # 聚类结果
        self.cluster_res = None
        # 迭代筛选线程
        self.tsp_thread = None
        # 测点筛选结果
        self.tsp_res = None
        # 存储不同模型是否需要重新创建数据集的布尔值字典
        self.flag_need_create_new_datasets = dict()
        # 训练热误差模型所用数据集
        self.datasets = None

        # 五折交叉验证划分
        self.kf = None
        # 热误差预测模型
        self.err_model = None
        # 图模型的边索引
        self.edge_index = None
        # 训练结束，在测试数据集上表现最好的热误差模型
        self.best_err_model = None

        # 温度预测模型
        self.tem_model = None
        # 记录第一次获取到的绝对温度
        self.init_abs_temp = None
        # 记录最近三分钟温升(相对温度)
        self.his_rel_temp = deque(
            [[0 for _ in range(6)], [0 for _ in range(6)], [0 for _ in range(6)]], maxlen=3
        )
        # 温度模型预测的温升
        self.pred_temp_numpy = None
        # 热误差模型预测的热误差
        self.pred_err_numpy = None
        # 平均转速数组
        self.sampled_rpm = None
        # 代理模型拟合的参数
        self.coef_fit = None
        # 代理补偿定时器
        self.compen_timer = QTimer()
        # 更新间隔
        self.compen_interval = None
        # 当前更新参数次数计数
        self.compen_count = 0

        # 从保存的配置文件中读取配置
        self.ui_para = YamlHandler("res/config.yml")
        self.ui_para.read()

    def load_torch(self):
        self.loader_thread = ModuleLoaderThread()
        self.loader_thread.module_loaded.connect(self.signal_module_loaded)
        self.loader_thread.start()

    def on_close_window(self, config):
        self.ui_para.data = config
        self.ui_para.write()
        self.disconnect_nc()
        self.disconnect_tem_card()
        self.close_port()
        self.stop_compen()

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
            if self.tem_modbus_client is None:
                self.error_assert_temp_card_not_none.emit(
                    "选择从采集卡采集温度, 但是采集卡没有连接成功"
                )
                return
        else:
            self.tem_from_nc = True
            self.nc_reg_num = para["reg_num"]
            if self.nc_client is None:
                self.error_assert_nc_client_not_none.emit("选择从NC采集温度, 但是NC没有连接成功")
                return
        # 如果选择量表采集，判断是否连接量表
        if para["type"] == "量表停留":
            if self.serial_port_client is None:
                self.error_assert_serial_port_client_not_none.emit(
                    "采样规则选择量表停留采集, 但是量表没有打开"
                )
                return
        # 如果选择坐标停留，判断是否连接机床
        if para["type"] == "坐标停留":
            if self.nc_client is None:
                self.error_assert_nc_client_not_none.emit(
                    "采样规则选择坐标停留采集, 但是机床没有连接成功"
                )
                return
        # 开始采集前根据连接的设备插入对应的空数据队列
        if self.nc_client:
            nc_keys = ["nc_reg_g", "nc_axis_x", "nc_axis_y", "nc_axis_z", "nc_chan"]
            for key in nc_keys:
                self.orin_data[key] = deque(maxlen=MAX_DATA_LEN)
                self.rule_data[key] = deque()
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
        from src.core.func import extract_numbers

        # data: [counter, {key: value}], key: "nc_data", "card_temp", "error"

        # 更新界面采集时间信息
        self.signal_update_time.emit()
        # 分类并提取数据
        err_flag = False  # 采集失败标记
        for key, value in data[1].items():
            if key == "nc_data":
                if value is not None:
                    # 机床G寄存器除以10才是温度
                    reg_temp = [num / 10 for num in value[0][: self.nc_reg_num]]
                    # 记录初始温度
                    if self.init_abs_temp is None:
                        self.init_abs_temp = reg_temp
                    # 插入数据
                    self.orin_data["nc_reg_g"].append(reg_temp)
                    self.orin_data["nc_axis_x"].append(value[1])
                    self.orin_data["nc_axis_y"].append(value[2])
                    self.orin_data["nc_axis_z"].append(value[3])
                    self.orin_data["nc_chan"].append(extract_numbers(value[4]))
                else:
                    # 采集失败则自动复制最后一次的数据
                    err_flag = True
                    # 这里还要判断失败之前是否成功采集过数据
                    if self.orin_data["nc_reg_g"]:
                        self.orin_data["nc_reg_g"].append(self.orin_data["nc_reg_g"][-1])
                        self.orin_data["nc_axis_x"].append(self.orin_data["nc_axis_x"][-1])
                        self.orin_data["nc_axis_y"].append(self.orin_data["nc_axis_y"][-1])
                        self.orin_data["nc_axis_z"].append(self.orin_data["nc_axis_z"][-1])
                    self.orin_data["nc_chan"].append(self.orin_data["nc_chan"][-1])

                # 统计一分钟平均转速和温度变化
                self.save_rpm_temp_data()
            else:
                # FIXME: 只简单处理了采集机床信息失败的情况, 但是采集卡采集到[]的情况没有处理
                self.orin_data[key].append(extract_numbers(value))

        if err_flag:
            self.err_count += 1
            self.err_idx.append(self.data_collector_thread.counter)  # 记录发生错误的组数
        else:
            # 没有发生错误才保存采集的所有数据作为原始数据
            self.orin_count += 1
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.orin_data_filepath, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([current_time] + extract_numbers(data[1]))

        # print(f"orin_count: {self.orin_count}")
        # print(f"actual_count: {data[0]}")
        # print(f"err_idx: {self.err_idx}")

        # 发送显示原始数据信号
        if self.show_orin:
            self.signal_show_orin_data.emit()

    def save_rpm_temp_data(self) -> None:
        from src.core.func import extract_numbers

        self.total_rpm += self.orin_data["nc_chan"][-1][2]
        if self.data_collector_thread.counter % RPM_AVG_INTER == 0:
            self.avg_rpm.append(round(self.total_rpm / RPM_AVG_INTER))
            if self.tem_from_nc:
                temp = self.orin_data["nc_reg_g"][-1][: self.nc_reg_num]
            elif "card_temp" in self.orin_data:
                temp = self.orin_data["card_temp"][-1]
            # 更新维护最近的温升
            self.his_rel_temp.append(
                [abs_temp - init for abs_temp, init in zip(temp, self.init_abs_temp)]
            )
            # print(self.his_rel_temp)
            with open(self.rpm_temp_data_filepath, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(extract_numbers([self.avg_rpm[-1], temp]))
            self.total_rpm = 0

    def append_rule_data(self) -> None:
        from src.core.func import extract_numbers

        """
        插入规则数据, 写入CSV文件, 只会写入选择的温度来源数据
        """
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

    def get_cluster_res(self, method_name: str, tsp_num: int) -> None:
        """
        根据聚类方法和聚类数量获取聚类结果
        """
        match method_name:
            case "FCM算法":
                from src.core.func import get_fcm_cluster_res

                self.cluster_res = get_fcm_cluster_res(self.data.Tdata, tsp_num)
            case "K均值算法":
                from src.core.func import get_kmeans_cluster_res

                self.cluster_res = get_kmeans_cluster_res(self.data.Tdata, tsp_num)

    def get_tra_tsp_res(self, method_name: str) -> None:
        """
        根据选择的关联度方法获取温度敏感点
        """
        match method_name:
            case "灰色关联度":
                from src.core.func import gra

                self.tsp_res = gra(self.data.Xdata, self.data.Tdata, self.cluster_res)
            case "相关系数":
                from src.core.func import cor

                self.tsp_res = cor(self.data.Xdata, self.data.Tdata, self.cluster_res)

    def get_ga_tsp_res(self, pop_size: int, iters: int) -> None:
        # TODO: 在另一个线程中计算, 有没有必要
        """
        约束遗传算法求解温度敏感点
        ----
        - pop_size: 种群大小
        - iters: 迭代次数
        """
        from functools import partial
        from src.core.func import get_kfold_cv, loss_func
        from src.thread.ga_tsp_thread import GATSPThread

        cv = get_kfold_cv(5, self.data.Tdata)
        # 设置的损失函数
        loss = partial(
            loss_func,
            Tdata=self.data.Tdata,
            Xdata=self.data.Xdata,
            idx=self.cluster_res,
            cv=cv,
        )

        self.tsp_thread = GATSPThread(loss, self.data.Tdata.shape[1], pop_size, iters, self.cluster_res)
        self.tsp_thread.signal_tsp_result.connect(self.ga_tsp_done)
        self.tsp_thread.start()

    def ga_tsp_done(self, tsp_res):
        """
        将约束遗传算法求解的温度敏感点传递给主界面
        """
        self.tsp_res = tsp_res
        self.signal_ga_tsp_done.emit()

    def check_edited_tsp_text(self, tsp_res_text):
        is_valid, error_message = self.validate_tsp_input(tsp_res_text)
        if not is_valid:
            return False, error_message
        tsp_list = list(map(int, tsp_res_text.split(",")))
        return True, tsp_list

    @staticmethod
    def validate_tsp_input(input_text: str) -> tuple[bool, str]:
        if not input_text.strip():
            return False, "输入不能为空。"

        # 使用正则表达式检查输入格式是否正确（逗号分隔的整数）
        import re

        if not re.match(r"^(\d+,)*\d+$", input_text.strip()):
            return False, "输入格式不正确，请确保是逗号分隔的整数列表。"

        return True, ""

    def save_chosen_data(
        self, file_path: str, tsp_res_text: str, inter_num: int
    ) -> tuple[bool, str]:
        """
        保存选取的测点温度数据和热误差数据, 并且可以插值
        -----
        返回: 是否成功保存，保存信息
        """
        if self.data is None:
            return False, "请先导入数据"
        is_valid, message = self.check_edited_tsp_text(tsp_res_text)
        if not is_valid:
            return False, message
        else:
            tsp_list = message
            self.data.write_file(file_path, tsp_list, inter_num)
            return True, f"已保存至{file_path}"

    def mlr_fit(self, tsp_res_text: str) -> tuple[bool, list]:
        """
        拟合MLR模型
        -----
        返回: 是否成功拟合，拟合结果(拟合曲线, 测温点曲线, 截距, 系数, 均方根误差)
        """

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

    # TODO: 需要测试导入功能
    def send_coef(self, coef: list) -> None:
        """
        导入拟合一阶参数
        """
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

    def load_err_model(self, para) -> None:
        """
        导入热误差模型
        """
        # para中包含了模型类型、模型参数和文件路径
        import torch

        self.reset_model(model_type=para["model_type"], model_para=para)
        self.err_model.load_state_dict(torch.load(para["file_path"]))

    # 重置模型
    def reset_model(self, model_type: str, model_para: dict) -> None:
        """
        根据模型类型和对应模型参数实例化模型, 相当于重置模型
        ----
        - model_type: 模型类型
        - model_para: 模型参数
        """
        match model_type:
            case "GAT-LSTM":
                import torch
                from src.core.gat_lstm import GATLSTM

                self.edge_index = torch.tensor(model_para["edge_list"], dtype=torch.long)

                self.err_model = GATLSTM(
                    in_dim=1,
                    out_dim=1,
                    gat_hidden_dim=model_para["gnn_dim"],
                    lstm_hidden_dim=model_para["lstm_dim"],
                    num_nodes=model_para["num_nodes"],
                    heads=model_para["num_heads"],
                )
            case "BPNN":
                from src.core.bpnn import BPNN

                self.err_model = BPNN(
                    input_shape=model_para["input_shape"],
                    hidden_units=model_para["hidden_dim"],
                    output_shape=1,
                )

    def start_train(self, para: dict, tsp_res_text: str) -> None:
        """
        训练按钮点击产生的动作
        ----
        - para: 参数
        - tsp_res_text: 温度测点输入框字符串
        """
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

    def increase_train(self, para: dict, tsp_res_text: str) -> None:
        """
        增量训练， 与start_train的区别在于, 不需要重置模型
        ----
        - para
        - tsp_res_text: 温度测点输入框字符串
        """
        tsp_list = list(map(int, tsp_res_text.split(",")))
        model_type = para["model_type"]
        model_para = para["model_para"]
        if (
            model_type not in self.flag_need_create_new_datasets
            or self.flag_need_create_new_datasets[model_type]
        ):
            self.datasets = self.get_datasets(model_type, model_para, tsp_list)

        self.train_thread_start(model_type, para["train_para"])

    # 获取训练数据集
    def get_datasets(self, model_type: str, model_para: dict, tsp_list: list) -> None:
        """
        获取不同模型类型需要的数据集
        ----
        - model_type: 模型类型
        - model_para: 模型参数
        - tsp_list: 温度测点列表
        """
        from sklearn.model_selection import KFold

        self.kf = KFold(n_splits=5, shuffle=True)  # 5折交叉验证
        self.signal_get_datasets.emit()  # 提示正在准备数据集
        datasets = None
        match model_type:
            case "GAT-LSTM":
                import torch
                from src.core.graph_data import GraphData

                self.edge_index = torch.tensor(model_para["edge_list"], dtype=torch.long)
                for idx, array in enumerate(self.data.data_arrays):
                    dataset = GraphData(
                        array[:, tsp_list],
                        self.edge_index,
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

    def train_thread_start(self, model_type: str, train_para: dict) -> None:
        """
        启动训练线程
        ----
        - model_type: 模型类型
        - train_para: 训练参数
        """
        self.signal_start_train.emit()  # 通知GUI界面开始训练

        # 根据模型类型选择不同的dataloader
        match model_type:
            case "GAT-LSTM":
                from torch_geometric.loader import DataLoader as GeometricDataLoader

                dataloader = GeometricDataLoader
            case "BPNN":
                from torch.utils.data import DataLoader as TorchDataLoader

                dataloader = TorchDataLoader

        from src.thread.model_train_thread import ModelTrainThread

        self.thread_train = ModelTrainThread(
            model=self.err_model,
            datasets=self.datasets,
            dataloader=dataloader,
            kfold=self.kf,
            train_para=train_para,
        )

        self.thread_train.signal_train_val_loss.connect(self.signal_train_val_loss)
        self.thread_train.signal_train_finished.connect(self.signal_train_finished)
        self.thread_train.start()

    def save_model(self, file_path: str) -> None:
        """
        保存模型参数
        ----
        - file_path: 模型保存路径
        """
        import torch

        torch.save(self.err_model.state_dict(), file_path)

    def import_tem_model(self, para: dict) -> None:
        """
        导入温度预测模型
        ----
        - para["temp_nums"]
        - para["hidden_dim"]
        - para["num_layers"]
        - para["file_path"]
        """
        import torch
        from src.core.tem_pred import TemPred

        self.tem_model = TemPred(
            temp_nums=para["temp_nums"],
            hidden_dim=para["hidden_dim"],
            num_layers=para["num_layers"],
        )
        self.tem_model.load_state_dict(torch.load(para["file_path"]))

    def import_rpm_file(self, file_path: str) -> None:
        """
        导入采集的平均转速和温度数据文件, 提取出平均转速, 并保存在self.avg_rpm中
        -----
        - file_path: 文件路径
        """
        import numpy as np

        data = np.loadtxt(file_path, delimiter=",")
        self.sampled_rpm = data[1:, 0]  # 第一行的没有用, 从第二行开始

    def start_compen(self, para: dict) -> None:
        """
        开始代理补偿
        ----
        - para["degree"]:int 参数拟合阶数
        - para["interval"]: int 补偿间隔
        """
        self.compen_interval = para["interval"]
        # 在创建定时器时就执行一次, 不然要等一段时间才能开始补偿
        self.cal_para(para["degree"])
        self.compen_timer.timeout.connect(lambda: self.cal_para(para["degree"]))
        self.compen_timer.start(para["interval"] * 60_000)

    def stop_compen(self) -> None:
        """
        停止代理补偿
        """
        self.compen_timer.stop()

    def cal_para(self, degree: int) -> None:
        """
        计算一阶、二阶代理模型拟合参数
        ----
        - degree: 模型阶数
        """
        import torch
        from src.core.gat_lstm import GATLSTM
        from src.core.bpnn import BPNN
        from src.core.poly_least_squares import PolyLeastSquares

        # 历史温度变化趋势转换为tensor
        his_tem_tensor = torch.tensor(self.his_rel_temp, dtype=torch.float32).unsqueeze(0)
        print(f"his_tem_tensor: \n {his_tem_tensor}")
        # 未来平均转速转换为tensor
        start_idx = self.compen_count * self.compen_interval
        end_idx = (self.compen_count + 1) * self.compen_interval
        print(f"start_idx: {start_idx}, end_idx: {end_idx}")
        if end_idx <= len(self.sampled_rpm):
            rpm_tensor = torch.tensor(
                self.sampled_rpm[start_idx:end_idx], dtype=torch.float32
            ).unsqueeze(0)
        else:
            print("未来平均转速数据不足, 不再更新参数")
            return
        print(f"rpm_tensor.shape: {rpm_tensor.shape}")
        # 预测温度
        pred_temp = self.tem_model(his_tem_tensor, rpm_tensor)
        print(f"pred_temp.shape: {pred_temp.shape}")
        # 根据模型类型计算预测的热误差, 先用最简单的BPNN
        if isinstance(self.err_model, BPNN):
            pred_err = self.err_model(pred_temp)
        elif isinstance(self.err_model, GATLSTM):
            from torch_geometric.data import Data

            cat_temp = torch.cat((his_tem_tensor, pred_temp), dim=1).squeeze(0)
            err_list = []
            for i in range(pred_temp.shape[1]):
                input = cat_temp[i : i + 3, :].reshape(-1, 1)
                data = Data(x=input, edge_index=self.edge_index)
                err = self.err_model(data)
                err_list.append(err)
            pred_err = torch.cat(err_list, dim=0).unsqueeze(0)
        else:
            # 未知模型类型
            return

        print(f"pred_err.shape: {pred_err.shape}")

        # 去除多余维度并转换为numpy数组
        self.pred_temp_numpy = pred_temp.squeeze(0).detach().numpy()
        self.pred_err_numpy = pred_err.squeeze(0).detach().numpy()
        print(f"self.pred_temp_numpy.shape: {self.pred_temp_numpy.shape}")
        print(f"self.pred_err_numpy.shape: {self.pred_err_numpy.shape}")

        # FIXME: 换成绝对温度再去拟合

        fit_model = PolyLeastSquares(degree)
        fit_model.fit(self.pred_temp_numpy, self.pred_err_numpy)

        self.compen_count += 1

        self.coef_fit = fit_model.get_coefficients()
        print(f"self.coef_fit.shape: {self.coef_fit.shape}")

        self.signal_fit_coef.emit([degree, self.coef_fit])
