from PyQt5.QtCore import QThread, pyqtSignal as Signal, QTimer

from src.core.serial_port_reader import SerialPortReader
from src.core.modbus_tcp import ModbusTCP
from src.core.nc_link import NCLink


class DataCollectorThread(QThread):
    signal_data_collected = Signal(list)
    signal_sample_collected = Signal()

    def __init__(
        self,
        para,
        nc_client: NCLink,
        tem_modbus_client: ModbusTCP,
        serial_port_client: SerialPortReader,
    ):
        super().__init__()
        self.para = para
        self.nc_client = nc_client
        self.tem_modbus_client = tem_modbus_client
        self.serial_port_client = serial_port_client

        # 根据传入的客户端是否为None作为是否采集该数据的开关
        self.collect_nc_data_flag = False if self.nc_client is None else True
        self.collect_reg_data_flag = False if self.tem_modbus_client is None else True
        self.collect_serial_data_flag = False if self.serial_port_client is None else True

        self.counter = 0
        self.once_rec_date = dict()
        self.query = self.get_query()

        self.coordinate_index = None
        self.axis_index = None
        self.target_val = None
        self.range = None
        self.timer_inter = None
        self.parse_para()

        self.last_val = 0.0
        self.repeat_times = 0
        self.need_new_sample = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.get_all_data)
        self.timer.start(1000)

    def parse_para(self):
        if self.para["type"] == "坐标停留":
            self.coordinate_index = 0 if self.para["coordinate"] == "机床实际" else 1
            if self.para["axis"] == "X轴":
                self.axis_index = 1
            elif self.para["axis"] == "Y轴":
                self.axis_index = 2
            elif self.para["axis"] == "Z轴":
                self.axis_index = 3
            if "axis_val" in self.para:
                self.target_val = self.para["axis_val"]
        elif self.para["type"] == "量表停留":
            self.target_val = self.para["init_pos"]
            self.range = self.para["range"]
        else:
            self.timer_inter = self.para["time"]

    # 采集所有数据
    def get_all_data(self):
        # 采集所有连接成功的设备的数据
        self.counter += 1
        if self.collect_nc_data_flag:
            rec_nc_data = self.nc_client.query(self.query)
            # 采集失败返回None
            self.once_rec_date["nc_data"] = rec_nc_data
        if self.collect_reg_data_flag:
            rec_temp = self.tem_modbus_client.read_temperature()
            # 采集失败返回[]
            self.once_rec_date["card_temp"] = rec_temp
        if self.collect_serial_data_flag:
            indict = self.serial_port_client.read()
            self.once_rec_date["error"] = indict
        # 如果这次满足规则采样要求
        if self.check_save():
            self.signal_sample_collected.emit()
        self.signal_data_collected.emit([self.counter, self.once_rec_date])

    # 判断是否满足采样条件
    def check_save(self) -> bool:
        save_flag = False
        if self.para["type"] == "定时采集":
            if self.counter % self.timer_inter == 0:
                save_flag = True
        elif self.para["type"] == "坐标停留" and self.once_rec_date["nc_data"]:
            # NClink读取的坐标值精度比系统显示精度多一位, 这里设定一个小范围
            val = self.once_rec_date["nc_data"][self.axis_index][self.coordinate_index]
            # print(val)
            save_flag = self.check_range_repeat(val, 0.0001)
        elif self.para["type"] == "量表停留":
            val = self.once_rec_date["error"]
            save_flag = self.check_range_repeat(val, self.range)

        return save_flag

    # 判断一个值是否在目标值设定的波动区间停留超过3次
    def check_range_repeat(self, val, range) -> bool:
        save_flag = False
        if abs(val - self.target_val) <= range:
            if abs(self.last_val - self.target_val) > range:
                self.repeat_times = 0
                self.need_new_sample = True
            elif self.need_new_sample and self.repeat_times >= 3:
                self.need_new_sample = False
                self.repeat_times = 0
                save_flag = True
            self.repeat_times += 1
        self.last_val = val
        return save_flag

    @staticmethod
    def get_query() -> list:
        # 每个轴采集
        items = [
            "REG_G:3080-3100",
            "AXIS_0:40,41,43,49,53",
            "AXIS_1:40,41,43,49,53",
            "AXIS_2:40,41,43,49,53",
            "CHAN_0:27,32,47",
        ]
        return items

    def stop(self):
        self.timer.stop()
        self.quit()
        self.wait()
