from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal as Signal

from src.components import (
    NCLinkSetting,
    TemCardSetting,
    SerialPortSetting,
    SampleRuleSetting,
    MultiPlotWidget,
)


class Sample(QWidget):
    signal_connect_nc = Signal(list)
    signal_disconnect_nc = Signal()
    signal_connect_tem_card = Signal(list)
    signal_disconnect_tem_card = Signal()
    signal_open_port = Signal(list)
    signal_close_port = Signal()

    signal_sample_save_path = Signal(str)
    signal_start_sample = Signal(dict)
    signal_change_orin_sample = Signal()
    signal_stop_sample = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.nc_link_widget = NCLinkSetting(self)
        self.tem_card_widget = TemCardSetting(self)
        self.serial_port_widget = SerialPortSetting(self)
        self.sample_rule_widget = SampleRuleSetting(self)

        self.sample_setting_area = QScrollArea()
        self.sample_setting_area.setWidgetResizable(True)
        self.sample_setting_area.setMinimumWidth(300)
        self.sample_widget = QWidget()
        self.sample_widget_layout = QVBoxLayout(self.sample_widget)
        self.sample_widget_layout.setSpacing(10)
        self.sample_widget_layout.addWidget(self.nc_link_widget)
        self.sample_widget_layout.addWidget(self.tem_card_widget)
        self.sample_widget_layout.addWidget(self.serial_port_widget)
        self.sample_widget_layout.addWidget(self.sample_rule_widget)
        self.sample_widget_layout.addStretch()
        self.sample_setting_area.setWidget(self.sample_widget)

        self.plot_area = QScrollArea()
        self.plot_widget = MultiPlotWidget()
        self.plot_widget.add_switch_orin_sample_btn()
        self.plot_area.setWidgetResizable(True)
        self.plot_area.setWidget(self.plot_widget)

        self.layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.sample_setting_area)
        self.splitter.addWidget(self.plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.layout.addWidget(self.splitter)

        self.connect_slots()

    def connect_slots(self):
        self.nc_link_widget.signal_connect_nc.connect(self.signal_connect_nc)
        self.nc_link_widget.signal_disconnect_nc.connect(self.signal_disconnect_nc)
        self.tem_card_widget.signal_connect_tem_card.connect(self.signal_connect_tem_card)
        self.tem_card_widget.signal_disconnect_tem_card.connect(self.signal_disconnect_tem_card)
        self.serial_port_widget.signal_open_port.connect(self.signal_open_port)
        self.serial_port_widget.signal_close_port.connect(self.signal_close_port)
        self.sample_rule_widget.signal_sample_save_path.connect(self.signal_sample_save_path)
        self.sample_rule_widget.signal_start_sample.connect(self.signal_start_sample)
        self.plot_widget.signal_change_orin_sample.connect(self.signal_change_orin_sample)
        self.sample_rule_widget.signal_stop_sample.connect(self.signal_stop_sample)

    def set_canvas_color(self, color):
        self.plot_widget.set_canvas_color(color)

    def vis_config(self, config: dict):
        self.nc_link_widget.edit_mqtt_ip.setText(config["mqtt_ip"])
        self.nc_link_widget.edit_mqtt_port.setText(str(config["mqtt_port"]))
        self.nc_link_widget.edit_nc_SN.setText(config["SN"])
        self.tem_card_widget.vis_config(config["ips"])
        self.serial_port_widget.vis_config(config["port_num"], config["baud_rate"])

    def update_config(self, config: dict) -> dict:
        config["mqtt_ip"] = self.nc_link_widget.edit_mqtt_ip.text()
        config["mqtt_port"] = int(self.nc_link_widget.edit_mqtt_port.text())
        config["SN"] = self.nc_link_widget.edit_nc_SN.text()
        config["ips"] = self.tem_card_widget.get_input_ips()
        config["port_num"] = self.serial_port_widget.edit_port_num.text()
        config["baud_rate"] = int(self.serial_port_widget.edit_baud_rate.text())

        return config


# .\.env\Scripts\python.exe -m src.pages.sample
if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt
    from PyQt5.QtWidgets import QApplication

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.unicode_minus"] = False

    app = QApplication(sys.argv)
    sample = Sample()
    sample.resize(1200, 800)
    sample.show()
    app.exec()
