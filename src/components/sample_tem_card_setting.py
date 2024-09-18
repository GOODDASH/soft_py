from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
)
from PyQt5.QtGui import QIcon, QIntValidator
from PyQt5.QtCore import pyqtSignal as Signal, QSize


class TemCardSetting(QGroupBox):
    signal_connect_tem_card = Signal(list)
    signal_disconnect_tem_card = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.input_ips = []
        self.input_slave = []
        self.input_port = []
        self.input_addr = []
        self.input_reg_num = []

        self.setTitle("采集卡设置")
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        self.vLayout_tab_tem_card = QVBoxLayout()
        self.vLayout_tab_tem_card.setSpacing(0)
        self.btn_layout = QHBoxLayout()
        self.btn_add_card = QPushButton()
        self.btn_add_card.setObjectName("btn_add_card")
        self.btn_add_card.setIcon(QIcon("src/icons/add.png"))
        self.btn_add_card.setIconSize(QSize(23, 23))
        self.btn_add_card.setFixedSize(38, 38)
        self.btn_add_card.setToolTip("添加采集卡")
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_add_card)
        self.btn_layout.addStretch()
        self.vLayout_tab_tem_card.addLayout(self.btn_layout)
        self.btn_add_card.clicked.connect(lambda: self.add_input_row())

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addStretch()
        self.btn_connect = QPushButton("连接测试", self)
        self.btn_layout.addWidget(self.btn_connect)

        self.layout.addLayout(self.vLayout_tab_tem_card)
        self.layout.addLayout(self.btn_layout)

        self.btn_connect.clicked.connect(self.on_connect_tem_card)

    def on_connect_tem_card(self):
        self.signal_connect_tem_card.emit(self.get_input_paras())

    def set_btn_disconnect(self):
        self.btn_connect.setText("断开连接")
        self.btn_connect.disconnect()
        self.btn_connect.clicked.connect(self.on_disconnect_tem_card)

    def on_disconnect_tem_card(self):
        self.signal_disconnect_tem_card.emit()
        self.btn_connect.setText("重新连接")
        self.btn_connect.disconnect()
        self.btn_connect.clicked.connect(self.on_connect_tem_card)

    def add_input_row(self, text=None):
        """添加采集卡ip地址输入框"""
        label = QLabel("IP:")
        if text is not None:
            edit_ip = QLineEdit(text)
        else:
            edit_ip = QLineEdit()
            edit_ip.setPlaceholderText("采集卡IP地址")

        button1 = QPushButton()
        button1.setToolTip("更多设置")
        button1.setIcon(QIcon("src/icons/more.png"))
        button1.setIconSize(QSize(24, 24))
        button1.setFixedSize(35, 35)

        button2 = QPushButton()
        button2.setToolTip("删除")
        button2.setIcon(QIcon("src/icons/x.png"))
        button2.setIconSize(QSize(27, 27))
        button2.setFixedSize(35, 35)

        vlayout = QVBoxLayout()
        vlayout.setSpacing(0)

        line_widget = QWidget()
        horizontalLayout = QHBoxLayout(line_widget)
        horizontalLayout.addWidget(label)
        horizontalLayout.addWidget(edit_ip)
        horizontalLayout.addWidget(button1)
        horizontalLayout.addWidget(button2)

        detail_widget = QWidget()
        detail_widget.setObjectName("detail_widget")
        detail_widget.setVisible(False)
        flayout = QFormLayout(detail_widget)
        int_validator = QIntValidator()
        edit_slave = QLineEdit("255")
        edit_port = QLineEdit("502")
        edit_addr = QLineEdit("620")
        edit_reg_num = QLineEdit("9")
        edit_slave.setValidator(int_validator)
        edit_port.setValidator(int_validator)
        edit_addr.setValidator(int_validator)
        edit_reg_num.setValidator(int_validator)
        flayout.addRow("从站号:", edit_slave)
        flayout.addRow("端口号:", edit_port)
        flayout.addRow("起始地址:", edit_addr)
        flayout.addRow("采集数量:", edit_reg_num)
        flayout.setContentsMargins(30, 10, 30, 10)

        vlayout.addWidget(line_widget)
        vlayout.addWidget(detail_widget)

        button1.clicked.connect(lambda: self.show_detail_edits(detail_widget))
        button2.clicked.connect(lambda: self.remove_input_row(vlayout, edit_ip))

        self.vLayout_tab_tem_card.insertLayout(len(self.input_ips), vlayout)
        self.input_ips.append(edit_ip)
        self.input_slave.append(edit_slave)
        self.input_port.append(edit_port)
        self.input_addr.append(edit_addr)
        self.input_reg_num.append(edit_reg_num)

    def show_detail_edits(self, widget: QWidget):
        cur_vis = widget.isVisible()
        widget.setVisible(not cur_vis)

    def remove_input_row(self, layout, edit_ip):
        """移除采集卡地址"""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.vLayout_tab_tem_card.removeItem(layout)
        if edit_ip in self.input_ips:
            self.input_ips.remove(edit_ip)

    def get_input_paras(self):
        """返回采集卡ip的所有设置"""
        para = [
            (ip.text(), int(slave.text()), int(port.text()), int(addr.text()), int(reg_num.text()))
            for ip, slave, port, addr, reg_num in zip(
                self.input_ips,
                self.input_slave,
                self.input_port,
                self.input_addr,
                self.input_reg_num,
            )
        ]
        return para

    def get_input_ips(self):
        return [ip.text() for ip in self.input_ips]

    def vis_config(self, ips: list):
        for ip in ips:
            self.add_input_row(text=ip)

    # 只保存每个采集卡的IP地址
    def update_config(self, config):
        config["ips"] = self.get_input_ips()
        return config


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    ui = TemCardSetting()
    ui.resize(600, 300)
    config = {"ips": ["192.168.201", "192.168.202", "192.168.203"]}
    ui.vis_config(config["ips"])
    print(ui.update_config(config))
    ui.show()
    sys.exit(app.exec())
