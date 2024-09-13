from PyQt5.QtWidgets import (
    QLabel,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QPushButton,
    QMessageBox,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import pyqtSignal as Signal


class SerialPortSetting(QGroupBox):
    signal_open_port = Signal(list)
    signal_close_port = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("串口设置")
        vLayout = QVBoxLayout(self)
        vLayout.setSpacing(10)

        self.edit_port_num = QLineEdit()
        self.edit_port_num.setPlaceholderText("量表输出端口")
        hLayout_port_num = QHBoxLayout()
        hLayout_port_num.addWidget(QLabel("端口号:"))
        hLayout_port_num.addWidget(self.edit_port_num)
        self.edit_baud_rate = QLineEdit()
        self.edit_baud_rate.setValidator(QIntValidator())
        self.edit_baud_rate.setPlaceholderText("对应波特率")
        hLayout_baud_rate = QHBoxLayout()
        hLayout_baud_rate.addWidget(QLabel("波特率:"))
        hLayout_baud_rate.addWidget(self.edit_baud_rate)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addStretch()
        self.btn_connect = QPushButton("连接测试", self)
        self.btn_layout.addWidget(self.btn_connect)

        vLayout.addLayout(hLayout_port_num)
        vLayout.addLayout(hLayout_baud_rate)
        vLayout.addLayout(self.btn_layout)

        self.btn_connect.clicked.connect(self.on_open_serial_port)

    def on_open_serial_port(self):
        try:
            com: str = self.edit_port_num.text()
            baud_rate: int = int(self.edit_baud_rate.text())
            self.signal_open_port.emit([com, baud_rate])
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入正确的参数")

    def set_btn_close(self):
        self.btn_connect.setText("关闭串口")
        self.btn_connect.disconnect()
        self.btn_connect.clicked.connect(self.on_close_serial_port)

    def on_close_serial_port(self):
        self.signal_close_port.emit()
        self.btn_connect.setText("重新打开")
        self.btn_connect.disconnect()
        self.btn_connect.clicked.connect(self.on_open_serial_port)

    def vis_config(self, port_num, baud_rate):
        self.edit_port_num.setText(port_num)
        self.edit_baud_rate.setText(str(baud_rate))
