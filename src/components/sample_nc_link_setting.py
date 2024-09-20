from PyQt5.QtWidgets import (
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QFormLayout,
    QGroupBox,
    QMessageBox,
)
from PyQt5.QtCore import pyqtSignal as Signal, Qt
from PyQt5.QtGui import QIntValidator


class NCLinkSetting(QGroupBox):
    signal_connect_nc = Signal(list)
    signal_disconnect_nc = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTitle("机床连接设置")
        self.vlayout = QVBoxLayout(self)
        self.vlayout.setSpacing(10)

        self.edit_mqtt_ip = QLineEdit(self)
        self.edit_mqtt_ip.setPlaceholderText("上位机地址或本机地址")
        self.edit_mqtt_port = QLineEdit(self)
        self.edit_mqtt_port.setPlaceholderText("默认1883")
        self.edit_mqtt_port.setValidator(QIntValidator())
        self.edit_nc_SN = QLineEdit(self)
        self.edit_nc_SN.setPlaceholderText("机床序列号")

        self.formLayout1 = QFormLayout()
        self.formLayout1.setContentsMargins(0, 0, 0, 0)
        self.formLayout1.setSpacing(10)
        self.formLayout1.addRow(QLabel("MQTT地址:"), self.edit_mqtt_ip)
        self.formLayout1.addRow(QLabel("MQTT端口:"), self.edit_mqtt_port)
        self.formLayout1.addRow(QLabel("机床SN码:"), self.edit_nc_SN)

        self.btn_connect_nc = QPushButton("连接机床")

        self.vlayout.addLayout(self.formLayout1)
        self.vlayout.addWidget(self.btn_connect_nc, 0, Qt.AlignRight)


        self.btn_connect_nc.clicked.connect(self.on_btn_connect_nc)

    def on_btn_connect_nc(self):
        mqtt_ip: str = self.edit_mqtt_ip.text()
        nc_sn: str = self.edit_nc_SN.text()
        try:
            mqtt_port: int = int(self.edit_mqtt_port.text())
            self.signal_connect_nc.emit([mqtt_ip, mqtt_port, nc_sn])
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入正确的参数")

    def set_btn_disconnect(self):
        self.btn_connect_nc.setText("断开连接")
        self.btn_connect_nc.disconnect()
        self.btn_connect_nc.clicked.connect(self.on_disconnect_nc)

    def on_disconnect_nc(self):
        self.signal_disconnect_nc.emit()
        self.btn_connect_nc.setText("重新连接")
        self.btn_connect_nc.disconnect()
        self.btn_connect_nc.clicked.connect(self.on_btn_connect_nc)
