from PyQt5.QtCore import pyqtSignal as Signal, Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QStackedWidget,
    QMessageBox,
    QCheckBox,
    QGroupBox,
)


class CompenGetPara(QGroupBox):
    signal_start_compen = Signal(dict)
    signal_stop_compen = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("代理补偿")

        self.fLayout = QFormLayout()
        self.combo_fit_degree = QComboBox()
        self.combo_fit_degree.addItems(["一阶拟合", "二阶拟合"])
        self.inter_hLayout = QHBoxLayout()
        self.edit_fit_inter = QLineEdit("10")
        self.edit_fit_inter.setPlaceholderText("模型参数更新间隔")
        self.inter_hLayout.addWidget(self.edit_fit_inter)
        self.inter_hLayout.addWidget(QLabel("min"))
        self.inter_hLayout.setContentsMargins(0, 0, 0, 0)
        self.fLayout.addRow("代理阶数:", self.combo_fit_degree)
        self.fLayout.addRow("更新间隔:", self.inter_hLayout)

        self.btn_start_compen = QPushButton("开始补偿")
        self.btn_start_compen.clicked.connect(self.on_start_compen)

        self.fit_res_layout = QVBoxLayout()

        self.vLayout = QVBoxLayout(self)
        self.vLayout.setSpacing(10)
        self.vLayout.addLayout(self.fLayout)
        self.vLayout.addWidget(self.btn_start_compen, 0, Qt.AlignRight)
        self.vLayout.addLayout(self.fit_res_layout)

    def on_start_compen(self):
        self.btn_start_compen.clicked.disconnect()
        self.btn_start_compen.setText("停止补偿")
        self.btn_start_compen.clicked.connect(self.on_stop_compen)
        degree = 1 if self.combo_fit_degree.currentText() == "一阶拟合" else 2
        interval = int(self.edit_fit_inter.text())
        self.signal_start_compen.emit({"degree": degree, "interval": interval})

    def on_stop_compen(self):
        self.btn_start_compen.clicked.disconnect()
        self.btn_start_compen.setText("开始补偿")
        self.btn_start_compen.clicked.connect(self.on_start_compen)
        self.signal_stop_compen.emit()

    def show_fit_para(self, degree, coef):
        # 清空上次
        while self.fit_res_layout.count():
            item = self.fit_res_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 先计算拟合的温度传感器数量
        num_sensor = (coef.shape[0] - 1) / degree

        widget = QWidget()
        formlayout = QFormLayout(widget)
        for i, c in enumerate(coef):
            label = QLineEdit(f"{c:.4f}")
            label.setFixedWidth(120)
            label.setEnabled(False)
            if i == 0:
                formlayout.addRow("C: ", label)
            else:
                de = int((i - 1) // num_sensor + 1)
                idx = int((i - 1) % num_sensor + 1)
                header = QLabel(f"T<sub>{idx}</sub><sup>{de}</sup>: ")
                formlayout.addRow(header, label)
        self.fit_res_layout.addWidget(widget, 0, Qt.AlignHCenter)
