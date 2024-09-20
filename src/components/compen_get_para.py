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
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("参数拟合")

        self.btn_layout = QHBoxLayout()
        self.btn_linear_fit = QPushButton("一阶拟合")
        self.btn_quadratic_fit = QPushButton("二阶拟合")
        self.btn_layout.addWidget(self.btn_linear_fit)
        self.btn_layout.addWidget(self.btn_quadratic_fit)

        self.fit_res_layout = QVBoxLayout()

        self.hLayout = QHBoxLayout()
        self.check_edit = QCheckBox("编辑")
        self.check_negative = QCheckBox("取反")
        self.btn_send_para = QPushButton("导入参数")
        self.hLayout.addStretch()
        self.hLayout.addWidget(self.check_edit)
        self.hLayout.addWidget(self.check_negative)
        self.hLayout.addWidget(self.btn_send_para)

        self.vLayout = QVBoxLayout(self)
        self.vLayout.addLayout(self.btn_layout)
        self.vLayout.addWidget(QLabel("拟合结果:"))
        self.vLayout.addLayout(self.fit_res_layout)
        self.vLayout.addLayout(self.hLayout)

    def show_fit_para(self, degree, coef):
        print(coef)
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
