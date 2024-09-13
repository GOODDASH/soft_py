from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QFormLayout,
    QGroupBox,
)


class TspConfig(QGroupBox):
    signal_tra_tsp = Signal(list)
    signal_ga_tsp = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setTitle("筛选设置")
        self.vLayout = QVBoxLayout(self)
        self.vLayout.setSpacing(10)

        self.fLayout = QFormLayout()

        self.edit_tsp_num = QLineEdit()

        self.combo_cluster_method = QComboBox()
        self.combo_cluster_method.addItem("FCM算法")
        self.combo_cluster_method.addItem("K均值算法")
        self.combo_cor_method = QComboBox()
        self.combo_cor_method.addItem("灰色关联度")
        self.combo_cor_method.addItem("相关系数")

        self.edit_size = QLineEdit()
        self.edit_epoch = QLineEdit()

        self.fLayout.addRow(QLabel("测点数量:"), self.edit_tsp_num)
        self.fLayout.addRow(QLabel("聚类方法:"), self.combo_cluster_method)
        self.fLayout.addRow(QLabel("关联度方法:"), self.combo_cor_method)
        self.fLayout.addRow(QLabel("种群数量:"), self.edit_size)
        self.fLayout.addRow(QLabel("迭代次数:"), self.edit_epoch)

        self.hLayout2 = QHBoxLayout()
        self.btn_tra_tsp = QPushButton("传统筛选")
        self.btn_ga_tsp = QPushButton("迭代筛选")
        self.hLayout2.addWidget(self.btn_tra_tsp)
        self.hLayout2.addWidget(self.btn_ga_tsp)

        self.vLayout.addStretch()
        self.vLayout.addLayout(self.fLayout)
        self.vLayout.addLayout(self.hLayout2)
        self.vLayout.addStretch()

        self.btn_tra_tsp.clicked.connect(self.on_btn_tra_tsp)
        self.btn_ga_tsp.clicked.connect(self.on_btn_ga_tsp)

    def on_btn_tra_tsp(self):
        para = [
            int(self.edit_tsp_num.text()),
            self.combo_cluster_method.currentText(),
            self.combo_cor_method.currentText(),
        ]
        self.signal_tra_tsp.emit(para)

    def on_btn_ga_tsp(self):
        para = [
            int(self.edit_tsp_num.text()),
            self.combo_cluster_method.currentText(),
            int(self.edit_size.text()),
            int(self.edit_epoch.text()),
        ]
        self.signal_ga_tsp.emit(para)
