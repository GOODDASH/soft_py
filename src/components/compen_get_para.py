from PyQt5.QtCore import pyqtSignal as Signal
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
    QGroupBox,
)


class CompenGetPara(QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("代理参数")
        
        self.btn_cal_para = QPushButton("计算代理模型参数")
        
        self.vLayout = QVBoxLayout(self)
        self.vLayout.addWidget(self.btn_cal_para)
        
        
