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


class CompenTemModel(QGroupBox):
    signal_import_tem_model = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("温度模型")

        self.edit_temp_nums = QLineEdit("6")
        self.edit_hidden_dim = QLineEdit("36")
        self.edit_num_layers = QLineEdit("2")

        self.fLayout = QFormLayout()
        self.fLayout.addRow("测点个数:", self.edit_temp_nums)
        self.fLayout.addRow("隐藏维度:", self.edit_hidden_dim)
        self.fLayout.addRow("网络层数:", self.edit_num_layers)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.addStretch()
        self.btn_import = QPushButton("导入模型")
        self.btn_layout.addWidget(self.btn_import)

        self.layout = QVBoxLayout(self)
        self.layout.addLayout(self.fLayout)
        self.layout.addLayout(self.btn_layout)

    def on_btn_import(self):
        file_filter = "Pytorch File(*.pth);;All files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", file_filter)
        if file_path:
            signal_para = self.get_model_para()
            signal_para["file_path"] = file_path
            self.signal_import_tem_model.emit(signal_para)

    def get_model_para(self):
        model_para = dict()
        model_para["temp_nums"] = int(self.edit_temp_nums.text())
        model_para["hidden_dim"] = int(self.edit_hidden_dim.text())
        model_para["num_layers"] = int(self.edit_num_layers.text())
        return model_para