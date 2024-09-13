from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QLineEdit,
    QComboBox,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QFormLayout,
    QGroupBox,
)

from src.components import ComboOptions


class SampleRuleSetting(QGroupBox):
    signal_sample_save_path = Signal(str)
    signal_start_sample = Signal(dict)
    signal_stop_sample = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("采样规则")
        self.vLayout = QVBoxLayout(self)
        self.vLayout.setSpacing(10)

        self.sample_widget = ComboOptions()
        self.vLayout.addWidget(self.sample_widget)

        self.choose_coordinate = QComboBox()
        self.choose_coordinate.addItem("机床实际")
        self.choose_coordinate.addItem("相对实际")
        self.choose_axis = QComboBox()
        self.choose_axis.addItem("Z轴")
        self.choose_axis.addItem("Y轴")
        self.choose_axis.addItem("X轴")
        self.edit_axis_val = QLineEdit("0.0000")
        self.edit_init_pos = QLineEdit("0.0000")
        self.edit_range = QLineEdit("0.0010")
        self.edit_range.setValidator(QDoubleValidator(0.0, 100.0, 4))  # 确保阈值大于0且为float
        self.edit_timer = QLineEdit("5")
        self.edit_timer.setValidator(QIntValidator(1, 36000))  # 确保间隔大于1且为int

        self.sample_widget.add_item_input(
            item="坐标停留",
            para_labels=["坐标系:", "坐标轴:", "坐标值:"],
            para_inputs=[self.choose_coordinate, self.choose_axis, self.edit_axis_val],
        )
        self.sample_widget.add_item_input(
            item="量表停留",
            para_labels=["初始位置:", "±阈值:"],
            para_inputs=[self.edit_init_pos, self.edit_range],
        )
        self.sample_widget.add_item_input(
            item="定时采集", para_labels=["时长(秒):"], para_inputs=[self.edit_timer]
        )
        self.sample_widget.combo_box.setCurrentIndex(2)

        self.tem_from_layout = QFormLayout()
        self.combo_tem_from = QComboBox()
        self.combo_tem_from.addItem("寄存器")
        self.combo_tem_from.addItem("采集卡")
        self.combo_tem_from.currentTextChanged.connect(self.add_reg_num_edit)
        self.tem_from_layout.addRow(QLabel("采集温度:"), self.combo_tem_from)
        self.edit_reg_num = QLineEdit("6")
        self.tem_from_layout.addRow("温度数量:", self.edit_reg_num)
        self.vLayout.addLayout(self.tem_from_layout)

        # self.check_rpm_tem = QCheckBox("转速温度")
        # self.check_compen_val = QCheckBox("轴补偿值")
        # self.other_para_layout = QFormLayout()
        # self.other_para_layout.addRow("其他数据:", self.check_rpm_tem)
        # self.other_para_layout.addRow("", self.check_compen_val)
        # self.vLayout.addLayout(self.other_para_layout)

        self.btn_layout = QHBoxLayout()
        self.btn_sample_data_save_path = QPushButton("保存目录")
        self.btn_start_sample = QPushButton("开始采集")
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_sample_data_save_path)
        self.btn_layout.addWidget(self.btn_start_sample)
        self.vLayout.addLayout(self.btn_layout)

        self.btn_sample_data_save_path.clicked.connect(self.on_btn_save_path)
        self.btn_start_sample.clicked.connect(self.on_btn_start_sample)

    def add_reg_num_edit(self, text):
        if text == "寄存器":
            self.edit_reg_num = QLineEdit("6")
            self.tem_from_layout.addRow("温度数量:", self.edit_reg_num)
        else:
            if self.tem_from_layout.rowCount() > 1:
                self.tem_from_layout.removeRow(1)

    def on_btn_save_path(self):
        file_filter = "采样数据 (*.csv)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存数据文件到", "", file_filter
        )
        if file_path:
            if selected_filter == "采样数据 (*.csv)" and not file_path.endswith(".csv"):
                file_path += ".pth"
            self.signal_sample_save_path.emit(file_path)

    def on_btn_start_sample(self):
        para = dict()
        match self.sample_widget.get_cur_combo():
            case "坐标停留":
                para["type"] = "坐标停留"
                para["coordinate"] = self.choose_coordinate.currentText()
                para["axis"] = self.choose_axis.currentText()
                para["axis_val"] = float(self.edit_axis_val.text())
            case "量表停留":
                para["type"] = "量表停留"
                para["init_pos"] = float(self.edit_init_pos.text())
                para["range"] = float(self.edit_range.text())
            case "定时采集":
                para["type"] = "定时采集"
                para["time"] = int(self.edit_timer.text())  # 单位为秒
        match self.combo_tem_from.currentText():
            case "采集卡":
                para["tem_from"] = "采集卡"
            case "寄存器":
                para["reg_num"] = int(self.edit_reg_num.text())
                para["tem_from"] = "寄存器"
        self.signal_start_sample.emit(para)

    def set_btn_stop_sample(self):
        self.btn_start_sample.clicked.disconnect()
        self.btn_start_sample.setText("停止采集")
        self.btn_start_sample.clicked.connect(self.on_btn_stop_sample)

    def on_btn_stop_sample(self):
        self.signal_stop_sample.emit()
        self.btn_start_sample.clicked.disconnect()
        self.btn_start_sample.setText("开始采集")
        self.btn_start_sample.clicked.connect(self.on_btn_start_sample)
