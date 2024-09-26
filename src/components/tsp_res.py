from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFormLayout,
    QFileDialog,
    QGroupBox,
)


class TspRes(QGroupBox):
    signal_saved_data_path = Signal(list)
    signal_send_coef = Signal(list)

    def __init__(self, parent=None):
        super(TspRes, self).__init__(parent)

        self.setTitle("筛选结果")

        # 存储多元线性回归拟合结果
        self.edit_mlr_paras = []
        

        self.vLayout = QVBoxLayout(self)
        self.vLayout.setSpacing(10)

        self.fLayout = QFormLayout()
        self.edit_tsp = QLineEdit(self)
        self.edit_tsp.setPlaceholderText("可手动更改")
        self.edit_interpolate_num = QLineEdit("0", self)
        self.edit_interpolate_num.setValidator(QIntValidator(0, 100))
        self.fLayout.addRow("筛选结果:", self.edit_tsp)
        self.fLayout.addRow("插值数量:", self.edit_interpolate_num)

        self.btn_layout = QHBoxLayout()
        self.btn_save_data = QPushButton("保存数据")
        self.btn_save_data.setEnabled(False)
        self.btn_save_data.clicked.connect(self.choose_save_path)
        self.btn_mlr_fit = QPushButton("一阶拟合")
        self.btn_mlr_fit.setEnabled(False)
        self.fit_res_layout = QVBoxLayout()
        self.btn_layout.addWidget(self.btn_save_data)
        self.btn_layout.addWidget(self.btn_mlr_fit)

        self.hLayout4 = QHBoxLayout()
        self.check_edit = QCheckBox("编辑")
        self.check_negative = QCheckBox("取反")
        self.btn_send_para = QPushButton("导入参数")
        self.btn_send_para.setEnabled(False)
        self.hLayout4.addStretch()
        self.hLayout4.addWidget(self.check_edit)
        self.hLayout4.addWidget(self.check_negative)
        self.hLayout4.addWidget(self.btn_send_para)
        self.check_edit.stateChanged.connect(self.change_editable)
        self.check_negative.stateChanged.connect(self.change_negative)
        self.btn_send_para.clicked.connect(self.on_send_coef)

        self.vLayout.addStretch()
        self.vLayout.addLayout(self.fLayout)
        self.vLayout.addLayout(self.btn_layout)
        self.vLayout.addWidget(QLabel("拟合结果:"))
        self.vLayout.addLayout(self.fit_res_layout)
        self.vLayout.addLayout(self.hLayout4)
        self.vLayout.addStretch()

    def choose_save_path(self):
        file_filter = (
            "CSV files (*.csv);;Excel files (*.xls *.xlsx);;Text files (*.txt);;All files (*.*)"
        )
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存文件到", "", file_filter
        )

        if file_path:
            if selected_filter == "CSV files (*.csv)" and not file_path.endswith(".csv"):
                file_path += ".csv"
            elif selected_filter == "Excel files (*.xls *.xlsx)" and not (
                file_path.endswith(".xls") or file_path.endswith(".xlsx")
            ):
                file_path += ".xlsx"
            elif selected_filter == "Text files (*.txt)" and not file_path.endswith(".txt"):
                file_path += ".txt"

            self.signal_saved_data_path.emit([file_path, int(self.edit_interpolate_num.text())])

    def on_send_coef(self):
        self.signal_send_coef.emit(self.get_intercept_coef())

    def show_mlr_fit_res(self, intercept: float, coef: list[float], rmse: float):
        # 先清空上次
        self.edit_mlr_paras = []
        while self.fit_res_layout.count():
            item = self.fit_res_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        widget = QWidget()
        formlayout = QFormLayout(widget)
        label_rmse = QLineEdit(f"{rmse:.4f}")
        label_rmse.setFixedWidth(120)
        label_inter = QLineEdit(f"{intercept:.4f}")
        label_inter.setFixedWidth(120)
        label_rmse.setEnabled(False)
        label_inter.setEnabled(False)

        formlayout.addRow("RMSE: ", label_rmse)
        formlayout.addRow("拟合常量: ", label_inter)

        self.edit_mlr_paras.append(label_inter)

        for i, c in enumerate(coef):
            label = QLineEdit(f"{c:.4f}")
            label.setFixedWidth(120)
            label.setEnabled(False)
            self.edit_mlr_paras.append(label)
            formlayout.addRow(f"拟合系数{i+1}: ", label)
            
        # 设置参数表起始位置
        self.edit_para_start_idx = QLineEdit("700000")
        self.edit_para_start_idx.setEnabled(False)
        self.edit_para_start_idx.setFixedWidth(120)
        self.edit_para_start_idx.setValidator(QIntValidator())
        formlayout.addRow("参数表起始: ", self.edit_para_start_idx)

        self.fit_res_layout.addWidget(widget, 0, Qt.AlignHCenter)

    def change_editable(self, flag):
        if self.edit_mlr_paras:
            for line_edit in self.edit_mlr_paras:
                line_edit.setEnabled(flag)
        if hasattr(self, "edit_para_start_idx") and self.edit_para_start_idx:
            self.edit_para_start_idx.setEnabled(flag)

    def change_negative(self, flag):
        if self.edit_mlr_paras:
            for line_edit in self.edit_mlr_paras:
                val = -float(line_edit.text())
                line_edit.setText(f"{val}")

    def get_intercept_coef(self) -> list[float]:
        res = []
        for line_edit in self.edit_mlr_paras:
            res.append(float(line_edit.text()))
        res.append(int(self.edit_para_start_idx.text()))
        return res
