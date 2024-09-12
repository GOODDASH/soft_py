from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QButtonGroup,
    QRadioButton,
    QLabel,
    QGridLayout,
    QPushButton,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QSizePolicy,
    QGroupBox,
    QFormLayout,
    QLineEdit,
)


class TspImportSetting(QGroupBox):
    signal_import_data = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("数据导入")
        self.vLayout = QVBoxLayout(self)

        self.gridLayout = QGridLayout()
        self.btn_group1 = QButtonGroup()
        self.radioBtn_ac_row = QRadioButton("按行")
        self.radioBtn_ac_col = QRadioButton("按列")
        self.btn_group1.addButton(self.radioBtn_ac_row, 0)
        self.btn_group1.addButton(self.radioBtn_ac_col, 1)
        self.btn_group2 = QButtonGroup()
        self.radioBtn_tab = QRadioButton("tab")
        self.radioBtn_semi = QRadioButton(",")
        self.radioBtn_blank = QRadioButton("空格")
        self.radioBtn_none = QRadioButton("无")
        self.btn_group2.addButton(self.radioBtn_tab, 0)
        self.btn_group2.addButton(self.radioBtn_semi, 1)
        self.btn_group2.addButton(self.radioBtn_blank, 2)
        self.btn_group2.addButton(self.radioBtn_none, 3)
        self.gridLayout.addWidget(QLabel("数据布局:"), 0, 0)
        self.gridLayout.addWidget(QLabel("分隔符号:"), 0, 1)
        self.gridLayout.addWidget(self.radioBtn_ac_row, 1, 0)
        self.gridLayout.addWidget(self.radioBtn_ac_col, 2, 0)
        self.hLayout_btn1 = QHBoxLayout()
        self.hLayout_btn2 = QHBoxLayout()
        self.hLayout_btn1.addWidget(self.radioBtn_tab)
        self.hLayout_btn1.addWidget(self.radioBtn_semi)
        self.hLayout_btn2.addWidget(self.radioBtn_blank)
        self.hLayout_btn2.addWidget(self.radioBtn_none)
        self.gridLayout.addLayout(self.hLayout_btn1, 1, 1)
        self.gridLayout.addLayout(self.hLayout_btn2, 2, 1)

        self.fLayout = QFormLayout()
        self.edit_tem_index_begin = QLineEdit()
        self.edit_tem_index_end = QLineEdit()
        self.edit_error_index = QLineEdit()
        self.fLayout.addRow("温度起始索引:", self.edit_tem_index_begin)
        self.fLayout.addRow("温度结束索引:", self.edit_tem_index_end)
        self.fLayout.addRow("热误差索引:", self.edit_error_index)

        self.btn_hLayout = QHBoxLayout()
        self.btn_import_file = QPushButton("导入数据")
        self.btn_plot_file = QPushButton("图示数据")
        self.btn_plot_file.setEnabled(False)
        self.btn_hLayout.addWidget(self.btn_import_file)
        self.btn_hLayout.addWidget(self.btn_plot_file)

        self.table_data = QTableWidget()

        self.vLayout.addStretch()
        self.vLayout.addLayout(self.fLayout)
        self.vLayout.addLayout(self.gridLayout)
        self.vLayout.addLayout(self.btn_hLayout)
        self.vLayout.addStretch()

        self.btn_import_file.clicked.connect(self.on_btn_import_file)

    def on_btn_import_file(self):
        file_filter = (
            "Excel files (*.xls *.xlsx);;CSV files (*.csv);;Text files (*.txt);;All files (*.*)"
        )
        file_paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", file_filter)
        if file_paths:
            para = [
                file_paths,
                self.btn_group1.checkedId(),
                self.btn_group2.checkedId(),
                int(self.edit_tem_index_begin.text()),
                int(self.edit_tem_index_end.text()),
                int(self.edit_error_index.text()),
            ]
            self.btn_plot_file.setEnabled(True)
            self.signal_import_data.emit(para)

    def show_data(self, data):
        self.vLayout.removeWidget(self.table_data)

        table_widget = QTableWidget()
        table_widget.setMinimumHeight(300)
        table_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        row = data.shape[0]
        col = data.shape[1]
        table_widget.setRowCount(row)
        table_widget.setColumnCount(col)
        for i in range(row):
            for j in range(col):
                item = QTableWidgetItem(f"{data[i, j]:.2f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)  # 设置数据居中
                table_widget.setItem(i, j, item)
        for j in range(col):
            table_widget.setColumnWidth(j, 70)

        self.table_data = table_widget
        self.vLayout.insertWidget(3, self.table_data)
