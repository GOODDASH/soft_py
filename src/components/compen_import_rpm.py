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
    QGroupBox,  QTableWidget, QTableWidgetItem,
)


class CompenImportRpm(QGroupBox):
    signal_import_rpm = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("导入工况")

        self.btn_import_rpm = QPushButton("导入平均转速文件")
        self.btn_import_rpm.clicked.connect(self.on_btn_import_rpm)

        self.vLayout = QVBoxLayout(self)
        self.vLayout.addWidget(self.btn_import_rpm)

        # 初始化表格属性为 None
        self.table = None

    def on_btn_import_rpm(self):
        file_filter = "CSV文件(*.csv);;All files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择采集的转速温度文件", "", file_filter)
        if file_path:
            self.signal_import_rpm.emit(file_path)

    def show_avg_rpm(self, data):
        # 检查是否已有表格存在，如果有则删除
        if self.table is not None:
            self.vLayout.removeWidget(self.table)
            self.table.deleteLater()

        row = data.shape[0]
        self.table = QTableWidget(row, 1)
        self.table.setFixedWidth(200)
        for i in range(row):
            item = QTableWidgetItem(str(data[i]))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(i, 0, item)
        
        self.vLayout.addWidget(self.table, 1, Qt.AlignHCenter)


