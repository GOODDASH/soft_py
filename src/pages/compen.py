from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal as Signal
from src.components import CompenTemModel, CompenImportRpm, CompenGetPara
from src.style.gui_const import SIDE_MIN_WIDTH


class Compen(QWidget):
    signal_import_tem_model = Signal(dict)
    signal_import_rpm = Signal(str)
    signal_start_compen = Signal(dict)
    signal_stop_compen = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.import_tem_model = CompenTemModel(self)
        self.import_rpm = CompenImportRpm(self)
        self.get_para = CompenGetPara(self)

        self.compen_setting_area = QScrollArea()
        self.compen_setting_area.setWidgetResizable(True)
        self.compen_setting_area.setMinimumWidth(SIDE_MIN_WIDTH)
        self.compen_widget_container = QWidget()
        self.compen_widget_container_layout = QHBoxLayout(self.compen_widget_container)
        # self.compen_widget_container_layout.setAlignment(Qt.AlignTop)
        self.compen_widget = QWidget()
        self.compen_widget.setMaximumWidth(500)
        self.compen_widget_layout = QVBoxLayout(self.compen_widget)
        self.compen_widget_layout.setContentsMargins(0, 0, 0, 0)
        self.compen_widget_layout.setSpacing(15)
        self.compen_widget_layout.addWidget(self.import_tem_model)
        self.compen_widget_layout.addWidget(self.import_rpm)
        self.compen_widget_layout.addWidget(self.get_para)
        self.compen_widget_layout.addStretch()
        self.compen_widget_container_layout.addWidget(self.compen_widget, 0, Qt.AlignVCenter)
        self.compen_setting_area.setWidget(self.compen_widget_container)

        self.plot_area = QScrollArea()
        self.plot_area.setWidgetResizable(True)
        self.plot_widget = QWidget()
        self.plot_area.setWidget(self.plot_widget)

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setContentsMargins(0, 10, 10, 0)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.compen_setting_area)
        self.splitter.addWidget(self.plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.vlayout.addWidget(self.splitter)
        self.vlayout.setSpacing(0)

        self.connect_slots()

    def connect_slots(self):
        self.import_tem_model.signal_import_tem_model.connect(self.signal_import_tem_model)
        self.import_rpm.signal_import_rpm.connect(self.signal_import_rpm)
        self.get_para.signal_start_compen.connect(self.signal_start_compen)
        self.get_para.signal_stop_compen.connect(self.signal_stop_compen)
