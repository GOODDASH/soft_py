from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal as Signal
from src.components import CompenTemModel

class Compen(QWidget):
    signal_import_tem_model = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.import_tem_model = CompenTemModel(self)

        self.compen_setting_area = QScrollArea()
        self.compen_setting_area.setWidgetResizable(True)
        self.compen_setting_area.setMinimumWidth(400)
        self.compen_widget = QWidget()
        self.compen_widget_layout = QVBoxLayout(self.compen_widget)
        self.compen_widget_layout.setSpacing(10)
        self.compen_widget_layout.addWidget(self.import_tem_model)
        self.compen_widget_layout.addStretch()
        self.compen_setting_area.setWidget(self.compen_widget)

        self.plot_area = QScrollArea()
        self.plot_area.setWidgetResizable(True)
        self.plot_widget = QWidget()
        self.plot_area.setWidget(self.plot_widget)

        self.layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.compen_setting_area)
        self.splitter.addWidget(self.plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.layout.addWidget(self.splitter)
        self.layout.setSpacing(0)

        self.connect_slots()

    def connect_slots(self):
        self.import_tem_model.signal_import_tem_model.connect(self.signal_import_tem_model)