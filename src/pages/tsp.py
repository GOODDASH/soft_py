from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QApplication,
    QScrollArea,
    QSplitter,
)
from PyQt5.QtCore import Qt

from src.components import TspImportSetting, TspConfig, TspRes, MultiPlotWidget
from src.style.gui_const import SIDE_MIN_WIDTH


class Tsp(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.import_data = TspImportSetting(self)
        self.tsp_config = TspConfig(self)
        self.tsp_res = TspRes(self)

        self.tsp_setting_area = QScrollArea()
        self.tsp_setting_area.setWidgetResizable(True)
        self.tsp_setting_area.setMinimumWidth(SIDE_MIN_WIDTH)
        self.tsp_widget_container = QWidget()
        self.tsp_widget_container_layout = QHBoxLayout(self.tsp_widget_container)
        # self.tsp_widget_container_layout.setAlignment(Qt.AlignTop)
        self.tsp_widget = QWidget()
        self.tsp_widget.setMaximumWidth(500)
        self.tsp_widget_layout = QVBoxLayout(self.tsp_widget)
        self.tsp_widget_layout.setSpacing(15)
        self.tsp_widget_layout.addWidget(self.import_data)
        self.tsp_widget_layout.addWidget(self.tsp_config)
        self.tsp_widget_layout.addWidget(self.tsp_res)
        self.tsp_widget_layout.addStretch()
        self.tsp_widget_container_layout.addWidget(self.tsp_widget, 0, Qt.AlignVCenter)
        self.tsp_setting_area.setWidget(self.tsp_widget_container)

        self.plot_area = QScrollArea()
        self.plot_widget = MultiPlotWidget()
        self.plot_area.setWidgetResizable(True)
        self.plot_area.setWidget(self.plot_widget)

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setContentsMargins(0, 10, 10, 10)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.tsp_setting_area)
        self.splitter.addWidget(self.plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.vlayout.addWidget(self.splitter)
        self.vlayout.setSpacing(0)

    def plot_files(self, data: list):
        self.plot_widget.plot_files(data)

    def set_canvas_color(self, color):
        self.plot_widget.set_canvas_color(color)

    def vis_config(self, config):
        self.import_data.edit_tem_index_begin.setText(str(config["t_begin"]))
        self.import_data.edit_tem_index_end.setText(str(config["t_end"]))
        self.import_data.edit_error_index.setText(str(config["e_idx"]))
        btns1 = self.import_data.btn_group1.buttons()
        btns2 = self.import_data.btn_group2.buttons()
        btns1[config["ac_row"]].setChecked(True)
        btns2[config["seq_type"]].setChecked(True)
        self.tsp_config.edit_tsp_num.setText(str(config["tsp_num"]))
        self.tsp_config.edit_size.setText(str(config["ga_size"]))
        self.tsp_config.edit_epoch.setText(str(config["ga_epoch"]))
        self.tsp_res.edit_tsp.setText(str(config["tsp_res"]))

    def update_config(self, config) -> dict:
        config["t_begin"] = int(self.import_data.edit_tem_index_begin.text())
        config["t_end"] = int(self.import_data.edit_tem_index_end.text())
        config["e_idx"] = int(self.import_data.edit_error_index.text())
        config["ac_row"] = self.import_data.btn_group1.checkedId()
        config["seq_type"] = self.import_data.btn_group2.checkedId()
        config["tsp_num"] = int(self.tsp_config.edit_tsp_num.text())
        config["ga_size"] = int(self.tsp_config.edit_size.text())
        config["ga_epoch"] = int(self.tsp_config.edit_epoch.text())
        config["tsp_res"] = self.tsp_res.edit_tsp.text()
        return config


# .\.env\Scripts\python.exe -m src.pages.tsp
if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt
    from PyQt5.QtWidgets import QApplication

    plt.rcParams["font.sans-serif"] = ["Sarasa UI SC"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.unicode_minus"] = False

    app = QApplication(sys.argv)
    tsp = Tsp()
    tsp.resize(1200, 800)
    tsp.show()
    app.exec()
