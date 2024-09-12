from PyQt5.QtWidgets import (
    QWidget,
    QStackedWidget,
    QMainWindow,
    QHBoxLayout,
    QShortcut,
)
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QKeySequence

from src.components.custom_btn import CustomBtn
from src.pages.sample import Sample
from src.pages.tsp import Tsp
from src.pages.model import Model
from src.components.side_menu import SideMenu


class View(QMainWindow):
    signal_connect_nc = Signal(list)
    signal_disconnect_nc = Signal()
    signal_connect_tem_card = Signal(list)
    signal_disconnect_tem_card = Signal()
    signal_open_port = Signal(list)
    signal_close_port = Signal()
    signal_sample_save_path = Signal(str)
    signal_start_sample = Signal(dict)
    signal_change_orin_sample = Signal()
    signal_stop_sample = Signal()

    signal_import_data = Signal(list)
    signal_plot_files = Signal()
    signal_tra_tsp = Signal(list)
    signal_ga_tsp = Signal(list)
    signal_saved_data_path = Signal(str)
    signal_mlr_fit = Signal()
    signal_send_coef = Signal(list)

    signal_import_model = Signal(dict)
    signal_start_train = Signal(dict)
    signal_increase_train = Signal(dict)
    signal_pause_train = Signal()
    signal_resume_train = Signal()
    signal_stop_train = Signal()
    signal_save_model = Signal(str)

    signal_close_window = Signal()

    def __init__(self):
        super().__init__()
        # 当前页面索引
        self.cur_page = 0
        self.scale_factor = 1.0

        self.setup_ui()
        self.connect_slots()

    def setup_ui(self):
        import matplotlib.pyplot as plt

        plt.style.use("bmh")

        self.resize(1800, 1200)
        self.setWindowTitle("Soft")
        self.statusBar().show()
        self.side_menu = SideMenu(self)
        self.side_menu.btns[0].click()

        self.stack = QStackedWidget(self)
        self.sample_page = Sample(self)
        self.tsp_page = Tsp(self)
        self.model_page = Model(self)
        self.stack.addWidget(self.sample_page)
        self.stack.addWidget(self.tsp_page)
        self.stack.addWidget(self.model_page)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vLayout = QHBoxLayout(central_widget)
        vLayout.setSpacing(0)
        vLayout.setContentsMargins(0, 0, 0, 0)
        vLayout.addWidget(self.side_menu)
        vLayout.addWidget(self.stack)

        with open("src/style/LightStyle.qss", "r") as file:
            self.setStyleSheet(file.read())

        self.adapt_background_color()
        self.create_shortcut()

    def connect_slots(self):
        # 换页
        self.side_menu.signal_change_page.connect(self.on_change_page)
        # 连接机床NCLink
        self.sample_page.signal_connect_nc.connect(self.signal_connect_nc)
        # 断连机床NCLink
        self.sample_page.signal_disconnect_nc.connect(self.signal_disconnect_nc)
        # 连接采集卡
        self.sample_page.signal_connect_tem_card.connect(self.signal_connect_tem_card)
        # 断连采集卡
        self.sample_page.signal_disconnect_tem_card.connect(self.signal_disconnect_tem_card)
        # 打开串口
        self.sample_page.signal_open_port.connect(self.signal_open_port)
        # 关闭串口
        self.sample_page.signal_close_port.connect(self.signal_close_port)
        # 保存数据路径
        self.sample_page.signal_sample_save_path.connect(self.signal_sample_save_path)
        # 开始采集
        self.sample_page.signal_start_sample.connect(self.signal_start_sample)
        # 切换显示原始\采样数据
        self.sample_page.signal_change_orin_sample.connect(self.signal_change_orin_sample)
        # 停止采集
        self.sample_page.signal_stop_sample.connect(self.signal_stop_sample)
        # 导入数据
        self.tsp_page.import_data.signal_import_data.connect(self.signal_import_data)
        # 绘制数据
        self.tsp_page.import_data.btn_plot_file.clicked.connect(self.signal_plot_files)
        # 传统筛选
        self.tsp_page.tsp_config.signal_tra_tsp.connect(self.signal_tra_tsp)
        # 迭代筛选
        self.tsp_page.tsp_config.signal_ga_tsp.connect(self.signal_ga_tsp)
        # 保存筛选数据
        self.tsp_page.tsp_res.signal_saved_data_path.connect(self.signal_saved_data_path)
        # 线性拟合
        self.tsp_page.tsp_res.btn_mlr_fit.clicked.connect(self.signal_mlr_fit)
        # 导入参数
        self.tsp_page.tsp_res.signal_send_coef.connect(self.signal_send_coef)
        # 导入模型
        self.model_page.model_choose.signal_import_model.connect(self.signal_import_model)
        # 开始训练模型
        self.model_page.signal_start_train.connect(self.signal_start_train)
        # 增量训练
        self.model_page.signal_increase_train.connect(self.signal_increase_train)
        # 暂停训练
        self.model_page.signal_pause_train.connect(self.signal_pause_train)
        # 恢复训练
        self.model_page.signal_resume_train.connect(self.signal_resume_train)
        # 停止训练
        self.model_page.signal_stop_train.connect(self.signal_stop_train)
        # 保存模型
        self.model_page.signal_save_model.connect(self.signal_save_model)

    def vis_config(self, config: dict):
        self.sample_page.vis_config(config)
        self.tsp_page.vis_config(config)
        self.model_page.vis_config(config)

    def update_config(self, config: dict) -> dict:
        config = self.sample_page.update_config(config)
        config = self.tsp_page.update_config(config)
        config = self.model_page.update_config(config)
        return config

    def on_change_page(self, index: int):
        self.stack.setCurrentIndex(index)

    def adapt_background_color(self):
        R, G, B = 236 / 255, 239 / 255, 241 / 255

        self.sample_page.set_canvas_color((R, G, B))
        self.tsp_page.set_canvas_color((R, G, B))
        self.model_page.set_canvas_color((R, G, B))

    def create_shortcut(self):
        next_page = QShortcut(QKeySequence("Ctrl+Down"), self)
        next_page.activated.connect(self.next_page)
        last_page = QShortcut(QKeySequence("Ctrl+Up"), self)
        last_page.activated.connect(self.last_page)

    def next_page(self):
        if self.cur_page < len(self.side_menu.btns) - 1:
            self.cur_page += 1
        self.side_menu.btns[self.cur_page].click()

    def last_page(self):
        if self.cur_page > 0:
            self.cur_page -= 1
        self.side_menu.btns[self.cur_page].click()

    def closeEvent(self, _event):
        self.signal_close_window.emit()


# .\.env\Scripts\python.exe -m src.view
if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt
    from PyQt5.QtWidgets import QApplication

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.unicode_minus"] = False

    app = QApplication(sys.argv)
    view = View()
    view.show()
    app.exec()
