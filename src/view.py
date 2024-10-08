from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QStackedWidget,
    QMainWindow,
    QHBoxLayout,
    QShortcut,
)
from PyQt5.QtCore import (
    pyqtSignal as Signal,
    Qt,
    QTimer,
)
from PyQt5.QtGui import QKeySequence

from src.pages import Sample, Tsp, Model, Compen
from src.components import SideMenu, PopUp


class View(QMainWindow):
    signal_window_show = Signal()

    signal_connect_nc = Signal(str, int, str)
    signal_disconnect_nc = Signal()
    signal_connect_tem_card = Signal(list)
    signal_disconnect_tem_card = Signal()
    signal_open_port = Signal(str, int)
    signal_close_port = Signal()
    signal_sample_save_path = Signal(str)
    signal_start_sample = Signal(dict)
    signal_switch_plot = Signal()
    signal_change_orin_rule = Signal(str)
    signal_stop_sample = Signal()

    signal_import_data = Signal(list)
    signal_plot_files = Signal()
    signal_tra_tsp = Signal(list)
    signal_ga_tsp = Signal(list)
    signal_saved_data_path = Signal(list)
    signal_mlr_fit = Signal()
    signal_send_coef = Signal(list)

    signal_import_model = Signal(dict)
    signal_start_train = Signal(dict)
    signal_increase_train = Signal(dict)
    signal_pause_train = Signal()
    signal_resume_train = Signal()
    signal_stop_train = Signal()
    signal_save_model = Signal(str)

    signal_import_tem_model = Signal(dict)
    signal_import_rpm = Signal(str)
    signal_start_compen = Signal(dict)
    signal_stop_compen = Signal()

    signal_close_window = Signal()

    def __init__(self):
        super().__init__()
        # 当前页面索引
        self.cur_page = 0
        self.setup_ui()

    def setup_ui(self):
        import matplotlib.pyplot as plt

        plt.style.use("bmh")

        self.resize(1500, 1000)
        self.setWindowTitle("Soft")
        self.side_menu = SideMenu(self)
        self.side_menu.btns[0].click()

        self.stack = QStackedWidget(self)
        self.sample_page = Sample(self)
        self.tsp_page = Tsp(self)
        self.model_page = Model(self)
        self.compen_page = Compen(self)
        self.stack.addWidget(self.sample_page)
        self.stack.addWidget(self.tsp_page)
        self.stack.addWidget(self.model_page)
        self.stack.addWidget(self.compen_page)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        hLayout = QHBoxLayout(central_widget)
        hLayout.setSpacing(10)
        hLayout.setContentsMargins(0, 0, 0, 0)
        hLayout.addWidget(self.side_menu)
        hLayout.addWidget(self.stack)

        self.info_label = QLabel()
        self.info_label.setObjectName("statusLabel")
        self.info_label.setAlignment(Qt.AlignCenter)

        self.info_container = QWidget()
        self.info_container.setObjectName("statusBar")
        self.info_container.setFixedHeight(40)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 3, 0, 3)
        layout.addWidget(self.info_label, 1, Qt.AlignCenter)
        self.info_container.setLayout(layout)

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addPermanentWidget(self.info_container, 1)
        self.statusBar().hide()

        # 定时器，用于隐藏消息
        self.timer = QTimer()
        self.timer.timeout.connect(self.clear_message)

        self.load_style_sheet()
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
        # 显示\关闭图像
        self.sample_page.signal_swtich_plot.connect(self.signal_switch_plot)
        # 切换显示原始\规则图像
        self.sample_page.signal_change_orin_rule.connect(self.signal_change_orin_rule)
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
        # 导入温度模型
        self.compen_page.signal_import_tem_model.connect(self.signal_import_tem_model)
        # 导入采集的转速数据
        self.compen_page.signal_import_rpm.connect(self.signal_import_rpm)
        # 开始代理模型补偿
        self.compen_page.signal_start_compen.connect(self.signal_start_compen)
        # 停止代理模型补偿
        self.compen_page.signal_stop_compen.connect(self.signal_stop_compen)

    def vis_config(self, config: dict):
        self.sample_page.vis_config(config)
        self.tsp_page.vis_config(config)
        self.model_page.vis_config(config)

    def update_config(self, config: dict) -> dict:
        config = self.sample_page.update_config(config)
        config = self.tsp_page.update_config(config)
        config = self.model_page.update_config(config)
        return config

    def show_status_message(self, message, timeout=0):
        self.statusBar().show()
        self.info_label.setText(message)

        # 如果设置了超时时间，启动定时器
        if timeout > 0:
            self.timer.start(timeout)

    def show_pop_message(self, message, timeout=None):
        self.popup = PopUp(message, timeout, self)

        # 移动弹窗到主窗口中心
        self.popup.setGeometry(
            self.geometry().center().x() - self.popup.width() // 2,
            self.geometry().center().y() - self.popup.height() // 2,
            self.popup.width(),
            self.popup.height(),
        )
        self.popup.show()

    def clear_message(self):
        """清除居中的消息."""
        self.statusBar().hide()
        self.info_label.clear()
        self.timer.stop()

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
        refresh_qss = QShortcut(QKeySequence("Ctrl+R"), self)
        refresh_qss.activated.connect(self.load_style_sheet)

    def next_page(self):
        if self.cur_page < len(self.side_menu.btns) - 1:
            self.cur_page += 1
        self.side_menu.btns[self.cur_page].click()

    def last_page(self):
        if self.cur_page > 0:
            self.cur_page -= 1
        self.side_menu.btns[self.cur_page].click()

    def load_style_sheet(self):
        with open("src/style/LightStyle.qss", "r") as file:
            self.setStyleSheet(file.read())

    def closeEvent(self, _event):
        self.signal_close_window.emit()


# .\.env\Scripts\python.exe -m src.view
if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt
    from PyQt5.QtWidgets import QApplication

    plt.rcParams["font.sans-serif"] = ["Sarasa UI SC"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.unicode_minus"] = False

    app = QApplication(sys.argv)
    view = View()
    view.show()
    app.exec()
