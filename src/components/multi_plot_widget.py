from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
    QLineEdit,
    QLabel,
    QButtonGroup,
)
from PyQt5.QtCore import pyqtSignal as Signal

from src.components import CustomBtn, SinglePlotWidget


class MultiPlotWidget(QWidget):
    signal_change_orin_sample = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas_color = None
        self.added_widgets = False
        self.widgets = []
        self.file_num = 1
        self.toggle_on = False
        self.btn_next = None
        self.btn_back = None
        self.edit_page_index = None
        self.cur_index = 1
        self.vLayout_main = QVBoxLayout(self)

        # Button layout
        self.hLayout_buttons = QHBoxLayout()
        self.hLayout_buttons.setSpacing(10)
        self.btn_tem = CustomBtn("top", QIcon("src/icons/splitscreen_left.png"))
        self.btn_tem.setToolTip("温度")
        self.btn_err = CustomBtn("top", QIcon("src/icons/splitscreen_right.png"))
        self.btn_err.setToolTip("热误差")
        self.btn_both = CustomBtn("top", QIcon("src/icons/splitscreen.png"))
        self.btn_both.setToolTip("同时显示")
        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.btn_tem, 1)
        self.btn_group.addButton(self.btn_err, 2)
        self.btn_group.addButton(self.btn_both, 3)
        self.hLayout_buttons.addStretch()
        self.hLayout_buttons.addWidget(self.btn_tem)
        self.hLayout_buttons.addWidget(self.btn_err)
        self.hLayout_buttons.addWidget(self.btn_both)

        self.btn_back = QPushButton("上一页")
        self.btn_next = QPushButton("下一页")
        self.edit_page_index = QLineEdit(f"{self.cur_index}")
        self.edit_page_index.setFixedWidth(50)
        self.edit_page_index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel("/")
        self.total_num_label = QLabel("1")
        self.total_num_label.setFixedWidth(50)
        self.total_num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.index_layout = QHBoxLayout()
        self.index_layout.addStretch()
        self.index_layout.addWidget(self.btn_back)
        self.index_layout.addWidget(self.edit_page_index)
        self.index_layout.addWidget(label)
        self.index_layout.addWidget(self.total_num_label)
        self.index_layout.addWidget(self.btn_next)
        self.index_layout.addStretch()
        self.index_layout.setContentsMargins(0, 0, 0, 0)

        self.stacked_plot_widget = QStackedWidget(self)
        self.initial_widget = SinglePlotWidget()
        self.widgets.append(self.initial_widget)
        self.stacked_plot_widget.addWidget(self.initial_widget)

        self.vLayout_main.addLayout(self.hLayout_buttons)
        self.vLayout_main.addWidget(self.stacked_plot_widget)
        self.vLayout_main.setContentsMargins(0, 0, 0, 0)
        self.vLayout_main.setSpacing(0)

        self.btn_group.buttonClicked.connect(self.on_button_clicked)
        self.btn_tem.click()  # 初始化的时候只显示左侧温度

        self.btn_back.clicked.connect(self.last_plot_widget)
        self.btn_next.clicked.connect(self.next_plot_widget)

    # 切换布局
    def on_button_clicked(self, button):
        btn_id = self.btn_group.id(button)
        for widget in self.widgets:
            match btn_id:
                case 1:
                    widget.show_left()
                case 2:
                    widget.show_right()
                case 3:
                    widget.show_both()

    def add_switch_orin_sample_btn(self):
        self.toggle_on_icon = QIcon("src/icons/switch_on.png")
        self.toggle_off_icon = QIcon("src/icons/switch_off.png")

        self.btn_switch = CustomBtn("switch", self.toggle_off_icon)
        self.hLayout_buttons.addWidget(self.btn_switch)
        self.btn_switch.setToolTip("切换原始\规则数据")
        self.btn_switch.clicked.connect(self.switch_orin_sample)

    def switch_orin_sample(self):
        self.toggle_on = not self.toggle_on
        if self.toggle_on:
            self.btn_switch.setIcon(self.toggle_on_icon)
        else:
            self.btn_switch.setIcon(self.toggle_off_icon)
        self.signal_change_orin_sample.emit()

    def plot_sample_data(self, data, tem_from_nc):
        self.initial_widget.plot_sample_data(data, tem_from_nc)

    # 初始化布局和页面
    def create_plot_widget(self):
        self.stacked_plot_widget.setCurrentIndex(0)
        self.cur_index = 1
        self.edit_page_index.setText(f"{self.cur_index}")
        self.btn_tem.click()

    # 显示多个文件的图形
    def plot_files(self, data_list: list):
        self.file_num = len(data_list)
        self.set_plot_widget(self.file_num)
        for idx, widget in enumerate(self.widgets):
            widget.plot(data_list[idx])
        self.create_plot_widget()

    # 显示多个文件选择的测点和拟合效果
    def plot_pred(self, data_list, pred_list, tsp_list):
        self.file_num = len(data_list)
        self.set_plot_widget(self.file_num)
        for idx, widget in enumerate(self.widgets):
            data = data_list[idx]
            pred = pred_list[idx]
            widget.plot_pred(data, pred, tsp_list)
        self.create_plot_widget()

    def set_plot_widget(self, num):
        # 在第一次导入文件后添加翻页按钮
        if not self.added_widgets:
            self.vLayout_main.insertLayout(1, self.index_layout)
            self.added_widgets = True
        if num == len(self.widgets):
            # 文件数量没有改变则直接跳过
            return
        elif num > len(self.widgets):
            # 文件数量变多则相应添加，并改变总页面数
            self.total_num_label.setText(f"{num}")
            for _ in range(num - len(self.widgets)):
                widget = SinglePlotWidget()
                self.widgets.append(widget)
                self.stacked_plot_widget.addWidget(widget)
        else:
            # 文件数量变少则从后前向删除页面和存储的widget对象
            self.total_num_label.setText(f"{num}")
            widget_num = len(self.widgets)  # 先存储当前的页面数量
            for i in range(widget_num - num):
                self.widgets.pop()
                # 从后向前清除页面
                widget_to_remove = self.stacked_plot_widget.widget(widget_num - i - 1)
                self.stacked_plot_widget.removeWidget(widget_to_remove)
                widget_to_remove.deleteLater()

        self.set_canvas_color(self.canvas_color)

    def last_plot_widget(self):
        if self.cur_index != 1:
            self.cur_index -= 1
            self.edit_page_index.setText(f"{self.cur_index}")
            self.stacked_plot_widget.setCurrentIndex(self.cur_index - 1)

    def next_plot_widget(self):
        if self.cur_index != self.file_num:
            self.cur_index += 1
            self.edit_page_index.setText(f"{self.cur_index}")
            self.stacked_plot_widget.setCurrentIndex(self.cur_index - 1)

    def set_canvas_color(self, color: tuple[float, float, float]):
        self.canvas_color = color
        for widget in self.widgets:
            widget.set_canvas_color(color)

    def decrease_font_size(self):
        for widget in self.widgets:
            widget.decrease_font_size()

    def increase_font_size(self):
        for widget in self.widgets:
            widget.increase_font_size()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main = MultiPlotWidget()
    main.show()
    sys.exit(app.exec())
