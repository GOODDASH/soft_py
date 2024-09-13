import time
from collections import deque
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QScrollArea,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ModelPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vLayout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.vLayout.addWidget(scroll_area)

        self.last_update_time = 0
        self.update_interval = 0.2

        self.cur_epoch = 0
        self.train_loss = deque(maxlen=200)
        self.val_loss = deque(maxlen=200)

        self.train_plot = self.set_train_plot()
        self.train_plot.setMinimumSize(QSize(450, 250))
        self.pred_plot = self.set_pred_plot()
        self.pred_plot.setMinimumSize(QSize(450, 250))
        self.plot_ara = self.cat_plot_area()
        scroll_area.setWidget(self.plot_ara)

    def set_train_plot(self) -> QWidget:
        plot_widget = QWidget(self)
        vlayout = QVBoxLayout(plot_widget)

        self.canvas1 = FigureCanvas(Figure(figsize=(5, 3)))
        self.toolbar1 = NavigationToolbar(self.canvas1)
        self.ax1 = self.canvas1.figure.add_subplot(1, 2, 1)
        self.ax2 = self.canvas1.figure.add_subplot(1, 2, 2)

        (self.line_train,) = self.ax1.plot([], label="训练集损失", color="#2A9D8F")
        (self.line_val,) = self.ax2.plot([], label="测试集损失", color="#F4A261")

        self.ax1.set_xlabel("迭代次数")
        self.ax2.set_xlabel("迭代次数")
        self.ax1.spines["right"].set_visible(False)
        self.ax1.spines["top"].set_visible(False)
        self.ax2.spines["right"].set_visible(False)
        self.ax2.spines["top"].set_visible(False)
        self.canvas1.figure.tight_layout()
        vlayout.addWidget(self.canvas1)
        vlayout.addWidget(self.toolbar1)
        vlayout.setStretch(0, 1)
        vlayout.setStretch(1, 0)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)

        return plot_widget

    def set_pred_plot(self) -> QWidget:
        pred_plot = QWidget()
        vlayout = QVBoxLayout(pred_plot)
        self.canvas2 = FigureCanvas(Figure(figsize=(5, 3)))
        self.toolbar2 = NavigationToolbar(self.canvas2)
        self.ax3 = self.canvas2.figure.add_subplot(1, 1, 1)

        (self.line_pred,) = self.ax3.plot([], label="预测值", color="#003049")
        (self.line_true,) = self.ax3.plot([], label="真实值", color="#C1121F")

        self.ax3.spines["right"].set_visible(False)
        self.ax3.spines["top"].set_visible(False)
        self.ax3.set_xlabel("数据点")
        self.canvas2.figure.tight_layout()
        vlayout.addWidget(self.canvas2)
        vlayout.addWidget(self.toolbar2)
        vlayout.setStretch(0, 1)
        vlayout.setStretch(1, 0)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)

        return pred_plot

    def cat_plot_area(self):
        plot_area = QWidget()
        vlayout = QVBoxLayout(plot_area)
        vlayout.addWidget(self.train_plot)
        vlayout.addWidget(self.pred_plot)
        vlayout.setStretch(1, 1)
        vlayout.setStretch(0, 1)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        return plot_area

    def set_canvas_color(self, color: tuple):
        self.canvas1.figure.patch.set_facecolor(color)
        self.ax1.set_facecolor(color)
        self.ax2.set_facecolor(color)
        self.canvas2.figure.patch.set_facecolor(color)
        self.ax3.set_facecolor(color)
        self.canvas1.draw()
        self.canvas2.draw()

    def update_loss_canvas(self, signal: tuple, true_val):
        # 不管是否绘制，首先要更新数据
        self.cur_epoch += 1
        self.train_loss.append(signal[3])
        self.val_loss.append(signal[4])
        self.pred = signal[6]
        # 设置了最短绘图间隔，不然太快会阻塞程序响应
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
        loss_x = [x + self.cur_epoch for x in range(len(self.train_loss))]
        self.line_train.set_data(loss_x, self.train_loss)
        self.line_val.set_data(loss_x, self.val_loss)
        self.line_pred.set_data(range(len(self.pred)), self.pred)
        self.line_true.set_data(range(len(true_val)), true_val)

        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim()
        self.ax2.autoscale_view()
        self.ax3.relim()
        self.ax3.autoscale_view()

        self.ax1.legend(loc="upper right")
        self.ax2.legend(loc="upper right")
        self.ax3.legend(loc="lower right")

        self.canvas1.draw_idle()  # 使用 draw_idle() 而不是 draw()
        self.canvas2.draw_idle()

        # 更新最后一次更新时间
        self.last_update_time = current_time

    def update_finished_canvas(self, signal: tuple):
        # TODO: 训练结束，图像相应反馈
        pass

    def reset_train_val_list(self):
        self.cur_epoch = 0
        self.train_loss.clear()
        self.val_loss.clear()
