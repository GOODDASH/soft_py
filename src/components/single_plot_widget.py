import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QVBoxLayout, QWidget, QHBoxLayout, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class SinglePlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.font_size = 14

        self.vLayout = QVBoxLayout(self)

        # Plot layout
        self.hLayout_plots = QHBoxLayout()

        self.tem_plot = QWidget(self)
        self.tem_plot.setMinimumSize(QSize(450, 500))
        self.vLayout_tem_plot = QVBoxLayout(self.tem_plot)
        self.canvas_tem = FigureCanvas(Figure())
        self.ax_tem = self.canvas_tem.figure.add_subplot(1, 1, 1)
        self.ax_tem.set_title("温度(℃)")
        self.ax_tem.set_xlabel("采样点")
        self.ax_tem.spines["right"].set_visible(False)
        self.ax_tem.spines["top"].set_visible(False)
        self.canvas_tem.figure.tight_layout()
        self.toolbar_tem = NavigationToolbar(self.canvas_tem, self.tem_plot)
        self.vLayout_tem_plot.addWidget(self.canvas_tem)
        self.vLayout_tem_plot.addWidget(self.toolbar_tem)
        self.vLayout_tem_plot.setContentsMargins(0, 0, 0, 0)
        self.vLayout_tem_plot.setSpacing(0)

        self.err_plot = QWidget(self)
        self.err_plot.setMinimumSize(QSize(450, 500))
        self.vLayout_err_plot = QVBoxLayout(self.err_plot)
        self.canvas_err = FigureCanvas(Figure())
        self.ax_err = self.canvas_err.figure.add_subplot(1, 1, 1)
        self.ax_err.set_title("热误差(µm)")
        self.ax_err.set_xlabel("采样点")
        self.ax_err.spines["right"].set_visible(False)
        self.ax_err.spines["top"].set_visible(False)
        self.canvas_err.figure.tight_layout()
        self.toolbar_err = NavigationToolbar(self.canvas_err, self.err_plot)
        self.vLayout_err_plot.addWidget(self.canvas_err)
        self.vLayout_err_plot.addWidget(self.toolbar_err)
        self.vLayout_err_plot.setContentsMargins(0, 0, 0, 0)
        self.vLayout_err_plot.setSpacing(0)

        self.hLayout_plots.addWidget(self.tem_plot)
        self.hLayout_plots.addWidget(self.err_plot)
        self.hLayout_plots.setContentsMargins(0, 0, 0, 0)
        self.hLayout_plots.setSpacing(5)

        self.vLayout.addLayout(self.hLayout_plots)
        self.vLayout.setContentsMargins(0, 0, 0, 0)

    # 简单图示单个数据
    def plot(self, data):
        self.ax_tem.clear()
        self.ax_err.clear()
        self.ax_tem.plot(data[:, :-1])
        self.ax_err.plot(data[:, -1])
        self.update_canvas()

    def plot_sample_data(self, data, from_nc):
        self.ax_tem.clear()
        self.ax_err.clear()
        if from_nc and "nc_reg_g" in data and len(data["nc_reg_g"]) > 0:
            numpy_data = np.array(data["nc_reg_g"])
            self.plot_temp(numpy_data)
        elif not from_nc and "card_temp" in data and len(data["card_temp"]) > 0:
            numpy_data = np.array(data["card_temp"])
            self.plot_temp(numpy_data)
        if "error" in data and len(data["error"]) > 0:
            text_x = len(data["error"]) - 1
            text_y = data["error"][-1][0]
            self.ax_err.plot(data["error"], label="热误差")
            self.ax_err.annotate(
                f"{text_y}",
                xy=(text_x, text_y),
                xytext=(text_x, text_y),
                textcoords="data",
                fontsize=12,
            )
            self.ax_err.legend(loc="upper left")
        self.update_canvas()

    def plot_temp(self, data: np.ndarray):
        text_x = data.shape[0] - 1
        for col in range(data.shape[1]):
            text_y = data[-1, col]
            self.ax_tem.plot(data[:, col], label=f"T{col}")
            self.ax_tem.annotate(
                f"{text_y}",
                xy=(text_x, text_y),
                xytext=(text_x, text_y),
                textcoords="data",
                fontsize=12,
            )
        self.ax_tem.legend(loc="upper left")

    # 图示选择的测点和MLR的预测值和真实值对比
    def plot_pred(self, data, pred, tsp_list):
        self.ax_tem.clear()
        self.ax_err.clear()
        t_data = data[:, tsp_list]
        x_data = data[:, -1]
        xval = np.arange(t_data.shape[0])
        for col in range(t_data.shape[1]):
            self.ax_tem.plot(xval, t_data[:, col], label=f"T{tsp_list[col]}")
        self.ax_err.plot(xval, x_data, color="b", label="真实值")
        self.ax_err.plot(xval, pred, color="r", label="预测值")
        self.ax_tem.legend()
        self.ax_err.legend()
        self.update_canvas()

    # 设置图窗和坐标轴颜色
    def set_canvas_color(self, color: tuple[float, float, float]):
        self.canvas_tem.figure.patch.set_facecolor(color)
        self.canvas_err.figure.patch.set_facecolor(color)
        self.ax_tem.set_facecolor(color)
        self.ax_err.set_facecolor(color)
        self.canvas_tem.draw()
        self.canvas_err.draw()

    # 添加标签、标题并绘制
    def update_canvas(self):
        self.ax_tem.set_xlabel("测量点")
        self.ax_tem.set_title("温度(℃)")
        self.ax_tem.spines["right"].set_visible(False)
        self.ax_tem.spines["top"].set_visible(False)
        self.ax_tem.set_xlim(left=0)
        self.ax_err.set_xlabel("测量点")
        self.ax_err.set_title("热误差(µm)")
        self.ax_err.spines["right"].set_visible(False)
        self.ax_err.spines["top"].set_visible(False)
        self.ax_err.set_xlim(left=0)
        self.canvas_tem.draw()
        self.canvas_err.draw()

    # 更新字体大小
    def update_font_size(self):
        for ax in [self.ax_tem, self.ax_err]:
            ax.xaxis.label.set_fontsize(self.font_size)
            ax.yaxis.label.set_fontsize(self.font_size)
            ax.tick_params(axis="both", which="major", labelsize=self.font_size)
            if ax.get_legend() is not None:
                ax.legend(fontsize=self.font_size)
        self.canvas_tem.draw()
        self.canvas_err.draw()

    # 增大字体
    def increase_font_size(self):
        self.font_size += 1
        self.update_font_size()

    # 减小字体
    def decrease_font_size(self):
        self.font_size -= 1
        self.update_font_size()

    def show_left(self):
        self.tem_plot.show()
        self.err_plot.hide()

    def show_right(self):
        self.tem_plot.hide()
        self.err_plot.show()

    def show_both(self):
        self.tem_plot.show()
        self.err_plot.show()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main = SinglePlotWidget()
    main.show()
    sys.exit(app.exec())
