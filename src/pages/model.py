from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSplitter,
    QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal as Signal

from src.components import ModelChoose, ModelTrain, ModelPlot


class Model(QWidget):
    signal_start_train = Signal(dict)
    signal_increase_train = Signal(dict)
    signal_pause_train = Signal()
    signal_resume_train = Signal()
    signal_stop_train = Signal()
    signal_save_model = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.model_choose = ModelChoose(self)
        self.model_train = ModelTrain(self)

        self.model_setting_area = QScrollArea()
        self.model_setting_area.setWidgetResizable(True)
        self.model_setting_area.setMinimumWidth(400)
        self.model_widget = QWidget()
        self.model_widget_layout = QVBoxLayout(self.model_widget)
        self.model_widget_layout.setSpacing(10)
        self.model_widget_layout.addWidget(self.model_choose)
        self.model_widget_layout.addWidget(self.model_train)
        self.model_widget_layout.addStretch()
        self.model_setting_area.setWidget(self.model_widget)

        self.plot_area = QScrollArea()
        self.plot_area.setWidgetResizable(True)
        self.plot_widget = ModelPlot(self)
        self.plot_area.setWidget(self.plot_widget)

        self.layout = QVBoxLayout(self)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.model_setting_area)
        self.splitter.addWidget(self.plot_area)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.layout.addWidget(self.splitter)
        self.layout.setSpacing(0)

        self.connect_slots()

    def connect_slots(self):
        self.model_train.btn_start_train.clicked.connect(self.on_btn_start_train)
        self.model_train.btn_increase_train.clicked.connect(self.on_increase_train)
        self.model_train.btn_save_model.clicked.connect(self.on_btn_save_model)
        self.model_train.signal_pause_resume_train.connect(self.on_btn_pause_resume_train)
        self.model_train.signal_stop_train.connect(lambda: self.signal_stop_train.emit())

    def on_btn_start_train(self):
        para_dict = self.get_para_dict()
        self.signal_start_train.emit(para_dict)
        self.model_train.btn_start_train.setEnabled(False)
        self.model_train.btn_increase_train.setEnabled(False)
        # 清空历史损失值
        self.plot_widget.reset_train_val_list()
        # 点击开始训练后，按钮变成暂停训练
        self.model_train.show_progress_bar_and_stop_btn()

    def on_increase_train(self):
        para_dict = self.get_para_dict()
        self.signal_increase_train.emit(para_dict)
        self.model_train.btn_start_train.setEnabled(False)
        self.model_train.btn_increase_train.setEnabled(False)
        # 增量（继续）训练不会清空历史训练损失值
        self.plot_widget.reset_train_val_list()
        # 点击开始训练后，添加暂停训练
        self.model_train.show_progress_bar_and_stop_btn()

    def get_para_dict(self):
        para_dict = dict()
        para_dict["model_para"] = self.model_choose.get_model_para()
        para_dict["model_type"] = self.model_choose.model_type.currentText()

        train_para = dict()
        train_para["lr"] = float(self.model_train.edit_learning_rate.text())
        train_para["batch_size"] = int(self.model_train.edit_batch_size.text())
        train_para["epoch"] = int(self.model_train.edit_epoch.text())

        optimizer_dict = dict()
        optimizer_para = dict()
        optimizer = self.model_train.scheduler_params_widget.get_cur_combo()
        match optimizer:
            case "StepLR":
                optimizer_dict["type"] = "StepLR"
                optimizer_para["step_size"] = int(self.model_train.edit_step_size.text())
                optimizer_para["step_gamma"] = float(self.model_train.edit_step_gamma.text())
            case "MultiStepLR":
                optimizer_dict["type"] = "MultiStepLR"
                milestones = list(map(int, self.model_train.edit_milestones.text().split(",")))
                optimizer_para["milestones"] = milestones
                optimizer_para["mile_gamma"] = float(self.model_train.edit_mile_gamma.text())
            case "CosineAnnealingLR":
                optimizer_dict["type"] = "CosineAnnealingLR"
                optimizer_para["T_max"] = int(self.model_train.edit_T_max.text())

        optimizer_dict["para"] = optimizer_para
        train_para["optimizer_dict"] = optimizer_dict
        para_dict["train_para"] = train_para
        return para_dict

    def on_btn_pause_resume_train(self, pause: bool):
        if pause:
            self.signal_pause_train.emit()
        else:
            self.signal_resume_train.emit()

    def on_btn_save_model(self):
        file_filter = "Pytorch模型文件 (*.pth)"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "保存模型文件到", "", file_filter
        )
        if file_path:
            if selected_filter == "Pytorch模型文件 (*.pth)" and not file_path.endswith(".pth"):
                file_path += ".pth"
            self.signal_save_model.emit(file_path)

    def update_loss_canvas(self, signal: tuple, true_val):
        # signal:
        # fold,
        # epoch,
        # num_epochs,
        # avg_train_loss,
        # avg_val_loss,
        # test_loss,
        # pred.squeeze(1)

        # 计算当前总进度
        cur = signal[1] + 1 + signal[0] * signal[2]
        total = 5 * signal[2]
        progress = int(cur * 100 / total)
        self.model_train.update_progress_bar(progress)
        # 绘图
        self.plot_widget.update_loss_canvas(signal, true_val)

    def update_finished_canvas(self, signal: tuple):
        # 训练结束后删除进度条和停止训练按钮
        self.model_train.hide_progress_bar_and_stop_btn()
        # 恢复按钮槽函数
        self.model_train.btn_start_train.setText("重新训练")
        # 绘图
        self.plot_widget.update_finished_canvas(signal)

    def set_canvas_color(self, color):
        self.plot_widget.set_canvas_color(color)

    def vis_config(self, config):
        self.model_choose.edit_num_heads.setText(str(config["num_heads"]))
        self.model_choose.edit_gnn_dim.setText(str(config["gnn_dim"]))
        self.model_choose.edit_lstm_dim.setText(str(config["lstm_dim"]))
        self.model_choose.edit_num_nodes.setText(str(config["num_nodes"]))
        self.model_choose.edit_seq_len.setText(str(config["seq_len"]))
        self.model_choose.edit_edge_start.setText(config["edge_start"])
        self.model_choose.edit_edge_end.setText(config["edge_end"])

        self.model_train.edit_learning_rate.setText(str(config["learning_rate"]))
        self.model_train.edit_batch_size.setText(str(config["batch_size"]))
        self.model_train.edit_epoch.setText(str(config["epoch"]))

        self.model_train.edit_step_size.setText(str(config["step_size"]))
        self.model_train.edit_step_gamma.setText(str(config["step_gamma"]))
        self.model_train.edit_T_max.setText(str(config["T_max"]))

    def update_config(self, config: dict):
        config["num_heads"] = int(self.model_choose.edit_num_heads.text())
        config["gnn_dim"] = int(self.model_choose.edit_gnn_dim.text())
        config["lstm_dim"] = int(self.model_choose.edit_lstm_dim.text())
        config["num_nodes"] = int(self.model_choose.edit_num_nodes.text())
        config["seq_len"] = int(self.model_choose.edit_seq_len.text())
        config["edge_start"] = self.model_choose.edit_edge_start.text()
        config["edge_end"] = self.model_choose.edit_edge_end.text()

        config["learning_rate"] = float(self.model_train.edit_learning_rate.text())
        config["batch_size"] = int(self.model_train.edit_batch_size.text())
        config["epoch"] = int(self.model_train.edit_epoch.text())

        config["step_size"] = int(self.model_train.edit_step_size.text())
        config["step_gamma"] = float(self.model_train.edit_step_gamma.text())
        config["T_max"] = int(self.model_train.edit_T_max.text())

        return config


# .\.env\Scripts\python.exe -m src.pages.model
if __name__ == "__main__":
    import sys
    from matplotlib import pyplot as plt
    from PyQt5.QtWidgets import QApplication

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.unicode_minus"] = False

    app = QApplication(sys.argv)
    model = Model()
    model.resize(1200, 800)
    model.show()
    app.exec()
