from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QApplication,
    QLabel,
    QHBoxLayout,
    QPushButton,
    QProgressBar,
    QGroupBox,
)

from src.components.combo_options import ComboOptions


class ModelTrain(QGroupBox):
    signal_stop_train = Signal(bool)
    signal_pause_resume_train = Signal(bool)

    def __init__(self, parent=None):
        super(ModelTrain, self).__init__(parent)

        self.setTitle("模型训练")
        self.btn_pause_resume = QPushButton("暂停训练")
        self.btn_stop = QPushButton("停止训练")
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setObjectName("progress_bar")
        self.btn_pause_resume.clicked.connect(self.on_btn_pause_resume)
        self.btn_stop.clicked.connect(lambda: self.signal_stop_train.emit(True))

        self.show_pause = True

        self.vLayout = QVBoxLayout(self)
        self.vLayout.setSpacing(10)

        self.formLayout = QFormLayout()
        self.edit_learning_rate = QLineEdit()
        self.edit_batch_size = QLineEdit()
        self.edit_epoch = QLineEdit()
        self.formLayout.addRow("学习率: ", self.edit_learning_rate)
        self.formLayout.addRow("批次大小: ", self.edit_batch_size)
        self.formLayout.addRow("训练步长: ", self.edit_epoch)

        self.edit_step_size = QLineEdit()
        self.edit_step_gamma = QLineEdit()
        self.edit_milestones = QLineEdit()
        self.edit_mile_gamma = QLineEdit()
        self.edit_T_max = QLineEdit()
        self.scheduler_params_widget = ComboOptions()
        
        self.scheduler_params_widget.add_item_input(
            item="StepLR",
            para_labels=["步长:", "衰减率:"],
            para_inputs=[self.edit_step_size, self.edit_step_gamma],
        )
        self.scheduler_params_widget.add_item_input(
            item="MultiStepLR",
            para_labels=["里程碑:", "衰减率:"],
            para_inputs=[self.edit_milestones, self.edit_mile_gamma],
        )
        self.scheduler_params_widget.add_item_input(
            item="CosineAnnealingLR",
            para_labels=["最大周期:"],
            para_inputs=[self.edit_T_max],
        )
        self.scheduler_params_widget.combo_box.setCurrentIndex(2)

        self.hLayout_btns1 = QHBoxLayout()
        self.btn_start_train = QPushButton("开始训练")
        self.btn_increase_train = QPushButton("增量训练")
        self.btn_increase_train.setEnabled(False)
        self.btn_save_model = QPushButton("保存模型")
        self.hLayout_btns1.addWidget(self.btn_start_train)
        self.hLayout_btns1.addWidget(self.btn_increase_train)

        self.hLayout_btns2 = QHBoxLayout()
        self.hLayout_btns2.addWidget(self.btn_pause_resume)
        self.hLayout_btns2.addWidget(self.btn_stop)

        self.vLayout.addStretch()
        self.vLayout.addLayout(self.formLayout)
        self.vLayout.addWidget(QLabel("学习率调度器:"))
        self.vLayout.addWidget(self.scheduler_params_widget)
        self.vLayout.addLayout(self.hLayout_btns1)
        self.vLayout.addWidget(self.progress_bar)
        self.vLayout.addLayout(self.hLayout_btns2)
        self.vLayout.addWidget(self.btn_save_model)
        self.vLayout.addStretch()

        self.progress_bar.hide()
        self.btn_pause_resume.hide()
        self.btn_stop.hide()
        self.btn_save_model.hide()

    def show_progress_bar_and_stop_btn(self):
        self.progress_bar.show()
        self.btn_pause_resume.show()
        self.btn_stop.show()
        self.btn_save_model.show()

    def on_btn_pause_resume(self):
        self.signal_pause_resume_train.emit(self.show_pause)
        text = "继续训练" if self.show_pause else "暂停训练"
        self.btn_pause_resume.setText(text)
        self.show_pause = not self.show_pause

    def hide_progress_bar_and_stop_btn(self):
        self.btn_pause_resume.setText("暂停训练")
        self.show_pause = True
        self.progress_bar.hide()
        self.btn_stop.hide()
        self.btn_pause_resume.hide()

    def update_progress_bar(self, val: int):
        self.progress_bar.setValue(val)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    main = ModelTrain()
    main.resize(400, 500)
    main.show()
    sys.exit(app.exec())
