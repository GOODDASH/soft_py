from PyQt5.QtCore import Qt, pyqtSignal as Signal
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QComboBox,
    QFormLayout,
    QLineEdit,
    QHBoxLayout,
    QLabel,
    QFileDialog,
    QStackedWidget,
    QMessageBox,
    QGroupBox,
)

from src.components.graph_edit import GraphEdit


class ModelChoose(QGroupBox):
    signal_import_model = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("设置模型")

        # GAT-LSTM需要的输入
        self.edit_num_heads = QLineEdit()
        self.edit_gnn_dim = QLineEdit()
        self.edit_lstm_dim = QLineEdit()
        self.edit_num_nodes = QLineEdit()
        self.edit_seq_len = QLineEdit()
        self.btn_set_edge_index = QPushButton("编辑图节点")
        self.edit_edge_start = QLineEdit()
        self.edit_edge_end = QLineEdit()

        # 示例BPNN需要的输入
        self.edit_bpnn_hidden_dim = QLineEdit(self)

        self.vLayout = QVBoxLayout(self)
        self.vLayout.setSpacing(10)

        self.fLayout_type = QFormLayout()
        self.model_type = QComboBox()
        self.model_type.addItem("GAT-LSTM")
        self.model_type.addItem("BPNN")
        self.fLayout_type.addRow("模型类型:", self.model_type)

        self.btn_import_exist_model = QPushButton("导入已有模型文件")

        self.model_stacked = self.create_model_type_widget()

        self.vLayout.addStretch()
        self.vLayout.addLayout(self.fLayout_type)
        self.vLayout.addWidget(self.model_stacked)
        self.vLayout.addWidget(self.btn_import_exist_model)
        self.vLayout.addStretch()

        self.update_model_type_widget(self.model_type.currentText())

        self.model_stacked.currentChanged.connect(self.adjust_stacked)
        self.btn_import_exist_model.clicked.connect(self.on_btn_import_exist_model)
        self.model_type.currentTextChanged.connect(self.update_model_type_widget)
        self.btn_set_edge_index.clicked.connect(self.on_btn_set_edge_index)

    def on_btn_import_exist_model(self):
        file_filter = "Pytorch File(*.pth);;All files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", file_filter)
        if file_path:
            signal_para = self.get_model_para()
            self.signal_import_model.emit(signal_para)

    def get_model_para(self) -> dict:
        model_para = dict()
        match self.model_type.currentText():
            case "GAT-LSTM":
                model_para["model_type"] = "GAT-LSTM"
                model_para["num_heads"] = int(self.edit_num_heads.text())
                model_para["gnn_dim"] = int(self.edit_gnn_dim.text())
                model_para["lstm_dim"] = int(self.edit_lstm_dim.text())
                model_para["num_nodes"] = int(self.edit_num_nodes.text())
                model_para["seq_len"] = int(self.edit_seq_len.text())
                edge_list = [
                    list(map(int, self.edit_edge_start.text().split(","))),
                    list(map(int, self.edit_edge_end.text().split(","))),
                ]
                model_para["edge_list"] = edge_list
            case "BPNN":
                model_para["model_type"] = "BPNN"
                model_para["hidden_dim"] = int(self.edit_bpnn_hidden_dim.text())
        return model_para

    def create_model_type_widget(self) -> QStackedWidget:
        model_stacked = QStackedWidget()

        gat_lstm_widget = QWidget()
        gat_lstm_widget_layout = QVBoxLayout(gat_lstm_widget)
        gat_lstm_widget_layout.setContentsMargins(30, 10, 30, 10)
        bpnn_formlayout = QFormLayout()
        bpnn_formlayout.addRow("注意力头数:", self.edit_num_heads)
        bpnn_formlayout.addRow("GAT隐藏维度: ", self.edit_gnn_dim)
        bpnn_formlayout.addRow("LSTM隐藏维度: ", self.edit_lstm_dim)
        bpnn_formlayout.addRow("测点个数: ", self.edit_num_nodes)
        bpnn_formlayout.addRow("时间窗口长度: ", self.edit_seq_len)
        hLayout1 = QHBoxLayout()
        hLayout2 = QHBoxLayout()
        hLayout1.addWidget(QLabel("起始点:"))
        hLayout1.addWidget(self.edit_edge_start)
        hLayout2.addWidget(QLabel("终止点:"))
        hLayout2.addWidget(self.edit_edge_end)
        gat_lstm_widget_layout.addLayout(bpnn_formlayout)
        gat_lstm_widget_layout.addWidget(self.btn_set_edge_index)
        gat_lstm_widget_layout.addLayout(hLayout1)
        gat_lstm_widget_layout.addLayout(hLayout2)
        gat_lstm_widget_layout.setSpacing(10)

        bpnn_widget = QWidget()
        bpnn_widget_layout = QVBoxLayout(bpnn_widget)
        bpnn_widget_layout.setContentsMargins(30, 10, 30, 10)
        bpnn_formlayout = QFormLayout()
        bpnn_formlayout.addRow("隐藏维度:", self.edit_bpnn_hidden_dim)
        bpnn_widget_layout.addLayout(bpnn_formlayout)

        model_stacked.addWidget(gat_lstm_widget)
        model_stacked.addWidget(bpnn_widget)

        return model_stacked

    def get_edge_index(self):
        edge_index = []
        start_index_list = list(map(int, self.edit_edge_start.text().split(",")))
        end_index_list = list(map(int, self.edit_edge_end.text().split(",")))
        edge_index.append(start_index_list)
        edge_index.append(end_index_list)

        return edge_index

    def adjust_stacked(self, index):
        current_widget = self.model_stacked.currentWidget()
        if current_widget:
            height = current_widget.sizeHint().height()
            self.model_stacked.setMaximumHeight(height)
            self.model_stacked.resize(self.model_stacked.width(), height)

    def update_model_type_widget(self, model_type):
        match model_type:
            case "GAT-LSTM":
                self.model_stacked.setCurrentIndex(0)
            case "BPNN":
                self.model_stacked.setCurrentIndex(1)

    def on_btn_set_edge_index(self):
        # 图形编辑边列表
        edge_list = self.get_edge_index()
        node_count = int(self.edit_num_nodes.text())
        self.graph_edit_window = GraphEdit(node_count, edge_list)
        self.graph_edit_window.show()
        # 编辑完成后将结果显示到界面
        self.graph_edit_window.edge_list_signal.connect(self.set_ui_edge_list)

    def set_ui_edge_list(self, changed_edge_index):
        start_index_list = ",".join(map(str, changed_edge_index[0]))
        end_index_list = ",".join(map(str, changed_edge_index[1]))
        self.edit_edge_start.setText(start_index_list)
        self.edit_edge_end.setText(end_index_list)
        QMessageBox.information(self, "提示", "边列表已更新")
