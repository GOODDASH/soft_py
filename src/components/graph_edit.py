import math
import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal as Signal, Qt, QRectF
from PyQt5.QtGui import QPen, QColor, QPainter, QFont


class NodeItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, x, y, index, radius=25):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.setPos(x, y)
        self.index = index
        self.radius = radius
        self.setPen(QPen(Qt.PenStyle.NoPen))
        self.setBrush(QColor("#94B9AF"))
        self.setZValue(1)  # 确保节点在线条之上

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # 启用抗锯齿
        super().paint(painter, option, widget)
        painter.setFont(QFont("Arial", 10))
        text_rect = QRectF(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)
        painter.setPen(QColor(Qt.GlobalColor.black))  # 设置画笔颜色为黑色
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(self.index))

    def setSelected(self, selected):
        if selected:
            self.setBrush(QColor("#90A583"))  # 选择时的颜色
        else:
            self.setBrush(QColor("#94B9AF"))  # 未选择时的颜色


class GraphEdit(QtWidgets.QMainWindow):
    edge_list_signal = Signal(list)  # 定义一个信号

    def __init__(self, node_count, edge_list=None):
        super().__init__()
        self.node_count = node_count
        self.nodes = []
        self.edges = {}  # 使用字典存储边
        self.selected_nodes = []

        self.setWindowTitle("编辑图")
        self.resize(400, 450)

        self.widget = QtWidgets.QWidget(self)
        self.widget.setStyleSheet("background-color: #e5e5e5")
        self.setCentralWidget(self.widget)

        self.layout = QtWidgets.QVBoxLayout(self.widget)
        self.view = QtWidgets.QGraphicsView(self)
        # self.view.setStyleSheet("background-color: #e5e5e5; border: none;")
        self.view.setGeometry(0, 0, 400, 400)
        self.scene = QtWidgets.QGraphicsScene()
        self.scene.setBackgroundBrush(QColor("#e5e5e5"))

        self.view.setScene(self.scene)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.redraw_button = QtWidgets.QPushButton("重绘")
        self.redraw_button.clicked.connect(self.redraw_graph)
        self.confirm_button = QtWidgets.QPushButton("确定")
        self.confirm_button.clicked.connect(self.confirm_graph)

        self.button_layout.addWidget(self.redraw_button)
        self.button_layout.addWidget(self.confirm_button)

        self.layout.addWidget(self.view)
        self.layout.addLayout(self.button_layout)

        self.create_nodes()

        if edge_list:
            self.draw_edges(edge_list)

    def create_nodes(self):
        radius = 150
        center_x = 200
        center_y = 200
        for i in range(self.node_count):
            angle = 2 * math.pi * i / self.node_count
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            node = NodeItem(x, y, i)
            self.nodes.append(node)
            self.scene.addItem(node)

    def draw_edges(self, edge_list):
        for i in range(len(edge_list[0])):
            node1_index = edge_list[0][i]
            node2_index = edge_list[1][i]
            node1 = self.nodes[node1_index]
            node2 = self.nodes[node2_index]
            self.add_edge(node1, node2)

    def mousePressEvent(self, event):
        item = self.view.itemAt(event.pos())
        if isinstance(item, NodeItem):
            if item in self.selected_nodes:
                # 如果节点已被选择，取消选择并从列表中移除
                self.selected_nodes.remove(item)
                item.setSelected(False)
            else:
                self.selected_nodes.append(item)
                item.setSelected(True)
            if len(self.selected_nodes) == 2:
                node1, node2 = self.selected_nodes[0], self.selected_nodes[1]
                if (node1.index, node2.index) in self.edges:
                    self.remove_edge(node1, node2)
                else:
                    self.add_edge(node1, node2)
                for node in self.selected_nodes:
                    node.setSelected(False)
                self.selected_nodes.clear()

    def add_edge(self, node1, node2):
        if (node1.index, node2.index) not in self.edges:
            line = self.scene.addLine(
                node1.x(),
                node1.y(),
                node2.x(),
                node2.y(),
                QPen(QColor(Qt.GlobalColor.black), 2),
            )
            self.edges[(node1.index, node2.index)] = line
            self.edges[(node2.index, node1.index)] = line  # 添加反向边
            line.setZValue(-1)  # 确保线条在节点之下

    def remove_edge(self, node1, node2):
        if (node1.index, node2.index) in self.edges:
            line = self.edges.pop((node1.index, node2.index))
            self.edges.pop((node2.index, node1.index))
            self.scene.removeItem(line)

    def get_edge_list(self):
        edge_list = [[], []]
        for node1, node2 in self.edges.keys():
            edge_list[0].append(node1)
            edge_list[1].append(node2)
        return edge_list

    def redraw_graph(self):
        self.scene.clear()
        self.nodes = []
        self.edges = {}
        self.create_nodes()

    def confirm_graph(self):
        edge_list = self.get_edge_list()
        self.edge_list_signal.emit(edge_list)  # 发出信号
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = GraphEdit(10)
    ui.show()
    sys.exit(app.exec())
