import sys
import torch
import torch.nn as nn
import time
from PyQt5.QtCore import pyqtSignal, QThread, QObject
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

# 一个简单的 PyTorch 模型示例
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 自定义的 Worker 类，负责进行模型的前向传播计算
class ModelWorker(QObject):
    resultReady = pyqtSignal(torch.Tensor)  # 计算结果通过信号发送给主线程

    def __init__(self, model):
        super(ModelWorker, self).__init__()
        self.model = model
        self.running = False

    def start_calculation(self, input_tensor):
        if not self.running:
            self.running = True
            result = self.model(input_tensor)
            time.sleep(1)  # 模拟耗时计算
            self.resultReady.emit(result)
            self.running = False

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.initUI()

        # 创建 PyTorch 模型
        self.model = SimpleModel()

        # 创建后台线程和 ModelWorker
        self.thread = QThread()
        self.worker = ModelWorker(self.model)
        self.worker.moveToThread(self.thread)

        # 启动后台线程
        self.thread.start()

        # 连接信号与槽
        self.worker.resultReady.connect(self.on_result_ready)
        
    def initUI(self):
        # 设置界面
        self.setWindowTitle("PyTorch Forward Calculation in QThread")
        self.setGeometry(100, 100, 300, 200)

        # 创建按钮
        self.button = QPushButton("Start Calculation", self)
        self.button.clicked.connect(self.start_calculation)

        # 布局
        layout = QVBoxLayout()
        layout.addWidget(self.button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_calculation(self):
        # 创建输入张量并启动计算
        input_tensor = torch.randn(1, 10)
        QThread.currentThread().eventDispatcher().flush()  # 确保 GUI 线程的事件循环在计算时不被阻塞
        self.worker.start_calculation(input_tensor)

    def on_result_ready(self, result):
        # 显示结果
        print("Calculation Result: ", result)

    def closeEvent(self, event):
        # 窗口关闭时，确保线程安全退出
        self.thread.quit()
        self.thread.wait()
        event.accept()

# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
