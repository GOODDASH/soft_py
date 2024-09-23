from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import sys

class PytorchLoaderThread(QThread):
    # 使用自定义信号来传递导入后的模块
    pytorch_loaded = pyqtSignal(object)

    def run(self):
        # 在后台线程中导入 PyTorch
        import torch

        # 模拟长时间导入操作
        # 你可以在这里执行一些初始化操作，如：torch.cuda.is_available() 等
        self.sleep(2)  # 模拟耗时操作

        # 发射信号，将导入的模块传递给主线程
        self.pytorch_loaded.emit(torch)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('主窗口')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()
        
        # 标签，用于显示状态信息
        self.status_label = QLabel("正在后台导入 PyTorch...")
        layout.addWidget(self.status_label)
        
        # 按钮，用于测试 PyTorch 的功能（在导入完成后才可用）
        self.test_button = QPushButton("测试 PyTorch")
        self.test_button.setEnabled(False)
        self.test_button.clicked.connect(self.test_pytorch)
        layout.addWidget(self.test_button)

        self.setLayout(layout)

        # 程序启动后自动加载 PyTorch
        self.load_pytorch()

    def load_pytorch(self):
        # 创建和启动后台线程
        self.loader_thread = PytorchLoaderThread()
        self.loader_thread.pytorch_loaded.connect(self.on_pytorch_loaded)
        self.loader_thread.start()

    def on_pytorch_loaded(self, pytorch_modules):
        # 后台线程导入完成，接收传递过来的模块
        self.torch = pytorch_modules
        
        # 更新状态显示
        self.status_label.setText("PyTorch 导入成功！")
        self.test_button.setEnabled(True)  # 启用测试按钮

        # 你可以在这里测试导入的模块
        print(f"PyTorch 版本: {self.torch.__version__}")

    def test_pytorch(self):
        # 测试 PyTorch 功能
        x = self.torch.tensor([1.0, 2.0, 3.0])
        y = self.torch.tensor([4.0, 5.0, 6.0])
        result = x + y
        self.status_label.setText(f"测试结果: {result}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
