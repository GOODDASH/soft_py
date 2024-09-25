import sys
from PyQt5.QtWidgets import QApplication
from src.controller import Controller

# TODO: 弄清楚轴数据中各种值的意义
# TODO: 采集中途可以设置停止\继续绘图，降低负担

if __name__ == "__main__":
    # from PyQt5.QtCore import Qt
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    controller = Controller()
    sys.exit(app.exec())
