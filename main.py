import sys
from PyQt5.QtWidgets import QApplication
from src.controller import Controller


# TODO: 样式变量, 将qss文件中的值设为变量
# TODO: 替换QMessageBox为自定义浮窗

if __name__ == "__main__":
    # from PyQt5.QtCore import Qt
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    controller = Controller()
    sys.exit(app.exec())
