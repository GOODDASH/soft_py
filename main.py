import sys
from PyQt5.QtWidgets import QApplication
from src.controller import Controller


# TODO: 大工程：将所有输入的值存储在state中，每次输入值改变都直接改变state中的值，不用每次从界面重新读取

if __name__ == "__main__":
    # from PyQt5.QtCore import Qt
    # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    controller = Controller()
    sys.exit(app.exec())
