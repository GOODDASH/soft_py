import sys
from PyQt5.QtWidgets import QApplication, QPushButton, QLabel, QWidget
from PyQt5.QtCore import Qt, QTimer, QPoint, QSize
from PyQt5.QtGui import QMovie

class GifToolTip(QWidget):
    def __init__(self, gif_path, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.ToolTip)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowOpacity(0.9)

        # 创建一个 QLabel 用来显示 GIF 动画
        self.label = QLabel(self)
        self.label.setFixedSize(QSize(500, 300))
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)
        self.movie.start()

        # 设置窗口大小和位置
        self.adjustSize()
    
    def show_tooltip(self, pos):
        self.move(pos)
        self.show()

class CustomButton(QPushButton):
    def __init__(self, text, gif_path, parent=None):
        super().__init__(text, parent)
        self.tooltip = GifToolTip(gif_path, self)

    def enterEvent(self, event):
        # 获取按钮的全局位置
        global_pos = self.mapToGlobal(QPoint(0, self.height()))
        self.tooltip.show_tooltip(global_pos)

    def leaveEvent(self, event):
        self.tooltip.hide()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    main_window = QWidget()
    main_window.setWindowTitle("Custom Tooltip Example")
    
    button = CustomButton("Hover me", "doc\sample.gif", main_window)
    button.setGeometry(50, 50, 100, 50)
    
    main_window.setGeometry(100, 100, 300, 200)
    main_window.show()
    
    sys.exit(app.exec_())
