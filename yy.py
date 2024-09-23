from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QPushButton, QLabel, QLineEdit
from PyQt5.QtCore import Qt, QTimer
import sys

class CustomPopup(QWidget):
    def __init__(self, message="这是一个自定义弹窗", display_time=None, parent=None):
        super().__init__(parent)
        # 去掉标题栏，设置为Popup类型
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup | Qt.NoDropShadowWindowHint)
        # 设置透明背景
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 设置弹窗大小比主窗口小
        self.resize(300, 200)

        # 布局和控件
        layout = QVBoxLayout()
        
        # 显示自定义消息的标签
        self.message_label = QLabel(message)
        layout.addWidget(self.message_label)

        # 关闭按钮
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        # 设置布局和样式
        self.setLayout(layout)
        self.setStyleSheet("""
            background-color: white; 
            border-radius: 10px; 
            border: 2px solid #888;
        """)

        # 判断是否设置了显示时间
        if display_time is not None:
            self.start_timer(display_time)

    def set_message(self, message):
        """设置自定义消息"""
        self.message_label.setText(message)

    def start_timer(self, display_time):
        """启动计时器，自动关闭窗口"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.close)
        self.timer.setSingleShot(True)  # 只触发一次
        self.timer.start(display_time)  # 开始计时

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('主窗口')
        self.setGeometry(100, 100, 600, 400)  # 设置主窗口大小

        layout = QVBoxLayout()
        
        # 输入自定义消息的文本框
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("在这里输入自定义消息")
        layout.addWidget(self.message_input)
        
        # 输入自动关闭时间的文本框（毫秒）
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("输入自动关闭时间（毫秒），如2000")
        layout.addWidget(self.time_input)
        
        # 显示弹窗按钮
        show_popup_button = QPushButton("显示弹窗")
        show_popup_button.clicked.connect(self.show_popup)
        layout.addWidget(show_popup_button)
        
        self.setLayout(layout)

    def show_popup(self):
        # 获取自定义消息
        message = self.message_input.text()
        if not message:
            message = "这是一个自定义弹窗"
        
        # 获取自动关闭时间
        display_time = self.time_input.text()
        if display_time.isdigit():
            display_time = int(display_time)
        else:
            display_time = None
        
        # 创建弹窗实例，并传递自定义消息和显示时间
        self.popup = CustomPopup(message, display_time, self)
        
        # 获取主窗口的几何位置和尺寸
        main_window_geometry = self.geometry()

        # 计算弹窗应该出现的位置（主窗口的中心）
        popup_width = self.popup.width()
        popup_height = self.popup.height()

        # 主窗口的中心点
        main_center_x = main_window_geometry.center().x()
        main_center_y = main_window_geometry.center().y()

        # 弹窗的左上角坐标
        popup_x = main_center_x - popup_width // 2
        popup_y = main_center_y - popup_height // 2

        # 移动弹窗到主窗口中心
        self.popup.move(popup_x, popup_y)
        self.popup.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())
