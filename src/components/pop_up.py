from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, QTimer


class PopUp(QWidget):
    def __init__(self, message: str, display_time=None, parent=None):
        super().__init__(parent)
        self.objectName = "popup"
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Popup | Qt.NoDropShadowWindowHint)

        self.resize(500, 200)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        self.message_label = QLabel(message)
        self.message_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.message_label)

        self.setLayout(layout)
        self.setStyleSheet(
            """
                border-radius: 20px;
                border: 1px solid rgb(144, 164, 174);
            """
        )
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
        self.timer.setSingleShot(True)
        self.timer.start(display_time)
