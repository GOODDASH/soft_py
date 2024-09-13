from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton


# 简单的对PushButton加载图标和设置尺寸
class CustomBtn(QPushButton):
    def __init__(self, btn_type: str, btn_icon: QIcon = None):
        super().__init__()
        match btn_type:
            case "menu":
                self.setFixedSize(60, 60)
                self.setCheckable(True)
                self.setObjectName("menuBtn")
                self.setIcon(btn_icon)
                self.setIconSize(QSize(40, 40))
            case "top":
                self.setFixedSize(50, 50)
                self.setCheckable(True)
                self.setObjectName("menuBtn")
                self.setIcon(btn_icon)
                self.setIconSize(QSize(45, 45))
            case "switch":
                self.setFixedSize(50, 50)
                self.setObjectName("swBtn")
                self.setIcon(btn_icon)
                self.setIconSize(QSize(45, 45))
