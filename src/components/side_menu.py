from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QApplication,
)
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QIcon

from src.components.custom_btn import CustomBtn


class SideMenu(QWidget):
    signal_change_page = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("side_menu")
        self.btns = []
        self.btns_layout = QVBoxLayout(self)
        self.btns_layout.addStretch()
        self.btns_layout.setSpacing(10)
        self.btns_layout.setContentsMargins(10, 30, 0, 0)
        self.add_btn_tip(CustomBtn("menu", QIcon("src/icons/connect.png")), "采集数据")
        self.add_btn_tip(CustomBtn("menu", QIcon("src/icons/points.png")), "分析数据")
        self.add_btn_tip(CustomBtn("menu", QIcon("src/icons/nn.png")), "训练模型")
        self.add_btn_tip(CustomBtn("menu", QIcon("src/icons/compen.png")), "训练模型")

    # 添加按钮和按钮提示
    def add_btn_tip(self, btn: CustomBtn, btn_str: str):
        index = len(self.btns)
        btn.setToolTip(btn_str)
        btn.clicked.connect(lambda: self.adjust_btns(index))
        btn.clicked.connect(lambda: self.signal_change_page.emit(index))
        self.btns_layout.insertWidget(index, btn)
        self.btns.append(btn)

    def adjust_btns(self, index: int):
        for idx, btn in enumerate(self.btns):
            if idx != index:
                btn.setChecked(False)
            else:
                btn.setChecked(True)


if __name__ == "__main__":
    app = QApplication([])
    side_menu = SideMenu()
    for i in range(5):
        btn = QPushButton(f"Button {i + 1}")
        side_menu.add_btn_tip(btn, f"Button {i + 1} Tip")

    main_window = QWidget()
    layout = QVBoxLayout(main_window)
    layout.addWidget(side_menu)
    main_window.setWindowTitle("SideMenu Test")
    main_window.show()
    app.exec()
