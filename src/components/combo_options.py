from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QComboBox,
    QStackedWidget,
    QFormLayout,
    QLineEdit,
    QApplication,
)


# 可以复用的下拉选项
class ComboOptions(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.vlayout = QVBoxLayout(self)
        self.combo_box = QComboBox()
        self.stacked_widget = QStackedWidget()
        self.vlayout.addWidget(self.combo_box)
        self.vlayout.addWidget(self.stacked_widget)
        self.vlayout.setSpacing(10)
        self.vlayout.setContentsMargins(0, 0, 0, 0)

        self.combo_box.currentIndexChanged.connect(self.change_page)

    def add_item_input(self, item: str, para_labels: list[str], para_inputs: list[QWidget]) -> None:
        assert len(para_labels) == len(para_inputs)
        self.combo_box.addItem(item)
        widget = QWidget()
        flayout = QFormLayout(widget)
        flayout.setContentsMargins(20, 5, 20, 5)
        flayout.setSpacing(10)
        for para_label, para_input in zip(para_labels, para_inputs):
            flayout.addRow(para_label, para_input)
        self.stacked_widget.addWidget(widget)

    def change_page(self, index) -> None:
        self.stacked_widget.setCurrentIndex(index)
        cur_widget = self.stacked_widget.currentWidget()
        if cur_widget:
            # 根据不同的组件，调整高度
            self.stacked_widget.setMaximumHeight(cur_widget.sizeHint().height())

    def get_cur_combo(self) -> str:
        return self.combo_box.currentText()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = ComboOptions()
    w.resize(800, 600)
    w.add_item_input(
        "test1",
        ["输入1", "输入2", "输入3"],
        [QLineEdit("1-1"), QLineEdit("1-2"), QLineEdit("1-3")],
    )
    w.add_item_input("test2", ["输入2-1", "输入2-2"], [QLineEdit("2-1"), QLineEdit("2-2")])
    w.show()
    sys.exit(app.exec())
