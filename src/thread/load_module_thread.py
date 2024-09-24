from PyQt5.QtCore import QThread, pyqtSignal as Signal


class ModuleLoaderThread(QThread):
    # 使用自定义信号来传递导入后的模块
    module_loaded = Signal()

    def run(self):
        # 后台导入模块用来缓存

        import numpy
        import pandas
        import sklearn
        import scipy
        import skfuzzy
        import torch
        import torch_geometric

        self.module_loaded.emit()
