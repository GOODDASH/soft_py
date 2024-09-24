from PyQt5.QtCore import QThread, pyqtSignal as Signal


class GATSPThread(QThread):
    signal_tsp_result = Signal(object)

    def __init__(self, loss, num_sensors, pop_size, iters, cluster_res):
        super().__init__()
        self.loss = loss
        self.num_sensors = num_sensors
        self.pop_size = pop_size
        self.iters = iters
        self.cluster_res = cluster_res

    def run(self):
        from src.core.ga import GA

        optimizer = GA(self.loss, self.num_sensors, self.pop_size, self.iters, self.cluster_res)
        self.signal_tsp_result.emit(optimizer.opt())
