import logging
import threading

import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader
from PyQt5.QtCore import QThread, pyqtSignal as Signal


class ModelTrainThread(QThread):
    """创建模型训练线程"""

    signal_train_val_loss = Signal(tuple)
    signal_train_finished = Signal(tuple)

    def __init__(self, model, datasets, dataloader, kfold: KFold, train_para: dict):
        super().__init__()
        self.running = True
        self.paused = False
        self.pause_cond = threading.Condition(threading.Lock())

        self.kf = kfold
        self.datasets = datasets
        self.dataloader = dataloader
        self.model = model
        self.train_para = train_para

        self.best_model = None
        self.best_pred = None
        self.best_loss_index = None
        self.best_loss_value = float("inf")

    def run(self):
        # 损失值函数
        criterion = torch.nn.MSELoss()
        # 优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_para["lr"])
        # 获取学习率调度器
        optimizer_dict = self.train_para["optimizer_dict"]
        scheduler = self.get_scheduler(optimizer, optimizer_dict["type"], optimizer_dict["para"])
        # 创建监测整体预测情况的数据集
        test_dataloader = self.dataloader(
            self.datasets, batch_size=len(self.datasets), shuffle=False
        )
        # 单折训练步长
        num_epochs = self.train_para["epoch"]

        # K折交叉验证
        for fold, (train_idx, val_idx) in enumerate(self.kf.split(self.datasets)):
            train_subset = torch.utils.data.Subset(self.datasets, train_idx)
            val_subset = torch.utils.data.Subset(self.datasets, val_idx)

            train_dataloader = self.dataloader(
                train_subset, batch_size=self.train_para["batch_size"], shuffle=True
            )
            val_dataloader = self.dataloader(
                val_subset, batch_size=self.train_para["batch_size"], shuffle=True
            )

            train_dataloader_num = len(train_dataloader)
            val_dataloader_num = len(val_dataloader)

            # 添加验证损失增加的计数器, 用于实现早熟机制
            increase_count = 0
            previous_val_loss = float("inf")

            for epoch in range(num_epochs):
                # 每个epoch训练开始前获取锁，检查是否处于暂停状态，如果处于暂停状态则暂停训练，一直等到锁的释放
                with self.pause_cond:
                    while self.paused:
                        self.pause_cond.wait()
                # 检测是否停止了训练
                if not self.running:
                    break

                self.model.train()
                total_train_loss = 0
                for batch_data in train_dataloader:
                    _, loss = self.get_loss(batch_data, criterion)
                    total_train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                avg_train_loss = total_train_loss / train_dataloader_num

                scheduler.step()

                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    # 计算验证集平均损失值
                    for batch_data in val_dataloader:
                        _, loss = self.get_loss(batch_data, criterion)
                        total_val_loss += loss.item()
                    # 计算整体数据的预测值和损失值
                    for batch_data in test_dataloader:
                        pred, loss = self.get_loss(batch_data, criterion)
                        test_loss = loss.item()

                avg_val_loss = total_val_loss / val_dataloader_num

                # 发送训练过程中的参数到主线程
                self.signal_train_val_loss.emit(
                    (
                        fold,
                        epoch,
                        num_epochs,
                        avg_train_loss,
                        avg_val_loss,
                        test_loss,
                        pred.squeeze(1),
                    )
                )
                # 记录最优模型、最好的预测结果和出现的位置
                if test_loss < self.best_loss_value:
                    self.best_fold = fold
                    self.best_loss_index = epoch
                    self.best_loss_value = test_loss
                    self.best_model = self.model
                    self.best_pred = pred.squeeze(1)

                # 检查验证损失是否增加
                if avg_val_loss > previous_val_loss:
                    increase_count += 1
                else:
                    increase_count = 0  # 重置计数器

                previous_val_loss = avg_val_loss

                # 如果增加计数达到 10 次，停止当前折的训练
                if increase_count >= 10:
                    print(f"在第{fold+1}折，第{epoch+1}epoch停止训练, 因为验证损失一直增加.")
                    break

        # 循环结束或者break后才会走到这里，表示训练结束
        # 向主线程发送训练结束信号
        self.signal_train_finished.emit(
            (
                self.best_fold,
                self.best_loss_index,
                self.best_loss_value,
                self.best_model,
                self.best_pred,
            )
        )

    def get_scheduler(self, optimizer, opt_type, opt_para):
        """
        获取学习率调度器
        -----
        paras:
        - optimizer: 优化器
        - opt_type: 学习率调度器类型
        - opt_para: 学习率调度器参数
        """
        match opt_type:
            case "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=opt_para["step_size"],
                    gamma=opt_para["step_gamma"],
                )
            case "MultiStepLR":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=opt_para["milestones"],
                    gamma=opt_para["mile_gamma"],
                )
            case "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=opt_para["T_max"]
                )
        return scheduler

    def get_loss(self, batch_data, criterion):
        """
        返回模型输出和损失值
        -----
        paras:
        - batch_data: 训练数据
        - criterion: 损失函数
        """
        if self.dataloader is GeometricDataLoader:
            outputs = self.model(batch_data)
            loss = criterion(outputs, batch_data.y.unsqueeze(-1))
        elif self.dataloader is TorchDataLoader:
            batch_X, batch_y = batch_data
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)

        return outputs, loss

    def stop(self):
        self.running = False
        self.resume()

    def pause(self):
        logging.debug("self.pause()被调用")
        with self.pause_cond:
            logging.debug("TrainingThread: 成功获取到线程锁用于暂停训练")
            self.paused = True

    def resume(self):
        logging.debug("self.resume()被调用")
        with self.pause_cond:
            logging.debug("TrainingThread: 成功获取到线程锁用于恢复训练")
            self.paused = False
            self.pause_cond.notify()
