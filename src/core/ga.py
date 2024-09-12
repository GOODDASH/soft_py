# -*- coding: utf-8 -*-
"""
Created on Sep 14 2023

@author: Out_From_Rest

"""

import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.core.func import create_random_posit


class GA:
    """
    在输入规定生成集合里产生子集的特征提取遗传算法
    """

    def __init__(self, loss_func, dim, pop_size, max_epoch, idx):
        self.loss_func = loss_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_epoch = max_epoch
        self.idx = idx

        self.posit = np.array([create_random_posit(idx, dim) for _ in range(pop_size)])
        self.loss = np.zeros(pop_size)

        self.loss_alpha = np.inf
        self.loss_beta = np.inf
        self.loss_delta = np.inf
        self.posi_alpha = np.zeros(dim)
        self.posi_beta = np.zeros(dim)
        self.posi_delta = np.zeros(dim)

        self.epoch = 1
        self.curve = np.zeros(max_epoch)

    def opt(self):
        """
        主程序
        """
        while self.epoch < self.max_epoch + 1:
            self.get_leaders()
            self.sort_posit()
            for i in range(self.pop_size):
                if i < self.pop_size / 2:
                    self.posit[i] = self.crossover_best(i)
                else:
                    self.posit[i] = self.crossover_random()

            self.curve[self.epoch - 1] = self.loss_alpha
            logging.debug(f"Iteration {self.epoch} Best fitness= {self.loss_alpha}")
            self.epoch += 1
        sf = np.where(self.posi_alpha == 1)

        return sf[0]

    def get_leaders(self):
        """
        计算整个群体的损失值矩阵（用来对群体进行排序），
        记录整个搜索过程中的前三个最优的解
        """
        self.loss = self.loss_func(self.posit)
        for i in range(self.pop_size):
            loss_ind = self.loss[i]
            if loss_ind < self.loss_alpha:
                self.loss_alpha = loss_ind
                self.posi_alpha = self.posit[i].copy()
            if self.loss_alpha < loss_ind < self.loss_beta:
                self.loss_beta = loss_ind
                self.posi_beta = self.posit[i].copy()
            if self.loss_alpha < loss_ind < self.loss_delta and loss_ind > self.loss_beta:
                self.loss_delta = loss_ind
                self.posi_delta = self.posit[i].copy()

    def sort_posit(self):
        """
        根据对应的损失值对解群体进行排序
        """
        sorted_idx = np.argsort(self.loss)
        self.posit[:] = self.posit[sorted_idx]

    def crossover_best(self, i):
        """
        遍历每个分类索引，
        从随机解和三个最优解中选择一个解，
        对应位置的值赋值给交叉结果
        """
        crossed_posi = np.zeros(self.dim)
        for cluster_idx in self.idx:
            selected_array = random.choice(
                [self.posi_alpha, self.posi_beta, self.posi_delta, self.posit[i]]
            )
            crossed_posi[cluster_idx] = selected_array[cluster_idx]

        return crossed_posi

    def crossover_random(self):
        """
        遍历每个分类索引，
        从随机解和最优解中选择一个解，
        对应位置的值赋值给交叉结果
        """
        crossed_posi = np.zeros(self.dim)
        for cluster_idx in self.idx:
            random_posi = create_random_posit(self.idx, self.dim)
            selected_best = random.choice([self.posi_alpha, self.posi_beta, self.posi_delta])
            selected_array = random.choice([selected_best, random_posi])
            crossed_posi[cluster_idx] = selected_array[cluster_idx]

        return crossed_posi

    def plot_curve(self):
        plt.figure(figsize=(6, 6))
        plt.suptitle("GA")
        plt.title("loss curve [" + str(round(self.curve[-1], 4)) + "]")
        plt.plot(np.arange(1, self.max_epoch + 1), self.curve, label="loss")
        plt.grid()
        plt.legend()
        plt.show()
