import numpy as np


class PolyLeastSquares:
    """
    多阶无交叉项多项式拟合模块，默认二阶
    """

    def __init__(self, degree=2):
        """
        响应面模型系数最小二乘法拟合。
        :param degree: 阶数
        """
        self.degree = degree  # 默认最高阶是2
        self.coefficients = None

    # 根据待拟合的温度和热误差计算出各项系数存储在类属性中
    # 输入:
    # X - (m, n), m行n个温度传感器的数据
    # y - (m, 1), m行对应的热误差数据
    def fit(self, X, y) -> None:
        # 自动处理以列表存储的数组
        X = np.array(X) if isinstance(X, list) else X
        y = np.array(y) if isinstance(y, list) else y

        A = self.__get_matrix_A(X)
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.coefficients = coef.reshape(-1)

    # 根据温度数据计算得到热误差
    # 输入：
    # X - (m, n), m行n个温度传感器的数据
    def predict(self, X) -> np.ndarray:
        # 自动处理以列表存储的数组
        X = np.array(X) if isinstance(X, list) else X

        A = self.__get_matrix_A(X)
        return np.dot(A, self.coefficients)

    # 从低阶到高阶返回系数, 格式为numpy.array
    def get_coefficients(self) -> np.ndarray:
        """依此返回常数项、一次项、二次项系数"""
        return self.coefficients

    # 根据温度数据得到设计矩阵
    def __get_matrix_A(self, X) -> np.ndarray:
        # 获取设计矩阵A
        row, col = X.shape
        num_terms = col * self.degree + 1
        A = np.ones((row, num_terms))
        index = 1
        for i in range(col):
            for d in range(1, self.degree + 1):
                A[:, index] = X[:, i] ** d
                index += 1
        return A
