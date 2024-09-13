import numpy as np
from scipy.interpolate import interp1d
from skfuzzy.cluster import cmeans as fcm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans


def get_fcm_cluster_res(data, ncenter):
    """
    获取聚类后的结果 - 输出（聚类数*每类对应的索引）的二维数组
    """
    _, u, _, _, _, _, _ = fcm(data, ncenter, 2, error=0.001, maxiter=2000, init=None)
    maxU = np.max(u, 0)
    cluster_idx = [np.where(u[i, :] == maxU)[0] for i in range(ncenter)]

    return cluster_idx


def get_kmeans_cluster_res(data, ncenter):
    kmeans = KMeans(n_clusters=ncenter, n_init=10)
    kmeans.fit(data.T)
    labels = kmeans.labels_
    cluster_idx = [np.where(labels == i)[0] for i in range(ncenter)]

    return cluster_idx


def get_gra_res(x0, xi, rho=0.5):
    """
    计算母序列x0和子序列xi之间的灰色关联度, 返回(i,)的关联度数组
    """
    # 均值化
    # x0_norm = x0 / np.mean(x0, 0)
    # xi_norm = xi / np.mean(xi, 0)

    # 区间化
    x0_norm = x0 / np.max(x0, 0)
    xi_norm = xi / np.max(xi, 0)

    # 母序列与子序列相减，找到两极最小、最大差
    abs_sub = np.abs(xi_norm - x0_norm)
    mmax = np.max(abs_sub)
    mmin = np.min(abs_sub)

    # 求每个指标的关联系数再平均得到灰色关联度
    rho_matrix = (mmin + rho * mmax) / (abs_sub + rho * mmax)
    res = np.mean(rho_matrix, 0)

    return res


def gra(X_data, T_data, idx):
    """
    常规 聚类+灰色关联度 的测温点筛选结果
    """
    gra = get_gra_res(X_data.reshape(-1, 1), T_data)
    selected_idx = np.array([], dtype=int)
    for i in range(len(idx)):
        c = np.argmax(gra[idx[i]])
        selected_idx = np.append(selected_idx, idx[i][c])

    return selected_idx


def cor(X_data, T_data, idx):
    """
    常规 聚类+相关系数 的测温点筛选结果
    """
    corr = np.corrcoef(T_data.T, X_data)
    corr_with_X = corr[:-1, -1]
    selected_idx = np.array([], dtype=int)
    for i in range(len(idx)):
        c = np.argmax(np.abs(corr_with_X[idx[i]]))
        selected_idx = np.append(selected_idx, idx[i][c])

    return selected_idx


def create_random_posit(cluster_res, dim):
    """
    从聚类结果中生成满足“每个类别中选一个”的随机解
    """
    random_posit = np.zeros(dim)
    for chosen_idx in [np.random.choice(np.ravel(cluster)) for cluster in cluster_res]:
        random_posit[chosen_idx] = 1

    return random_posit


def get_kfold_cv(num_folds, Tdata):
    """
    获取交叉验证所用的k折索引
    """
    kf = KFold(n_splits=num_folds, shuffle=True)
    cv = list(kf.split(Tdata))

    return cv


def loss_func(posit, Tdata, Xdata, idx, cv):
    """
    损失函数
    - 之所以传入已经划分好的索引，是不想出现选取同一组特征时出现不同的损失值
    """
    loss = np.zeros(posit.shape[0])
    for i in range(posit.shape[0]):
        xidx = np.where(posit[i, :] == 1)[0]
        chosen_Tdata = Tdata[:, xidx]  # 选取的特征
        model = LinearRegression()
        rmse_scores = []
        for train_index, test_index in cv:
            X_train, X_test = chosen_Tdata[train_index], chosen_Tdata[test_index]
            y_train, y_test = Xdata[train_index], Xdata[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)
        loss[i] = np.mean(rmse_scores)

    return loss


def check_tsp_per(datas, data_cat, tsp):
    """
    计算所选测温点进行MLR建模后的模型拟合效果
    """
    model = LinearRegression()
    model.fit(data_cat[:, tsp], data_cat[:, -1])
    pred_list = []
    for data in datas:
        pred = model.predict(data[:, tsp])
        pred_list.append(pred)
    pred_cat = np.hstack(pred_list)
    rmse = np.sqrt(mean_squared_error(pred_cat, data_cat[:, -1]))

    return pred_list, rmse, model.coef_, model.intercept_


def interpolate_data(data: np.ndarray, num_interpolations: int) -> np.ndarray:
    # TODO: 使用一次还是二次插值
    """
    对输入的二维数组进行插值，在每对相邻行之间插入指定数量的插值行。
    使用二次插值法，更适合每列代表曲线的情况。

    参数:
    data (np.ndarray): 原始二维数组
    num_interpolations (int): 每对相邻行之间插入的插值数量

    返回:
    np.ndarray: 插值后的数组
    """
    if num_interpolations == 0:
        return data
    rows, cols = data.shape
    new_rows = rows + num_interpolations * (rows - 1)
    interpolated_data = np.empty((new_rows, cols))

    # 对每一列进行二次插值
    for col in range(cols):
        y = data[:, col]
        x = np.arange(len(y))
        f = interp1d(x, y, kind="linear")  # 使用二次插值函数
        new_x = np.linspace(0, len(y) - 1, num=new_rows)
        interpolated_data[:, col] = f(new_x)

    return interpolated_data


def extract_numbers(data):
    """
    递归地从字典或列表中提取所有的数字（int或float）

    参数:
    data (dict or list): 字典或列表

    返回:
    list: 字典或列表中所有的数字
    """
    numbers = []
    if isinstance(data, (dict, list)):
        for item in data.values() if isinstance(data, dict) else data:
            numbers.extend(extract_numbers(item))
    elif isinstance(data, (int, float)):
        numbers.append(data)
    return numbers
