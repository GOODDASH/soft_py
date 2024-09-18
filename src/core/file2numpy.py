from numpy import around as np_around
from pandas import read_csv, read_excel, read_table, DataFrame


# 根据文件地址读取数据文件并转换为numpy数组
# 输出：前n-1列为温度，最后一列n为热误差
def read_datafile_to_numpy(file_path, trans=False, sep=None, t_begin=0, t_end=0, e_idx=0):
    if file_path.endswith(".csv"):
        df = read_csv(file_path)
    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        # excel文件无需分隔符号
        df = read_excel(file_path)
    elif file_path.endswith(".txt"):
        df = read_table(file_path, sep=sep)
    else:
        raise ValueError("不支持的文件格式")
    data_array = df.iloc[:, list(range(t_begin, t_end + 1)) + [e_idx]].to_numpy()

    return data_array if trans == False else data_array.T


def write_datafile_from_numpy(data_array, file_path, sep=None):
    data_array = np_around(data_array, decimals=3)
    df = DataFrame(data_array)

    if file_path.endswith(".csv"):
        df.to_csv(file_path, sep=sep if sep else ",", index=False, header=False)
    elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
        df.to_excel(file_path, index=False, header=False)
    elif file_path.endswith(".txt"):
        df.to_csv(file_path, sep=sep if sep else "\t", index=False, header=False)
    else:
        raise ValueError("不支持的文件格式")
