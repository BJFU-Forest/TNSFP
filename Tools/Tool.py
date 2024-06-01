import os
from math import factorial

import numpy as np
import pandas as pd


# 计算组合数
def CmbinationNumber(n, m):
    if m <= n:
        a = factorial(n) / (factorial(n - m) * factorial(m))
    else:
        raise "n must >= m"
    return a


# 创建随机元组
class RandIntMatrix(object):
    def __init__(self, low, high, shape=(1)):
        self.low = low
        self.high = high
        self.shape = shape

    def rvs(self, random_state=None):
        np.random.seed(random_state)
        return np.random.randint(self.low, self.high, self.shape)


def check_file_path(file_path, make_dirs=True):
    """Gets the absolute file path.

    Args:
        file_path ([str): The path to the file.
        make_dirs (bool, optional): Whether to create the directory if it does not exist. Defaults to True.

    Raises:
        FileNotFoundError: If the directory could not be found.
        TypeError: If the input directory path is not a string.

    Returns:
        str: The absolute path to the file.
    """
    if isinstance(file_path, str):
        if file_path.startswith("~"):
            file_path = os.path.expanduser(file_path)
        else:
            file_path = os.path.abspath(file_path)

        file_dir = os.path.dirname(file_path)
        if not os.path.exists(file_dir) and make_dirs:
            os.makedirs(file_dir)

        return file_path

    else:
        raise TypeError("The provided file path must be a string.")


# 检查文件夹是否存在
def create_path(path):
    if not path.endswith("/"):
        fater_path = os.path.abspath(path + os.path.sep + "..")
    else:
        fater_path = path
    if not os.path.isdir(fater_path):
        os.makedirs(fater_path)

# 文件夹清理
def del_file(path):
    if os.path.isdir(path) or os.path.isfile(path):
        for i in os.listdir(path):
            path_file = os.path.join(path, i)  # 取文件绝对路径
            if os.path.isfile(path_file):
                os.remove(path_file)
            else:
                del_file(path_file)

# 在文件第一行写入
def write_raw_index(file_path, text):
    with open(file_path, 'r+', encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(text + '\n' + content)


# 保留nan的求和
def sum_without_nan(df):
    if df.count() == 0:
        sum = np.nan
    else:
        sum = np.nansum(df)
    return sum


# 用月均值填充缺失行
def fill_nan(df):
    # 清洗存在缺失值的行
    for col in df.columns.values[df.isnull().any()]:
        df[col] = df[col].fillna(df.groupby(df.min_ac.month)[col].transform('mean'))
    return df


# df时间截取
def select_time(df, periods, leap=False):
    start = pd.to_datetime(periods[0])
    end = pd.to_datetime(periods[1])
    df.min_ac = pd.to_datetime(df.min_ac)  # 时间列格式转换
    df = df[(df.min_ac >= start) & (df.min_ac <= end)]
    if leap:
        df = df[~((df.min_ac.month == 2) & (df.min_ac.day == 29))]
    return df


# 按季节分组
def season_classify(month):
    season = None
    Spring = {3, 4, 5}
    Summer = {6, 7, 8}
    Autumn = {9, 10, 11}
    Winter = {12, 1, 2}
    if month in Spring:
        season = "Spring"
    elif month in Summer:
        season = "Summer"
    elif month in Autumn:
        season = "Autumn"
    elif month in Winter:
        season = "Winter"
    return season


season_classify = np.frompyfunc(season_classify, 1, 1)


# 月尺度求季节尺度平均
def month2season(df, how2hightempor):
    df.min_ac = pd.to_datetime(df.min_ac)
    average = df.groupby([df.min_ac.month]).mean()
    season = average.groupby(season_classify(average.min_ac)).agg(how2hightempor)
    season = season.reindex(index=["Spring", "Summer", "Autumn", "Winter"])
    return season


# 转年尺度
def temporal2year(df, how2hightempor):
    df.min_ac = pd.to_datetime(df.min_ac)
    year = df.groupby(df.min_ac.year).agg(how2hightempor)
    return year


def data_standardizing(data, data_type, eigenvalue=None):
    """
    when the index value is greater, the system is more advantageous.
    :param data: data (single ndarray-like)
    :param data_type: 1 or 0, 1 represent positive indicator, use the positive calculation method;
                              0 represent negative indicator, use the negative calculation method.
    :param eigenvalue: global eigenvalue of all data/matrix (max, min)
    :return:
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    max_data = np.nanmax(data) if eigenvalue is None else eigenvalue[0]
    min_data = np.nanmin(data) if eigenvalue is None else eigenvalue[1]

    if data_type:
        data_sta = (data - min_data) / (max_data - min_data)
    else:
        data_sta = (max_data - data) / (max_data - min_data)
    return data_sta
