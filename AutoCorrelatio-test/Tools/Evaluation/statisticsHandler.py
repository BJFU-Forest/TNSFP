import numpy as np
from Tools.Evaluation import MannKendall, SenSlope, Pettitt, LinearRegression


def get_ub_ua(data, k):
    """
    按突变点将数据整理成等长度的三列数据
    :param data: 数据序列 (ndarray-like)
    :param k: 突变点索引 (int)
    :return:
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    n = np.shape(data)[0]

    nan_list = np.full([n], np.nan)
    before_abrupt = nan_list.copy()
    before_abrupt[:k] = data[:k]
    after_abrupt = nan_list.copy()
    after_abrupt[k + 1:] = data[k + 1:]
    abrupt_point = nan_list.copy()
    abrupt_point[k] = data[k]
    return before_abrupt, after_abrupt, abrupt_point


def get_statistics_result(data, first_year):
    """
    计算MK趋势、Pettitt突变、sen's slope趋势
    :param data: 数据序列 (ndarray-like)
    :param first_year: 数据开始年份/序号 (int)
    :return:
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    # MK趋势检验
    mk, p, tr_type = MannKendall.trend_test(data)
    tr_type = "**" if (tr_type == 2) or (tr_type == -2) else "*" if (tr_type == 1) or (tr_type == -1) else ""
    mk = "%.2f" % mk + tr_type

    # Pettitt突变点检验
    k, k_type = Pettitt.pettitt_change_point_detection(data)
    k_type = "**" if k_type == 2 else "*" if k_type == 1 else ""
    abrupt = "%d" % (k + first_year) + k_type

    # 突变前后的线性回归斜率
    beta_b, b_type_b = LinearRegression.liner_slope(data[:k])
    b_type_b = "**" if (b_type_b == 2) or (b_type_b == -2) else "*" if (b_type_b == 1) or (
            b_type_b == -1) else ""
    beta_b = "%.2f" % beta_b + b_type_b

    beta_a, b_type_a = LinearRegression.liner_slope(data[k + 1:])
    b_type_a = "**" if (b_type_a == 2) or (b_type_a == -2) else "*" if (b_type_a == 1) or (
            b_type_a == -1) else ""
    beta_a = "%.2f" % beta_a + b_type_a

    abrupt = "%s (%s) %s" % (beta_b, abrupt, beta_a)

    # Sen slope 检验
    slope, c = SenSlope.sen_slope_estimator(data)

    return mk, slope, abrupt, k


def get_multi_year_average(data, years):
    """
    计算多年移动平均
    :param data: 数据序列 (ndarray-like)
    :param years: 窗口/移动年份 (int)
    :return:
    """
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    multi_year_average = np.convolve(np.ones(years) / years, data, mode="valid")
    return multi_year_average
