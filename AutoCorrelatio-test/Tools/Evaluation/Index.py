# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
from scipy.stats import pearsonr

# 计算方差使用np.var(arr)
# 计算相关性系数使用pd-> np.corrcoef(x,y)


# 计算偏相关系数Partial correlation coefficient
def par_cc(x, y, covar):
    """
    计算偏相关系数Partial correlation coefficient
    :param x: 变量x
    :param y: 变量y
    :param covar: 控制变量
    :return:
    """
    r_xy = pearsonr(x, y)[0]
    r_xc = pearsonr(x, covar)[0]
    r_yc = pearsonr(y, covar)[0]
    r_xy_c = (r_xy - r_xc * r_yc) / (((1 - r_xc ** 2) ** 0.5) * ((1 - r_yc ** 2) ** 0.5))
    return r_xy_c


# 计算变异系数 Coefficient of Variation
def CoV(data):
    std = np.std(data)
    mean = np.nanmean(data)
    cov = std / mean if mean != 0 else 0
    return cov


# 计算纳什效率系数 Nash-Sutcliffe Efficiency coefficient
def NSE(Qobseved, Qsimulate):
    dimension = len(Qobseved)
    Qaverage = np.mean(Qobseved[:dimension + 1])
    distance = 0.0
    for i in range(dimension):
        distance += (Qobseved[i] - Qsimulate[i]) ** 2
    deviation = 0.0
    for i in range(dimension):
        deviation += (Qobseved[i] - Qaverage) ** 2
    nse = 1 - (distance / deviation)
    return nse


# 计算百分比偏差 Percentage Bias
def PBIAS(Qobseved, Qsimulate):
    dimension = len(Qobseved)
    distance = 0.0
    for i in range(dimension):
        distance += (Qsimulate[i] - Qobseved[i])
    Qsum = np.sum(Qobseved)
    pbias = distance / Qsum * 100
    return pbias


def get_average(records):
    """
    平均值
    """
    return sum(records) / len(records)


def get_variance(records):
    """
    方差 反映一个数据集的离散程度
    """
    average = get_average(records)
    return sum([(x - average) ** 2 for x in records]) / len(records)


def get_standard_deviation(records):
    """
    标准差 == 均方差 反映一个数据集的离散程度
    """
    variance = get_variance(records)
    return math.sqrt(variance)


def get_STDn(records_real, records_predict):
    return get_standard_deviation(records_predict) / get_standard_deviation(records_real)


def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rsr(records_real, records_predict):
    """
    标准偏差比
    """
    try:
        return get_rmse(records_real, records_predict) / get_standard_deviation(records_real)
    except:
        raise Exception(1)


# 写入csv
def WriteResult(inputpath, outputpath, collist, outputname):
    result = pd.DataFrame(index=["CoV", "CC", "NSE", "PBIAS"], columns=collist)
    data = pd.read_csv(inputpath)
    observedname = result.columns[0]
    Qobserved = data[observedname].values
    result[observedname]["CoV"] = CoV(Qobserved)
    for set in result.columns[1:]:
        Qsimulate = data[set].values
        result[set]["CoV"] = CoV(Qsimulate)
        result[set]["CC"] = np.corrcoef(Qobserved, Qsimulate)[1, 0]
        result[set]["NSE"] = NSE(Qobserved, Qsimulate)
        result[set]["PBIAS"] = PBIAS(Qobserved, Qsimulate)
    result.to_csv(outputpath + outputname + ".csv")
    print(result)

