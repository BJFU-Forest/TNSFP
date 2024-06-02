# coding=utf-8

import numpy as np
import pandas as pd
import time
# 机器学习库
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.datasets import make_hastie_10_2
import xgboost as xgb
import shap


def normalization(data):
    """归一化"""
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))


def xgbModel_trainer(X, y, num_rounds, params=None):
    """
    训练xgboost模型
    :param X: 影响因素 factor (ndarray-like)
    :param y: 样本 sample (ndarray-like)
    :param num_rounds: 训练次数 (int)
    :param params: 模型参数 (dict)
    :return: xgbModel
    """
    # xgb矩阵赋值
    xgb_train = xgb.DMatrix(X, label=y)

    # 参数
    params = {
        # Globel Parameters
        'booster': 'gbtree',
        'verbosity': 0,  # 运行信息输出, 0为静默, 1为警告
        'nthread': 30,  # cpu 线程数 默认最大

        # Parameters for Tree Booster
        'eta': 0.01,  # 如同学习率
    } if params is None else params

    model = xgb.train(params, xgb_train, num_rounds)
    # model.save_model('./model/xgb.model') # 用于存储训练出的模型
    return model


def shap_aa(X, y, num_rounds, params=None, normal=True):
    """
    基于可解释机器学习库SHAP的归因分析
    :param X: 影响因子 factor (ndarray-like)
    :param y: 样本 sample (ndarray-like)
    :param num_rounds: 训练次数 (int)
    :param params: 模型参数 (dict)
    :param normal: 是否归一化 (boolean)
    :return:
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # 归一化数据
    if normal:
        X = normalization(X)
        y = normalization(y)
    model = xgbModel_trainer(X, y, num_rounds, params)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
    shap_result = np.abs(shap_values).mean(axis=0)
    return shap_result


def main_factor(X, y, num_rounds, params=None):
    """
    返回SHAP的归因分析结果
    :param X: 影响因子 factor (ndarray-like)
    :param y: 样本 sample (ndarray-like)
    :param num_rounds: 训练次数 (int)
    :param params: 模型参数 (dict)
    :return: max shap value & label
    """
    shap_values = shap_aa(X, y, num_rounds, params)
    shap_values = shap_values / np.nansum(shap_values)
    return np.max(shap_values), np.argmax(shap_values)


def factors_contibution(X, y, num_rounds, params=None):
    """
    返回SHAP的归因分析结果
    :param X: 影响因子 factor (ndarray-like)
    :param y: 样本 sample (ndarray-like)
    :param num_rounds: 训练次数 (int)
    :param params: 模型参数 (dict)
    :return:
    """
    shap_values = shap_aa(X, y, num_rounds, params)
    shap_values = shap_values / np.nansum(shap_values)
    return shap_values


if __name__ == "__main__":
    start_time = time.time()
    X, y = shap.datasets.boston()
    mf, l = main_factor(X, y, num_rounds=100)
    print(mf, l)
