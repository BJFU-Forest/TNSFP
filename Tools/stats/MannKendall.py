# coding=utf-8
import scipy.stats as st
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from Tools.Evaluation.SenSlope import sen_slope_estimator as sen


def trend_test(vals, axis):
    """
    Mann-Kendall Test
    :param vals: set of data to analyze. Must be in chronological order
    :return:
    """
    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)

    try:
        s = calc_s(vals, axis=axis)
        variance = var_s(vals, axis=axis)

        z = calc_z(s, variance)
        return z
    except:
        return np.nan


def calc_s(arr, axis):
    """Calculate S Statistic"""
    in_dims = np.arange(arr.ndim)
    arr = np.transpose(arr, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    dif=arr[..., :-1]-arr[..., 1:]
    return np.sum(dif>0 - dif <0, axis=-1)

def var_s(arr, axis):
    """Calculate Variance of S statistic"""
    def v(x):
        return x * (x - 1) * (2 * x + 5)

    ties = np.unique(arr, return_counts=True, axis=axis)
    ties[ties==1] = 0

    total = v(arr.shape[axis])
    tie_v = np.nansum(v(ties), axis=axis)

    return (total - tie_v) / 18


def calc_z(s, variance):
    """Calculate Z Score from S and VAR(S)"""
    s[s>0] = (s[s>0] - 1) / np.sqrt(variance[s>0])
    s[s < 0] = (s[s < 0] + 1) / np.sqrt(variance[s < 0])
    return s
