# coding=utf-8
import numpy as np
from math import sqrt


def trend_test(vals):
    """
    Mann-Kendall Test
    :param vals: set of data to analyze. Must be in chronological order
    :return:
    """
    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)

    if np.isnan(vals).all() or np.isinf(vals).all() or np.nanstd(vals) == 0:
        return np.nan
    vals[np.isnan(vals)] = np.nanmean(vals)
    vals[np.isnan(vals)] = np.nanmean(vals)
    try:
        s = calc_s(vals)
        variance = var_s(vals)

        z = calc_z(s, variance)
        return z
    except:
        return np.nan


def calc_s(vals):
    """Calculate S Statistic"""
    n = len(vals)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            sign = _sgn(vals[j] - vals[k])
            s += sign
    return s


def var_s(vals):
    """Calculate Variance of S statistic"""

    def v(x):
        return x * (x - 1) * (2 * x + 5)

    ties = np.unique(vals, return_counts=True)[1].tolist()
    while 1 in ties:
        ties.remove(1)

    total = v(len(vals))
    for tie in ties:
        total -= v(tie)

    return total / 18


def calc_z(s, variance):
    """Calculate Z Score from S and VAR(S)"""
    if not s:
        return 0

    return (s - _sgn(s)) / sqrt(variance)


def _sgn(delta):
    """Sign function"""
    if delta == 0:
        return 0
    return delta / abs(delta)