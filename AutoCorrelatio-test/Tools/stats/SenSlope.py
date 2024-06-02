# coding=utf-8
import scipy.stats as st
import numpy as np
from math import sqrt
# vals=[1,2,3,4,5,7]

def sen_slope_estimator(vals, alpha=0.05):
    """
    Sen Slope Test
    :param vals: set of data to analyze. Must be in chronological order
    :return:
    """
    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)
    if np.isnan(vals).any() or np.isinf(vals).any() or np.nanstd(vals) == 0:
        return np.nan
    try:
        slope = calc_qm(vals)
        # variance = var_s(vals)
        # c = calc_c(alpha, variance)
        return slope
    except:
        return np.nan


def calc_qm(vals):
    n = len(vals)
    qlist = []
    for r in range(n):
        for c in range(n):
            if r > c:
                qlist.append((vals[r] - vals[c]) / (r - c))
    qm = np.median(qlist)
    return qm


def calc_c(alpha, variance):
    c = st.norm.ppf(1 - (alpha / 2)) * sqrt(variance)
    return c


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

