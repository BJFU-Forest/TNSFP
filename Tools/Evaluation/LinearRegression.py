import numpy as np
import scipy.stats as st


def liner_slope(data_y, data_x=None):
    """
    返回y=bx+a斜率beta和显著性p
    :param data_x: 自变量, ndarray
    :param data_y: 因变量, ndarray
    :return:
        beta: 变化速率
        si_type: 显著性
            Classification method:
            0 represents No significant
            1 represents Significantly
            2 represents Extremely significant
    """
    if not isinstance(data_y, np.ndarray):
        data_y = np.asarray(data_y)

    n = len(data_y)
    if data_x is None:
        x = np.array(range(len(data_y)))
    elif len(data_x) != len(data_y):
        raise Exception("data_x and data_y must have the same length")
    else:
        if not isinstance(data_x, np.ndarray):
            data_x = np.asarray(data_x)
        x = data_x.copy()

    y = data_y.copy()
    average_x = np.mean(x)
    average_y = np.mean(y)
    delta_x = x - average_x
    delta_y = y - average_y
    y_delta_x = np.array([y[i] * delta_x[i] for i in range(n)])
    delta_xy = np.array([delta_y[i] * delta_x[i] for i in range(n)])
    lxx = np.sum(delta_x ** 2)
    lyy = np.sum(delta_y ** 2)
    lxy = np.sum(delta_xy)
    ylx = np.sum(y_delta_x)
    beta = lxy / lxx if lxx != 0 else 0
    sigma2 = (lyy - beta * ylx) / (n - 2)
    t = beta / np.sqrt(sigma2 / lxx) if lxx != 0 else 0
    t = abs(t)
    p = 2 * st.t.sf(t, n-2)
    si_type = 2 if p < 0.01 else 1 if 0.01 <= p < 0.05 else 0
    si_type = si_type * beta / abs(beta) if beta != 0 else 0
    return beta, si_type
