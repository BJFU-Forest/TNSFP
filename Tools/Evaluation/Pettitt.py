import numpy as np


def pettitt_change_point_detection(inputdata, _tfpw=True):
    if not isinstance(inputdata, np.ndarray):
        inputdata = np.array(inputdata)

    move = 0
    if _tfpw:
        inputdata = tfpw(inputdata)
        move = 1

    n = inputdata.shape[0]
    Uk = np.array([])
    Ut = 0
    for t in range(1, n):
        z = 0
        for j in range(n):
            z += _sgn(inputdata[t] - inputdata[j])
        Ut += z
        Uk = np.append(Uk, Ut)

    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    k = Uka.index(U)
    pvalue = 2 * np.exp((-6 * (U ** 2)) / (n ** 3 + n ** 2))
    tr_type = 2 if pvalue < 0.01 else 1 if 0.01 <= pvalue < 0.05 else 0
    return k + move, tr_type


def _sgn(delta):
    """Sign function"""
    if delta == 0:
        return 0
    return delta / abs(delta)
