import numpy as np


def recognition_accuracy(x, y):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    hd = np.sum((x <= -2) & (y <= -2))  # 同时捕捉到干旱
    hw = np.sum((x >= 2) & (y >= 2))  # 同时捕捉到洪涝

    md = np.sum((x <= -2) & (y > -2))  # x捕捉到干旱，y未捕捉到
    mw = np.sum((x >= 2) & (y < 2))  # x捕捉到洪涝，y未捕捉到

    fd = np.sum((x > -2) & (y <= -2))  # y捕捉到干旱，x未捕捉到
    fw = np.sum((x < 2) & (y >= 2))  # y捕捉到洪涝，x未捕捉到

    pod_d = hd / (hd + md)
    far_d = fd / (hd + fd)

    pod_w = hw / (hw + mw)
    far_w = fw / (hw + fw)
    return pod_d, far_d, pod_w, far_w
