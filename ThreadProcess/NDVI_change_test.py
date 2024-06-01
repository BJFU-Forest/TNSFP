# coding=utf-8
import numpy as np
import pandas as pd
import os
import time
import scipy.stats as st


from multiprocessing import Pool, Process, Manager
from math import sqrt
from Tools.Evaluation import MannKendall, SenSlope, Index, Pettitt

def change_point_detection(vals):
    """
    Mann-Kendall突变点检测
    :param vals: required, set of data to analyze. Must be in chronological order
    :return:
    """
    inputdata = np.array(vals)
    n = inputdata.shape[0]
    # 正序列计算---------------------------------
    # 定义累计量序列Sk，初始值=0
    Sk = [0]
    # 定义统计量UFk，初始值 =0
    UFk = [0]
    # 定义Sk序列元素s，初始值 =0
    s = 0
    Exp_value = [0]
    Var_value = [0]
    # i从1开始，因为根据统计量UFk公式，i=0时，Sk(0)、E(0)、Var(0)均为0
    # 此时UFk无意义，因此公式中，令UFk(0)=0
    for i in range(1, n):
        for j in range(i):
            if inputdata[i] > inputdata[j]:
                s = s + 1
            else:
                s = s + 0
        Sk.append(s)
        Exp_value.append((i + 1) * (i + 2) / 4)  # Sk[i]的均值
        Var_value.append((i + 1) * i * (2 * (i + 1) + 5) / 72)  # Sk[i]的方差
        UFk.append((Sk[i] - Exp_value[i]) / np.sqrt(Var_value[i]))
    # ------------------------------正序列计算

    # 逆序列计算---------------------------------
    # 定义逆序累计量序列Sk2，长度与inputdata一致，初始值=0
    Sk2 = [0]
    # 定义逆序统计量UBk，长度与inputdata一致，初始值=0
    UBk = [0]
    UBk2 = [0]
    # s归0
    s2 = 0
    Exp_value2 = [0]
    Var_value2 = [0]
    # 按时间序列逆转样本y
    inputdataT = list(reversed(inputdata))
    # i从1开始，因为根据统计量UBk公式，i=0时，Sk2(0)、E(0)、Var(0)均为0
    # 此时UBk无意义，因此公式中，令UBk(0)=0
    for i in range(1, n):
        for j in range(i):
            if inputdataT[i] > inputdataT[j]:
                s2 = s2 + 1
            else:
                s2 = s2 + 0
        Sk2.append(s2)
        Exp_value2.append((i + 1) * (i + 2) / 4)  # Sk[i]的均值
        Var_value2.append((i + 1) * i * (2 * (i + 1) + 5) / 72)  # Sk[i]的方差
        UBk.append((Sk2[i] - Exp_value2[i]) / np.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])
    # 由于对逆序序列的累计量Sk2的构建中，依然用的是累加法，即后者大于前者时s加1，
    # 则s的大小表征了一种上升的趋势的大小，而序列逆序以后，应当表现出与原序列相反
    # 的趋势表现，因此，用累加法统计Sk2序列，统计量公式(S(i)-E(i))/sqrt(Var(i))
    # 也不应改变，但统计量UBk应取相反数以表征正确的逆序序列的趋势
    #  UBk(i)=0-(Sk2(i)-E)/sqrt(Var)
    # ------------------------------逆序列计算

    # 此时上一步的到UBk表现的是逆序列在逆序时间上的趋势统计量
    # 与UFk做图寻找突变点时，2条曲线应具有同样的时间轴，因此
    # 再按时间序列逆转结果统计量UBk，得到时间正序的UBkT，
    UBkT = list(reversed(UBk2))
    diff = np.array(UFk) - np.array(UBkT)
    klist = list()
    # 找出交叉点
    for k in range(1, n):
        if diff[k - 1] * diff[k] < 0:
            klist.append(k)
    return UFk, UBkT, klist


def sen_slope_estimator(vals, alpha=0.05):
    """
    Sen Slope Test
    :param vals: set of data to analyze. Must be in chronological order
    :return:
    """
    if not isinstance(vals, np.ndarray):
        vals = np.asarray(vals)

    slope = calc_qm(vals)
    variance = var_s(vals)
    c = calc_c(alpha, variance)
    return slope, c


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


def find_slow_ndvi(ndvi, ac, trigger):
    UFk, UBkT, klist = change_point_detection(ndvi)
    klist = np.asarray(klist)
    klist = klist[klist > trigger]
    state = 0
    min_ac = np.nan
    for idx in klist:
        alpha_before = sen_slope_estimator(ndvi[:idx])[0]
        alpha_after = sen_slope_estimator(ndvi[idx:])[0]
        if alpha_after < alpha_before:
            state = 1
            min_ac = min(ac[idx], min_ac) if ~np.isnan(min_ac) else ac[idx]
    return state, min_ac


def get_statistics(parameter, backup_path, coordinate, matrix, num_list, lock, trigger):
    n, z, x = np.shape(matrix)
    state = np.full([x], np.nan)
    min_ac = state.copy()

    for i in range(x):
        data = []
        # with lock:
        #     num_list[0] += 1

        for k in range(n):
            data_k = matrix[k, :, i]
            if np.isnan(data_k).all():
                continue
            if np.isinf(data_k).all():
                continue
            if (data_k == 0).all():
                continue
            data_k[data_k != data_k] = np.nanmean(data_k)
            data_k[np.isinf(data_k)] = 0
            data.append(data_k)

        if len(data) != n:
            # with lock:
            #     num_list[2] += 1
            continue

        data = np.asarray(data).T
        try:
            state[i], min_ac[i] = find_slow_ndvi(data[:, 0], data[:, 1], trigger)
        except:
            # with lock:
            #     num_list[1] += 1
            pass
    record = {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate,
              "state": state, "min_ac": min_ac,
              }
    return record


def record_as_npz(record):
    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    state = record["state"]
    min_ac = record["min_ac"]
    address = backup_path + parameter + "\\"
    if not os.path.isdir(address):
        os.makedirs(address)
    np.savez("%spart_%d.npz" % (address, coordinate), coordinate=coordinate,
             state=state, min_ac=min_ac,
             )


def time_record(start_time, num_list, total):
    while 1:
        num = num_list[0]
        err = num_list[1]
        temp = num_list[2]
        now_time = time.time()
        up_time = now_time - start_time
        run_seconds = up_time % 60
        run_minutes = (up_time // 60) % 60
        run_hours = (up_time // (60 * 60)) % 60
        run_days = (up_time // (60 * 60 * 24)) % 24
        print("\rrate of progress = %.2f %% (%d/%d) err: %d NULL: %d  Run time: %d days %d h %d min %d s" % (
            num * 100 / total, num, total, err, temp, run_days, run_hours, run_minutes, run_seconds), end="")
        time.sleep(1)


def result_interpretation(x, y, parameter, backup_path):
    state = np.zeros([y, x])
    min_ac = state.copy()
    address = backup_path + parameter + "\\"
    out_path = backup_path + "interpretation\\"
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = out_path + parameter + ".npz"
    for i in range(y):
        address_part = "%spart_%d.npz" % (address, i)
        if os.path.isfile(address_part):
            try:
                result_part = np.load(address_part)
                state[i, :] = result_part["state"]
                min_ac[i, :] = result_part["min_ac"]
            except:
                raise Exception("load npz file error")
        else:
            raise Exception("npz file does not exist")
    np.savez(out_file,
             state=state, min_ac=min_ac,
             )
    return state, min_ac


def handler(parameter_name, matrix_x, matrix_y, trigger, process_num=None, backup_path=None):
    start_time = time.time()

    if np.shape(matrix_x) == np.shape(matrix_y):
        matrix = np.asarray([matrix_x, matrix_y])
        n, z, y, x = np.shape(matrix)
    else:
        raise Exception("数组大小不同")

    m = Manager()
    lock = m.Lock()
    num_list = m.list()
    num_list.append(0)
    num_list.append(0)
    num_list.append(0)
    total = x * y

    time_reporter = Process(target=time_record, args=(start_time, num_list, total))
    time_reporter.start()

    backup_path = r"..\Backup\GLDAS" + "\\" if backup_path is None else backup_path
    pool = Pool(processes=process_num) if process_num is not None else Pool()

    for i in range(y):
        pool.apply_async(func=get_statistics,
                         args=(parameter_name, backup_path, i, matrix[:, :, i, :], num_list, lock, trigger),
                         callback=record_as_npz)

    pool.close()
    pool.join()

    state, min_ac = result_interpretation(x, y, parameter_name, backup_path)

    if time_reporter.is_alive:
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return state, min_ac