# coding=utf-8
from multiprocessing import Pool, Process, Manager
from sklearn.linear_model import LinearRegression
from dlm.dlm_functions import *
import numpy as np
import os
import time

from Tools.Tool import create_path

def get_statistics(parameter, backup_path, coordinate, matrix, num_list, lock):
    n, z, x = np.shape(matrix)
    alpha = np.full([z, x], np.nan)
    variance = alpha.copy()

    for i in range(x):
        data = []

        for k in range(n):
            data_k = matrix[k, :, i]
            if np.isnan(data_k).all():
                continue
            if np.isinf(data_k).all():
                continue
            if (data_k == 0).all():
                continue
            data_k[data_k != data_k] = np.nanmean(data_k)
            data_k[np.isinf(data_k)] =  np.nanmean(data_k)
            data.append(data_k)

        data = np.asarray(data).T
        try:
            X = data[:, 0].reshape(-1, 1)
            Y = data[:, 1]

            # use two seasonal harmonic components
            rseas = [1, 2]

            # set up model and run forward filtering
            delta = 0.98
            M = Model(Y, X, rseas, delta)
            FF = forwardFilteringM(M)

            # extract estimates on the coefficient corresponding to lag-1 NDVI
            vid = 2  # index of autocorrelation
            alpha[:, i] = FF.get('sm')[vid, 1:]  # mean of autocorrelation
            variance[:, i] = FF.get('sC')[vid, vid, 1:]  # variance of autocorrelation
        except:
            alpha[:, i], variance[:, i] = np.full([z], np.nan), np.full([z], np.nan)
    record = {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate,
              "alpha": alpha, "variance": variance
              }
    return record


def record_as_npz(record):
    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    alpha = record["alpha"]
    variance = record["variance"]
    address = backup_path + parameter + "\\"
    if not os.path.isdir(address):
        os.makedirs(address)
    np.savez("%spart_%d.npz" % (address, coordinate), coordinate=coordinate, alpha=alpha, variance=variance)


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


def result_interpretation(z, x, y, parameter, backup_path):
    alpha = np.zeros([z, y, x])
    variance = alpha.copy()
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
                alpha[:, i, :] = result_part["alpha"]
                variance[:, i, :] = result_part["variance"]
            except:
                raise Exception("load npz file error")
        else:
            raise Exception("npz file does not exist")
    np.savez(out_file, alpha=alpha, variance=variance)
    return alpha, variance


def handler(parameter_name, matrix_x, matrix_y, process_num=None, backup_path=None):
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

    backup_path = r"..\Backup\ForXH" + "\\" if backup_path is None else backup_path
    create_path(backup_path)
    pool = Pool(processes=process_num) if process_num is not None else Pool()

    for i in range(y):
        pool.apply_async(func=get_statistics,
                         args=(parameter_name, backup_path, i, matrix[:, :, i, :], num_list, lock),
                         callback=record_as_npz)

    pool.close()
    pool.join()

    alpha, variance = result_interpretation(z, x, y, parameter_name, backup_path)

    if time_reporter.is_alive:
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return alpha, variance
