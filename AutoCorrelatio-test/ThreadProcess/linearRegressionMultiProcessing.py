# coding=utf-8
from multiprocessing import Pool, Process, Manager
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import time

from Tools.Tool import create_path

def get_statistics(parameter, backup_path, coordinate, matrix, num_list, lock):
    n, z, x = np.shape(matrix)
    alpha = np.full([x], np.nan)
    residual = alpha.copy()

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
            if np.nanstd(data_k) == 0:
                continue
            data.append(data_k)

        # if len(data) != n:
        #     with lock:
        #         num_list[2] += 1
        #     continue
        if data:
            data = np.asarray(data).T
            data = data[~np.any(np.isnan(data) | np.isinf(data), axis=1)]
            try:
                # alpha[i], residual[i] = pearsonr(data[:, 0], data[:, 1])
                lr = LinearRegression()
                lr.fit(data[:, 0].reshape(-1, 1), data[:, 1])
                alpha[i] = lr.coef_[0]
                residual[i] = lr.intercept_
            except:
                alpha[i], residual[i] = np.nan, np.nan
                # with lock:
                #     num_list[1] += 1
        else:
            alpha[i], residual[i]= np.nan, np.nan
    record = {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate,
              "alpha": alpha, "residual": residual
              }
    return record


def record_as_npz(record):
    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    alpha = record["alpha"]
    residual = record["residual"]
    address = backup_path + parameter + "\\"
    if not os.path.isdir(address):
        os.makedirs(address)
    np.savez("%spart_%d.npz" % (address, coordinate), coordinate=coordinate, alpha=alpha, residual=residual)


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
    alpha = np.zeros([y, x])
    residual = alpha.copy()
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
                alpha[i, :] = result_part["alpha"]
                residual[i, :] = result_part["residual"]
            except:
                raise Exception("load npz file error")
        else:
            raise Exception("npz file does not exist")
    np.savez(out_file, alpha=alpha, residual=residual)
    return alpha, residual


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

    alpha, residual = result_interpretation(x, y, parameter_name, backup_path)

    if time_reporter.is_alive:
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return alpha, residual
