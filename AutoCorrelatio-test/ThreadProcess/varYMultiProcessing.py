# coding=utf-8
from multiprocessing import Pool, Process, Manager
import numpy as np
import os
import time

from Tools.Tool import create_path

import warnings

# 关闭所有警告
warnings.filterwarnings("ignore")

def get_statistics(parameter, backup_path, coordinate, matrix, num_list, lock):
    n, z, x = np.shape(matrix)
    var_y = np.full([z, x], np.nan)

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
            sigma=np.var(data[:, 1])
            var_y[:, i] = sigma / (1-np.square(data[:, 0]))
        except:
            var_y[:, i] = np.full([z], np.nan)
            # with lock:
            #     num_list[1] += 1
    record = {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate,
              "var_y": var_y
              }
    return record


def record_as_npz(record):
    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    var_y = record["var_y"]
    address = backup_path + parameter + "\\"
    if not os.path.isdir(address):
        os.makedirs(address)
    np.savez("%spart_%d.npz" % (address, coordinate), coordinate=coordinate, var_y=var_y)


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


def result_interpretation(z, y, x,  parameter, backup_path):
    var_y = np.zeros([z, y, x])
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
                var_y[:, i, :] = result_part["var_y"]
            except:
                raise Exception("load npz file error")
        else:
            raise Exception("npz file does not exist")
    np.savez(out_file, var_y=var_y)
    return var_y


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

    var_y = result_interpretation(z, y, x, parameter_name, backup_path)

    if time_reporter.is_alive:
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return var_y
