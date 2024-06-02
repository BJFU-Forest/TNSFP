# coding=utf-8
import numpy as np
import os
import time

from multiprocessing import Pool, Process, Manager
from Tools.Evaluation import MannKendall, SenSlope


def get_statistics(parameter, backup_path, coordinate, matrix, num_list, lock):
    z, y = np.shape(matrix)
    avg = np.full([y], np.nan)
    mk = avg.copy()
    sen = avg.copy()

    for i in range(y):
        data = matrix[:, i]
        # with lock:
        #     num_list[0] += 1
        if np.isnan(data).any() or np.isinf(data).any() or (np.nanstd(data) == 0):
            # with lock:
            #     num_list[2] += 1
            continue
        # data[data > 3 * np.nanstd(data)] = np.nan
        # data = data[data < 3 * np.nanstd(data)]
        try:
            avg[i] = np.nanmean(data)
            mk[i] = MannKendall.trend_test(data)
            sen[i] = SenSlope.sen_slope_estimator(data)
        except:
            # with lock:
            #     num_list[1] += 1
            continue
    record = {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate,
              "avg": avg, "mk": mk, "sen": sen
              }
    return record


def record_as_npz(record):
    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    avg = record["avg"]
    mk = record["mk"]
    sen = record["sen"]
    address = backup_path + parameter + "\\"
    if not os.path.isdir(address):
        os.makedirs(address)
    np.savez("%spart_%d.npz" % (address, coordinate), coordinate=coordinate,
             avg=avg, mk=mk, sen=sen)


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
        # print("\rrate of progress = %.2f %% (%d/%d) err: %d NULL: %d  Run time: %d days %d h %d min %d s" % (
        #     num * 100 / total, num, total, err, temp, run_days, run_hours, run_minutes, run_seconds), end="")
        print("\r           Run time: %d days %d h %d min %d s" % (run_days, run_hours, run_minutes, run_seconds), end="")
        time.sleep(1)


def result_interpretation(x, y, parameter, backup_path):
    avg = np.zeros([y, x])
    mk = avg.copy()
    sen = avg.copy()
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
                avg[i, :] = result_part["avg"]
                mk[i, :] = result_part["mk"]
                sen[i, :] = result_part["sen"]
            except:
                raise Exception("load npz file error")
        else:
            raise Exception("npz file does not exist")
    np.savez(out_file,
             avg=avg,  mk=mk, sen=sen
             )
    return avg,mk, sen


def handler(parameter_name, matrix, process_num=None, backup_path=None):
    start_time = time.time()

    z, y, x = np.shape(matrix)
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
        pool.apply_async(func=get_statistics, args=(parameter_name, backup_path, i, matrix[:, i, :], num_list, lock),
                         callback=record_as_npz)

    pool.close()
    pool.join()

    avg, mk, sen = result_interpretation(x, y, parameter_name, backup_path)

    if time_reporter.is_alive:
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return avg, mk, sen
