# coding=utf-8
from multiprocessing import Pool, Process, Manager
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np
import os
import time

from Tools.Tool import create_path


def get_statistics(parameter, backup_path, coordinate, matrix, num_list, lock):
    n, x = np.shape(matrix)
    classify = np.full([x], np.nan)

    for i in range(x):
        # with lock:
        #     num_list[0] += 1
        hisAc = matrix[0, i]
        hisNPP = matrix[1, i]
        hisLUC = matrix[2, i]
        hisP = matrix[3, i]
        hisET = matrix[4, i]
        futureP = matrix[5, i]
        adP = matrix[6, i]
        allP = futureP - adP
        if np.isnan(hisAc) or np.isnan(hisP) or np.isnan(hisET) or np.isnan(hisLUC) or np.isnan(futureP):
            classify[i] = np.nan
            # with lock:
            #     num_list[2] += 1
        elif hisLUC not in [1, 2, 3, 4, 5, 6, 10]:
            classify[i] = np.nan
            # with lock:
            #     num_list[2] += 1
        elif allP < 175:
            classify[i] = 1
        else:
            try:
                if hisLUC in [1, 2, 3, 4, 5]:
                    p_et_h = hisP - hisET
                    etF = 0.0002 * futureP + 0.78
                    p_et_f = futureP - hisNPP * etF
                    # unsuti = 1 if allP < 400 else 0
                    # unsuti = 1 if (p_et_f - adP) < 0 else 0
                    ac0 = 1 if hisAc > 0.05 else 0
                    # ac0 = 1 if hisAc > 0.429 else 0
                    # unsuti = 1 if ((p_et_f + adP < 0) and (p_et_f < p_et_h) and ac0) else 0
                    unsuti =  1 if (((p_et_f + adP < 0) and (allP < 400)) and (p_et_f < p_et_h) and ac0) else 0
                    risk = (1 if (p_et_f > p_et_h) else 0) and ac0
                elif hisLUC == 6:
                    p_et_h = hisP - hisET
                    etF = -1.23 + 2.59 * (1 - np.exp(-0.008 * allP))
                    p_et_f = futureP - hisNPP * etF
                    # unsuti = 1 if allP < 315 else 0
                    # unsuti = 1 if (p_et_f - adP) < 0 else 0
                    ac0 = 1 if hisAc > 0.16 else 0
                    # ac0 = 1 if hisAc > 0.589 else 0
                    # unsuti = 1 if ((p_et_f + adP < 0) and (p_et_f < p_et_h) and ac0) else 0
                    unsuti =  1 if (((p_et_f + adP < 0) and (allP < 315)) and (p_et_f < p_et_h) and ac0) else 0
                    risk = (1 if (p_et_f < p_et_h) else 0) and ac0
                else:   # hisLUC==10
                    p_et_h = hisP - hisET
                    etF = 0.93 + 1.38 * np.exp(-0.005 * allP)
                    p_et_f = futureP - hisNPP * etF
                    # unsuti = 1 if (p_et_f - adP) < -100 else 0
                    ac0 = 1 if hisAc > 0.6 else 0
                    # ac0 = 1 if hisAc > 0.437 else 0
                    # unsuti =  1 if ((p_et_f + adP < 0) and (p_et_f < p_et_h) and ac0) else 0
                    unsuti =  1 if (((p_et_f + adP < 0) and (allP < 175)) and (p_et_f < p_et_h) and ac0) else 0
                    risk = 1 if ((p_et_f < p_et_h) and ac0) else 0
                # classify[i] = 1 if risk else 2
                classify[i] = 1 if unsuti else 2 if risk else 3
            except:

                classify[i] = np.nan
                # with lock:
                #     num_list[1] += 1
    record = {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate,
              "classify": classify
              }
    return record


def record_as_npz(record):
    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    classify = record["classify"]
    address = backup_path + parameter + "\\"
    if not os.path.isdir(address):
        os.makedirs(address)
    np.savez("%spart_%d.npz" % (address, coordinate), coordinate=coordinate, classify=classify)


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
    classify = np.zeros([y, x])
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
                classify[i, :] = result_part["classify"]
            except:
                raise Exception("load npz file error")
        else:
            raise Exception("npz file does not exist")
    np.savez(out_file, classify=classify)
    return classify


def handler(parameter_name, matrix_ac, matrix_npp, matrix_luc, matrix_his_p, matrix_his_et, matrix_fp,
            matrix_adp, process_num=None, backup_path=None):
    start_time = time.time()

    if (np.shape(matrix_ac) == np.shape(matrix_his_p)) \
            and (np.shape(matrix_npp) == np.shape(matrix_npp))\
            and (np.shape(matrix_ac) == np.shape(matrix_his_et)) \
            and (np.shape(matrix_ac) == np.shape(matrix_luc))\
            and (np.shape(matrix_ac) == np.shape(matrix_fp)) \
            and (np.shape(matrix_ac) == np.shape(matrix_adp)):
        matrix = np.asarray([matrix_ac, matrix_npp, matrix_luc,matrix_his_p, matrix_his_et,  matrix_fp, matrix_adp])
        n, y, x = np.shape(matrix)
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
                         args=(parameter_name, backup_path, i, matrix[:, i, :], num_list, lock),
                         callback=record_as_npz)

    pool.close()
    pool.join()

    classify = result_interpretation(x, y, parameter_name, backup_path)

    if time_reporter.is_alive:
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return classify
