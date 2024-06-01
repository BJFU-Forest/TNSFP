# coding=utf-8
from multiprocessing import Pool, Process
import multiprocessing
import numpy as np
import os
import time

import pandas as pd

from Tools.Tool import create_path, del_file


def remove_long_term_trend(group):
    arr = group["Data"].values
    # 去平均
    arr = arr - np.nanmean(arr)

    # 去线性趋势
    mask = ~np.isnan(arr)
    # 初始化一个与group同长度且全部为NaN的Series
    result = pd.Series(np.nan, index=group.index)
    # 当全部为NaN或标准差为0时，返回全NaN的Series
    if np.isnan(arr).all() or np.nanstd(arr) == 0:
        return result
    # 取出非NaN的部分进行计算
    valid_arr = arr[mask]
    x = np.arange(len(arr))[mask]
    # 使用线性拟合
    slope, intercept = np.polyfit(x, valid_arr, 1)
    fitted_line = slope * x + intercept
    # 计算差值
    delta = valid_arr - fitted_line
    result[mask] = delta

    return result


def delta_var_calculate(data, dayofyear):
    df = pd.DataFrame({"DOY": dayofyear, "Data": data})
    # 应用 remove_long_term_trend 函数，并保留原始索引
    delta_ndvi = df.groupby("DOY").apply(lambda x: remove_long_term_trend(x)).reset_index(level=0, drop=True)
    # 重置索引后，delta_ndvi 会丢失原始的索引顺序
    # 如果您需要按照原始 DataFrame 的索引顺序进行排序，可以使用 sort_index()
    delta_ndvi = delta_ndvi.sort_index()
    return delta_ndvi.values


def get_statistics(parameter, backup_path, coordinate, matrix, **kwargs):
    z, x = np.shape(matrix)
    result_data = np.full([z, x], np.nan, dtype=np.float32)
    is_all_nan = True

    for i in range(x):
        data = matrix[:, i]
        if np.isnan(data).all() or np.isinf(data).all() or np.nanstd(data) == 0:
            continue
        try:
            result_data[:, i] = delta_var_calculate(data=data, **kwargs)
            is_all_nan = False
        except Exception as e:
            print(f"Error in calculation for column {i}: {e}")

    if is_all_nan:
        return None
    else:
        return {"parameter": parameter, "backup_path": backup_path, "coordinate": coordinate, "result_data": result_data}

def record_as_npz(record):
    if record is None:
        return

    parameter = record["parameter"]
    backup_path = record["backup_path"]
    coordinate = record["coordinate"]
    result_data = record["result_data"]
    address = os.path.join(backup_path, parameter, f"part_{coordinate}.npz")
    create_path(address)
    np.savez(os.path.join(address), coordinate=coordinate, result_data=result_data)

def time_record(start_time):
    while True:
        now_time = time.time()
        elapsed_time = now_time - start_time
        print("\rRun time: %s" % time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), end="")
        time.sleep(1)

def error_callback(error):
    print(f"Pool Error >>> {error}")

def load_npz_file_process(args):
    address, y_index, z, x = args
    address_part = os.path.join(address, f"part_{y_index}.npz")

    if not os.path.isfile(address_part):
        return y_index, None
    try:
        with np.load(address_part) as result_part:
            return y_index, result_part["result_data"]
    except Exception as e:
        print(f"Error loading {address_part}: {e}")
        return y_index, None

def result_interpretation(x, y, z, parameter, backup_path, process_num=4):
    address = os.path.join(backup_path, parameter)
    combined_data = np.full((z, y, x), np.nan, dtype=np.float32)

    with Pool(processes=process_num) as pool:
        args = [(address, i, z, x) for i in range(y)]
        results = pool.map(load_npz_file_process, args)

    for y_index, slice_data in results:
        if slice_data is not None:
            combined_data[:, y_index, :] = slice_data

    return combined_data

def handler(parameter_name, matrix, process_num=None, backup_path=None, **kwargs):
    start_time = time.time()

    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    z, y, x = np.shape(matrix)

    time_reporter = Process(target=time_record, args=(start_time, ))
    time_reporter.start()

    backup_path = os.path.join("..", "Backup", "ProcessFile") if backup_path is None else backup_path
    del_file(backup_path + parameter_name + "/")

    pool = Pool(processes=process_num if process_num is not None else multiprocessing.cpu_count())

    tasks = [pool.apply_async(func=get_statistics, args=(parameter_name, backup_path, i, matrix[:, i, :]),
                              kwds=kwargs, callback=record_as_npz, error_callback=error_callback) for i in range(y)]
    pool.close()
    for task in tasks:
        task.wait()
    pool.join()

    print("\nIntegrated data...")
    combined_data = result_interpretation(x, y, z, parameter_name, backup_path, process_num)

    if time_reporter.is_alive():
        time_reporter.terminate()
        print("\nFinish!")
        time_reporter.join()

    return combined_data
