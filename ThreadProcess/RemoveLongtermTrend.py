# coding=utf-8
from multiprocessing import Pool, Process
import multiprocessing
import numpy as np
import os
import time

import pandas as pd
from statsmodels.tsa.seasonal import STL
from Tools.Tool import create_path, del_file


def robust_stl(series, period, smooth_length=7):
    def nt_calc(f,ns):
        '''Calcualte the length of the trend smoother based on
        Cleveland et al., 1990'''
        nt = (1.5*f)/(1-1.5*(1/ns)) + 1 #Force fractions to be rounded up
        if int(nt) % 2. == 1:
            return int(nt)
        elif int(nt) % 2. == 0:
            return int(nt) + 1
    def nl_calc(f):
        '''Calcualte the length of the low-pass filter based on
        Cleveland et al., 1990'''
        if int(f) % 2. == 1:
            return int(f)
        elif int(f) % 2. == 0:
            return int(f) + 1
    ### REFERENCE FOR LOESS PARAMS BASED ON ORIGINAL FORTRAN CODE ###
    # np = f              # period of seasonal component
    # ns = 7              # length of seasonal smoother
    # nt = nt_calc(f,ns)  # length of trend smoother
    # nl = nl_calc(f)     # length of low-pass filter
    # isdeg = 1           # Degree of locally-fitted polynomial in seasonal smoothing.
    # itdeg = 1           # Degree of locally-fitted polynomial in trend smoothing.
    # ildeg = 1           # Degree of locally-fitted polynomial in low-pass smoothing.
    # nsjump = None       # Skipping value for seasonal smoothing.
    # ntjump = 1          # Skipping value for trend smoothing. If None, ntjump= 0.1*nt
    # nljump = 1          # Skipping value for low-pass smoothing. If None, nljump= 0.1*nl
    # robust = True       # Flag indicating whether robust fitting should be performed.
    # ni = 1              # Number of loops for updating the seasonal and trend  components.
    # no = 3              # Number of iterations of robust fitting. The value of no should
    #                       be a nonnegative integer. If the data are well behaved without
    #                       outliers, then robustness iterations are not needed. In this case
    #                       set no=0, and set ni=2-5 depending on how much security you want
    #                       that the seasonal-trend looping converges. If outliers are present
    #                       then no=3 is a very secure value unless the outliers are radical,
    #                       in which case no=5 or even 10 might be better. If no>0 then set ni
    #                       to 1 or 2. If None, then no is set to 15 for robust fitting,
    #                       to 0 otherwise.
    res = STL(series, period, seasonal=smooth_length, trend=nt_calc(period,smooth_length),
              low_pass=25, seasonal_deg=1, trend_deg=1,
              low_pass_deg=1, seasonal_jump=1,trend_jump=1, low_pass_jump=1, robust=True)
    result = res.fit()
    return result.resid


def get_statistics(parameter, backup_path, coordinate, matrix, **kwargs):
    z, x = np.shape(matrix)
    result_data = np.full([z, x], np.nan, dtype=np.float32)
    is_all_nan = True

    for i in range(x):
        data = matrix[:, i]
        if np.isnan(data).all() or np.isinf(data).all() or np.nanstd(data) == 0:
            continue
        if (np.isnan(data).sum() / data.shape[0]) > 0.05:
            continue
        else:
            df = pd.DataFrame({'Value': data})
            df['value_fit'] = df['Value'].interpolate(method='linear')
            data = df['value_fit'].values
            # print(1)
        try:
            result_data[:, i] = robust_stl(data, **kwargs)
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
