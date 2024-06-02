# coding=utf-8
"""
    main线
    1. 原始数据读取（【txt/nc to csv】被预测因子+预测因子）
    2. 空间降尺度（历史模型校正+未来预测）
    3. 多模式集合平均（暂定岭回归方法）
    4. 时间降尺度（【马尔可夫链+分布型拟合+偏差校正】历史模型校正+未来预测）
    5. 泰森平均（计算流域平均，用于结果评估和出图）
    6. 降尺度结果评估+出图
    7. 日尺度结果导出为SWAT格式

    参数
    1. 输入数据路径
        a. 原始日尺度观测数据：gauge_path
        b. 气象站站点信息：stationInfo
        c. 流域边界矢量：boundary
        d. 输入/输出时间尺度: (Monthly or Daily)
        e. 原始预测因子/GCM父文件夹：gcm_father_path
        f. GCM命名(一级文件夹)：gcm_basename
        g. 历史情景命名(二级文件夹)：historical
        h. 未来情景命名(二级文件夹)：scenarios
    2. 输出数据路径
        a. 观测数据输出：observe_path
        b. 预测因子数据输出：predictor_path
        c. 降尺度结果：downscaling_path
        d. 降尺度模型备份：model_path
        e. 泰森平均结果：voronoi_path
        f. 降尺度结果评估：evaluation_path
        g. SWAT格式输出：swat_path
        h. 过程文件缓存：process_file
    3. 数据处理参数
        a. GCM是否存在闰年记录：gcm_record_leap
        b. 空缺值：nan_value
        c. 空间降尺度方法：spatial_downscaling_method
        d. 多模式集合平均方法：mmea_method
        e. 被预测变量：tand_values
        f. 非负变量：condition_values
        g. 概率变量：occur_values
        h. 温度变量标签：temperature_label
        i. 降尺度历史校验时段：periods
        j. 后处理时间尺度：scales
        k. 降水发生下限：precipitation_Threshold
        l. 相关性修正中降水放大倍率: rate
        m. 时间升尺度方法：how2hightempor
        n. 偏差校正方法：how2corrector
        o. SWAT输出数据有效数字：significant_digits
"""
import time
import glob
import numpy as np
import pandas as pd

from Tools.DateHandler import dateHandler
from Tools.Tool import create_path, sum_without_nan

from MainProcess import read_data, spatial_downscaling, multi_mode_ensemble_averaging, temporal_downscaling, \
    swat_format_writer, multi_site_correlation_correction, frequency_decomposition

if __name__ == '__main__':
    """************************************************参数设置*******************************************************"""
    # 1. 输入数据路径
    # 原始日尺度观测数据：
    gauge_path = glob.glob(r"../dataFile/Climate/*.txt")
    # 气象站站点信息：
    stationInfo = r"../dataFile/Station/TNR_756.csv"
    sta_location = pd.read_csv(stationInfo, usecols=["Station", "LAT", "LONG", "ELEVATION"]).set_index("Station")
    sta_location.index = sta_location.index.astype(str)
    # 输入/输出时间尺度: (Monthly or Daily)
    scale = "Monthly"
    # 原始预测因子 / GCM父文件夹：
    gcm_father_path = r"/Volumes/Hydrological Group/1. IPCC/Data" + "/" + scale + "/"
    # GCM命名：
    gcm_basename = [
        # "ACCESS-CM2",#
        # "ACCESS-ESM1-5",
        # "AWI-CM-1-1-MR",
        # "CanESM5",#
        # "CESM2-WACCM",#
        # "CMCC-CM2-SR5",
        # "CMCC-ESM2",#
        # "CNRM-ESM2-1",
        # "EC-Earth3",
        "EC-Earth3-Veg",#
        # "EC-Earth3-Veg-LR",#
        # "FGOALS-f3-L",#
        # "FGOALS-g3",
        # "GFDL-ESM4",#
        # "INM-CM4-8",
        # "INM-CM-5-0",#
        # "IPSL-CM6A-LR",#
        # "KACE-1-0-G",#
        # "MIROC6",#
        # "MPI-ESM1-2-HR",
        # "MRI-ESM2-0",#
    ]
    # 预测因子名：
    predictor_name = [
        # Predictands variables
        "tasmin",
        "tasmax",
        "pr",
        "hurs",
        "rsds",
        "sfcWind",
        # Surface variables
        "psl",
        "huss",
        "tas",
        "clt",
        "prc",
        "prsn",
        "hfls",
        "hfss",
        "rlds",
        "rlus",
        "rsus",
        "rlut",
        # Upper-atmosphere variables (500 hpa and 850 hpa)
        "ta500", "hur500", "hus500", "wap500", "va500", "ua500", "zg500",
        "ta850", "hur850", "hus850", "wap850", "va850", "ua850", "zg850",
    ]
    # 历史情景命名：
    historical = "historical"    # 读取下一个GCM时改回historical，取消下方ssp126注释[如果匹配报错，还原这里，注释掉262-264]
    # 未来情景命名：
    scenarios = [       #提取预测因子数据: ACCESS-CM2-ssp585
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    ]
    ####################################################################################################################
    # 2. 输出数据路径
    # 输出结果文件夹
    result_path = r"/Volumes/FluxGroup/GCM_P/"
    # 观测数据输出：
    observe_path = result_path + r"Predictand/Original/"
    # 预测因子数据输出：
    predictor_path = result_path + r"Predictor/Original/All" + "/" + scale + "/"
    # 预测因子数据匹配输出：
    matching_predictor_path = result_path + r"Predictor/Original/Matching" + "/" + scale + "/"
    # 降尺度结果：
    downscaling_path = result_path + r"Downscaling" + "/"
    # 多站点相关调整结果：
    multicorr_path = result_path + r"MSCC" + "/"
    # 降尺度模型备份：
    model_path = result_path + r"Model" + "/"
    # 缩放器备份：
    scaler_path = result_path + r"Model/Scaler" + "/"
    # SWAT格式输出：
    swat_path = result_path + r"SWAT" + "/"
    # 创建文件夹
    paths = [
        result_path, observe_path, predictor_path,
        matching_predictor_path, downscaling_path,
        multicorr_path, model_path, scaler_path, swat_path
    ]
    for path in paths:
        create_path(path=path)
    ####################################################################################################################
    # 3. 数据处理参数
    # GCM是否存在闰年记录：
    gcm_record_leap = {
        "ACCESS-CM2": True,
        "ACCESS-ESM1-5": True,
        "AWI-CM-1-1-MR": True,
        "CanESM5": True,
        "CESM2-WACCM": True,
        "CMCC-CM2-SR5": True,
        "CMCC-ESM2": True,
        "CNRM-ESM2-1": True,
        "EC-Earth3": True,
        "EC-Earth3-Veg": True,
        "EC-Earth3-Veg-LR": True,
        "FGOALS-f3-L": True,
        "FGOALS-g3": True,
        "GFDL-ESM4": True,
        "INM-CM4-8": True,
        "INM-CM-5-0": True,
        "IPSL-CM6A-LR": True,
        "KACE-1-0-G": True,
        "MIROC6": True,
        "MPI-ESM1-2-HR": True,
        "MRI-ESM2-0": True,
    }
    # 空缺值
    nan_value = -99
    # EEMD分解参数
    max_imf = 7
    eemd_figure = False
    decomp_scale = "Monthly"
    # 空间降尺度方法：
    spatial_downscaling_method = [
        "MLP",
    ]
    # 多模式集合平均方法
    mmea_method = "bma_mmea"
    # 被预测变量
    tand_values = [
        "PRE",
        # "RHU",
        # "RS",
        # "MINTEM",
        # "MAXTEM",
        # "WIN"
    ]
    # 非负变量：
    condition_values = ["PRE", "RHU", "RS", "WIN"]
    # 概率变量：
    occur_values = ["PRE"]
    # 温度变量标签：
    temperature_label = "TEM"
    # 降尺度历史校验时段：
    periods = ["1999/1/1", "2014/12/31"]
    start = pd.to_datetime(periods[0])
    end = pd.to_datetime(periods[1])
    # 后处理时间尺度
    stats_scales = [
        "Monthly",
    ]
    # 降水发生下限
    precipitation_Threshold = 0.1
    # 相关性修正中降水放大倍率
    rate = 1e3
    # 时间升尺度方法：
    how2hightempor = {
        "PRE": sum_without_nan,
        "RHU": np.nanmean,
        "RS": sum_without_nan,
        "MINTEM": np.nanmean,  # np.nanmin,
        "MAXTEM": np.nanmean,  # np.nanmax,
        "WIN": np.nanmean
    }
    # 偏差校正方法：
    how2corrector = {
        "PRE": "scaling",
        "RHU": "translation",
        "RS": "scaling",
        "MINTEM": "translation",
        "MAXTEM": "translation",
        "WIN": "translation"
    }
    # SWAT输出数据有效数字：
    significant_digits = {
        "PRE": 1,
        "RHU": 3,
        "RS": 3,
        "MINTEM": 2,
        "MAXTEM": 2,
        "WIN": 3
    }
    """************************************************参数设置*******************************************************"""

    """********************************************降尺度及校正流程****************************************************"""
    start_Time = time.time()
    """读取数据"""
    # 被预测数据(Predictand/Original 宏才格式)
    # read_data.read_predictand_from_csv(input_path=observe_path+"Daily/", stations=sta_location.index.values,
    #                                     how2hightempor=how2hightempor, output_path=observe_path, nan_value=nan_value)
    # 预测因子
    read_data.read_predictor(father_path=gcm_father_path, scenarios=scenarios, gcm_basename=gcm_basename,
                               sta_location=sta_location, output_path=predictor_path, historical_label=historical)
    # 预测因子匹配
    read_data.matching_predictor(father_path=predictor_path, scenarios=scenarios, gcm_basename=gcm_basename,
                                  predictor_name=predictor_name, sta_location=sta_location,
                                  output_path=matching_predictor_path, historical_label=historical)
    """空间降尺度"""
     # 历史模型校正
    spatial_downscaling.historical_downscaling(observe_path=observe_path,
                                                predictor_path=matching_predictor_path,
                                                tand_values=tand_values, downscaling_path=downscaling_path,
                                                model_path=model_path, sta_location=sta_location, periods=periods,
                                                gcm_record_leap=gcm_record_leap,
                                                gcm_basename=gcm_basename, condition_values=condition_values,
                                                historical=historical, downscaling_method=spatial_downscaling_method
                                                )
    # 未来预测
    spatial_downscaling.future_predict(predictor_path=matching_predictor_path,
                                       tand_values=tand_values, gcm_basename=gcm_basename,
                                       scenarios=scenarios, sta_location=sta_location, model_path=model_path,
                                       output_path=downscaling_path, condition_values=condition_values,
                                       downscaling_method=spatial_downscaling_method)
    """多模式集合平均"""
    # MMEA模型构建
    multi_mode_ensemble_averaging.historical_mmea(sta_location=sta_location, observe_path=observe_path,
                                                  downscaling_path=downscaling_path, model_path=model_path,
                                                  gcm_basename=gcm_basename, periods=periods,
                                                  gcm_record_leap=gcm_record_leap,
                                                  downscaling_method=spatial_downscaling_method,
                                                  mmea_method=mmea_method,
                                                  condition_values=condition_values, historical=historical)
    # MMEA模型应用
    multi_mode_ensemble_averaging.future_mmea(sta_location=sta_location, downscaling_path=downscaling_path,
                                              model_path=model_path,
                                              gcm_basename=gcm_basename, scenarios=scenarios,
                                              downscaling_method=spatial_downscaling_method, mmea_method=mmea_method,
                                              condition_values=condition_values)
    """********************************************降尺度及校正流程****************************************************"""

    now_Time = time.time()
    print("\rDownscaling Finished, total time: %s" % dateHandler.uptimereport(start_Time, now_Time))
