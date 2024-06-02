# coding=utf-8
import os

import joblib
import pandas as pd
import numpy as np

from Tools.Tool import create_path, select_time, fill_nan
from Downscaling.spatial_downscaling import PredictorsDownscaling
from Downscaling.future_predictor import predict_future_monthly


def historical_downscaling(observe_path, predictor_path, tand_values, downscaling_path, model_path,
                           sta_location, periods, gcm_record_leap, gcm_basename, condition_values,
                           historical, downscaling_method,
                           decomp_tand_path=None, decomp_tor_path=None, tor_scaler_path=None,
                           select_best=False):
    """空间降尺度"""
    for i, basename in enumerate(gcm_basename):
        # 创建输出文件夹
        downscaling_df_path = downscaling_path + "/Monthly/" + basename + "/"

        print("统计降尺度: %s" % basename)
        for station in sta_location.index.values:
            # 模型储存位置
            model_pkl_path = model_path + "spatial_downscaling/" + basename + "/" + station + "/"
            create_path(model_pkl_path)
            # 读取数据
            tand = pd.read_csv(observe_path + "Monthly/" + station + ".csv", index_col="Date")
            tor = pd.read_csv(predictor_path + basename + "/" + historical + "/" + station + ".csv", index_col="Date")
            # 筛选时间
            tand = select_time(tand, periods, leap=True)
            tor = select_time(tor, periods, leap=gcm_record_leap[basename])
            new_time_index = np.intersect1d(tand.index, tor.index)
            tand = tand[np.isin(tand.index, new_time_index)]
            tor = tor[np.isin(tor.index, new_time_index)]
            if tand.shape[0] != tor.shape[0]:
                raise Exception("整理数据失败，请检查数据的时间格式")
            """数据清洗"""
            # 清洗存在缺失值的行
            tand = fill_nan(tand)
            # 匹配被预测数据和预测因子的时间
            tor = tor.loc[tand.index]
            # 用每月的均值填充tor中的空值
            tor = fill_nan(tor, drop=True, drop_threshold=0.05)
            """数据清洗"""

            """Decomposition输入处理"""
            if decomp_tand_path is not None:
                # 读取数据
                decomp_tand = pd.read_csv(decomp_tand_path + "Monthly/" + station + ".csv", index_col="Date")
                # 筛选时间
                decomp_tand = select_time(decomp_tand, periods, leap=True)
                if tand.shape[0] != decomp_tand.shape[0]:
                    raise Exception("整理数据失败，请检查数据的时间格式")
                # 匹配被预测数据和预测因子的时间
                decomp_tand = decomp_tand.loc[tand.index]
                # 用每月的均值填充tor中的空值
                decomp_tand = fill_nan(decomp_tand)
            else:
                decomp_tand = None

            if decomp_tor_path is not None:
                # 读取数据
                decomp_tor = pd.read_csv(decomp_tor_path + basename + "/" + historical + "/" + station + ".csv",
                                         index_col="Date")
                # 筛选时间
                decomp_tor = select_time(decomp_tor, periods, leap=gcm_record_leap[basename])
                if tand.shape[0] != decomp_tor.shape[0]:
                    raise Exception("整理数据失败，请检查数据的时间格式")
                # 匹配被预测数据和预测因子的时间
                decomp_tor = decomp_tor.loc[tand.index]
                # 用每月的均值填充tor中的空值
                decomp_tor = fill_nan(decomp_tor, drop=True, drop_threshold=0.05)
            else:
                decomp_tor = None
            """Decomposition输入处理"""

            """降尺度"""
            if tor_scaler_path is not None:
                tor_labels = np.load(
                    tor_scaler_path + "/".join([basename, station, "predictor_all_scenario_label.npy"]),
                    allow_pickle=True)
                tor_scaler = joblib.load(
                    tor_scaler_path + "/".join([basename, station, "predictor_all_scenario_min_max_scaler.pkl"]))
            else:
                tor_labels = None
                tor_scaler =None
            ds_df = downscaling(tand, tor, tand_values, condition_values, model_path=model_pkl_path,
                                downscaling_method=downscaling_method, decomp_obs=decomp_tand, decomp_tor=decomp_tor,
                                tor_scaler=tor_scaler, tor_labels=tor_labels, select_best=select_best)
            for method in downscaling_method:
                # 检验文件夹
                df_path = downscaling_df_path + method + "/" + historical + "/"
                create_path(df_path)
                # 输出降尺度结果
                ds_df[method].to_csv(df_path + station + ".csv")
            """降尺度"""


def future_predict(predictor_path, tand_values, gcm_basename, scenarios, sta_location, model_path,
                   output_path, condition_values, downscaling_method, decomp_tor_path=None, tor_scaler_path=None):
    """空间降尺度"""
    for i, basename in enumerate(gcm_basename):
        # 创建输出文件夹
        downscaling_df_path = output_path + "/Monthly/" + basename + "/"
        for ssp in scenarios:
            print("未来情景 %s 空间降尺度: %s" % (ssp, basename))
            for station in sta_location.index.values:
                # 读取数据
                tor_fur = pd.read_csv(predictor_path + "/".join([basename, ssp, station]) + ".csv", index_col="Date")
                if decomp_tor_path is not None:
                    decomp_fur = pd.read_csv(decomp_tor_path + "/".join([basename, ssp, station]) + ".csv",
                                             index_col="Date")
                    tor_fur = pd.concat([tor_fur, decomp_fur], axis=1)

                """数据清洗"""
                tor_fur.index = pd.to_datetime(tor_fur.index)  # 时间列格式转换
                tor_fur = fill_nan(tor_fur)
                """数据清洗"""

                """降尺度"""
                for method in downscaling_method:
                    print("GCM: %s Scenario: %s Method:%s Station: %s" % (basename, ssp, method, station))
                    # 结果输出位置
                    predict_output_path = downscaling_df_path + method + "/" + ssp + "/"
                    create_path(predict_output_path)
                    # 模型储存位置
                    model_pkl_path = model_path + "spatial_downscaling/" + basename + "/" + station + "/"
                    if tor_scaler_path is not None:
                        tor_labels = np.load(
                            tor_scaler_path + "/".join([basename, station, "predictor_all_scenario_label.npy"]),
                            allow_pickle=True)
                        tor_scaler = joblib.load(
                            tor_scaler_path + "/".join(
                                [basename, station, "predictor_all_scenario_min_max_scaler.pkl"]))
                    else:
                        tor_labels = None
                        tor_scaler = None
                    predict_df = predict_future(tor_fur, tand_values, model_pkl_path, condition_values, method,
                                                tor_scaler=tor_scaler, tor_labels=tor_labels)
                    # 输出降尺度结果
                    predict_df.to_csv(predict_output_path + station + ".csv")
                """降尺度"""


########################################################################################################################
def downscaling(observe, predictors, tand_values, condition_values, model_path,
                downscaling_method, decomp_obs=None, decomp_tor=None, tor_scaler=None, tor_labels=None,
                select_best=False):
    """
    调用降尺度方法将GCM数据降尺度到气象站尺度
    :param observe: 校准数据，一般为气象站观测数据
    :type observe: pd.DataFrame
    :param predictors:  历史预报因子数据，一般为GCM的historical情景输出
    :type predictors: pd.DataFrame
    :param tand_values:  待预报要素名
    :type tand_values: list
    :param condition_values: 非负变量
    :type condition_values: list
    :param model_path: 模型储存位置
    :type model_path: str
    :param downscaling_method: 降尺度方法
    :type downscaling_method: str
    :param decomp_obs: 分解后的校正数据，一般为气象站观测数据,可预先设置
    :type decomp_obs: pd.DataFrame OR None
    :param decomp_tor:  分解后，历史预报因子数据，一般为GCM的historical情景输出
    :type decomp_tor: pd.DataFrame OR None
    :return:
    """

    # 创建和observe同shape的df(nan)储存结果
    df_fit = observe.copy()
    df_fit[:] = np.full(observe.values.shape, np.nan)
    ds_df = {}
    for method in downscaling_method:
        ds_df[method] = df_fit.copy()
    # 确定性部分降尺度
    for val in tand_values:
        is_condition = True if val in condition_values else False
        ref_data = observe[[val]]
        model_backup_path = model_path + val + "/"
        create_path(model_backup_path)
        if decomp_obs is not None:
            val_col = decomp_obs.columns.values[np.asarray(
                [True if val in col else False for col in decomp_obs.columns.values])]
            ref_decomp = decomp_obs[val_col]
        else:
            ref_decomp = None
        # 空间降尺度
        obj = PredictorsDownscaling(ref_data=ref_data, predictors=predictors,
                                    model_backup_path=model_backup_path, train_size=0.5, max_predictors=5,
                                    method="forward", is_condition=is_condition,
                                    ref_decomp=ref_decomp, tor_decomp=decomp_tor,
                                    tor_scaler=tor_scaler, tor_labels=tor_labels, select_best=select_best)

        # 基于多元线性回归进行降尺度
        if "MLR" in downscaling_method:
            mlr_present = obj.multiple_linear_regressions()
            ds_df["MLR"][val] = mlr_present

        # 基于多层感知机模型进行降尺度
        if "MLP" in downscaling_method:
            mlp_present = obj.multilayer_perceptron()
            ds_df["MLP"][val] = mlp_present

    return ds_df


def predict_future(predictors, tand_values, model_path, condition_values, method, tor_scaler=None, tor_labels=None):
    """
    调用降尺度方法将GCM数据降尺度到气象站尺度
    :param predictors:  未来预报因子数据，一般为GCM的historical情景输出
    :type predictors: pd.DataFrame
    :param tand_values: 被预测因子
    :type tand_values: list
    :param model_path: 模型储存位置
    :type model_path: str
    :param condition_values: 非负变量
    :type condition_values: list
    :param method: 降尺度方法
    :type method: list
    :return:
    """

    # 创建和Reference同shape的df(nan)储存结果
    future_df = pd.DataFrame({"Date": predictors.index}).set_index("Date")

    # 确定性部分降尺度
    for val in tand_values:
        print("Predictor Future %s" % val)
        is_condition = True if val in condition_values else False
        model_backup = model_path + val + "/"

        # 空间降尺度预测未来
        future_val = predict_future_monthly(method=method, predictors=predictors, model_backup_path=model_backup,
                                            is_condition=is_condition, scaler=tor_scaler, labels=tor_labels)
        future_df[val] = future_val

    return future_df
