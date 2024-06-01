# coding=utf-8
import pandas as pd
from Tools.Tool import select_time, fill_nan, create_path
from Downscaling.multi_model_ensemble_averaging import MultiModeEnsembleAveraging, get_mmea


def historical_mmea(sta_location, observe_path, downscaling_path, model_path, gcm_basename, periods,
                    gcm_record_leap, downscaling_method, mmea_method, condition_values, historical):
    for station in sta_location.index.values:
        monthly_obs = pd.read_csv(observe_path + "Monthly/" + station + ".csv", index_col="Date")  # 读取数据
        monthly_obs = select_time(monthly_obs, periods, leap=True)
        monthly_obs = fill_nan(monthly_obs)

        print("降尺度结果-多模式集合平均: %s" % station)
        for method in downscaling_method:
            # 结果输出路径
            downscaling_df_path = downscaling_path + "Monthly/MMEA/" + method + "/" + historical + "/"
            create_path(downscaling_df_path)
            # 模型备份路径
            model_backup_path = model_path + "multi_mode_ensemble_averaging/" + method + "/" + station + "/"
            create_path(model_backup_path)

            multi_dict = {}
            """读取各模式降尺度结果"""
            for i, basename in enumerate(gcm_basename):
                # 月尺度输出文件夹
                gcm_df_path = downscaling_path + "Monthly/" + basename + "/" + method + "/" + historical + "/"

                # 读取数据
                method_df = pd.read_csv(gcm_df_path + station + ".csv", index_col="Date")

                # 时间列格式转换
                method_df = select_time(method_df, periods, leap=gcm_record_leap[basename])

                # 结果储存
                multi_dict[basename] = method_df

            obj = MultiModeEnsembleAveraging(monthly_obs, multi_dict, condition_values)
            mmea_df = obj.get_mmea(model_backup_path=model_backup_path, method=mmea_method)
            mmea_df.to_csv(downscaling_df_path + station + ".csv")


def future_mmea(sta_location, downscaling_path, model_path, gcm_basename, scenarios, downscaling_method, mmea_method,
                condition_values):
    for station in sta_location.index.values:
        print("降尺度结果-多模式集合平均: %s" % station)
        for method in downscaling_method:
            # 模型备份路径
            model_backup_path = model_path + "multi_mode_ensemble_averaging/" + method + "/" + station + "/"
            create_path(model_backup_path)
            for ssp in scenarios:
                # 结果输出路径
                downscaling_df_path = downscaling_path + "Monthly/MMEA/" + method + "/" + ssp + "/"
                create_path(downscaling_df_path)

                multi_dict = {}
                """读取各模式降尺度结果"""
                for i, basename in enumerate(gcm_basename):
                    # 月尺度输出文件夹
                    gcm_df_path = downscaling_path + "Monthly/" + basename + "/" + method + "/" + ssp + "/"

                    # 读取数据
                    method_df = pd.read_csv(gcm_df_path + station + ".csv", index_col="Date")

                    # 时间列格式转换
                    method_df.index = pd.to_datetime(method_df.index)  # 时间列格式转换

                    # 结果储存
                    multi_dict[basename] = method_df
                mmea_df = get_mmea(gcm_dict=multi_dict, condition_values=condition_values,
                                   model_backup_path=model_backup_path, method=mmea_method)
                mmea_df.to_csv(downscaling_df_path + station + ".csv")
