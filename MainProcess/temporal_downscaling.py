# coding=utf-8
import pandas as pd
from Tools.Tool import select_time, fill_nan, create_path
from Downscaling.temporal_downscaling import WeatherGenerator
from Downscaling.future_predictor import PredictFutureDaily


def historical_downscaling(sta_location, observe_path, model_path, periods, gcm_basename, downscaling_method,
                           downscaling_path, historical, how2hightempor, how2corrector, precipitation_Threshold):
    for station in sta_location.index.values:
        # 日尺度实测数据
        tand = pd.read_csv(observe_path + "Daily/" + station + ".csv", index_col="Date")  # 读取数据
        tand = select_time(tand, periods, leap=True)
        tand = fill_nan(tand)

        for i, basename in enumerate(gcm_basename):
            print("降尺度至日尺度: %s" % basename)
            for method in downscaling_method:
                # 模型备份文件夹
                model_pkl_path = model_path + "temporal_downscaling/" + basename + "/" + method + "/" + station + "/"
                create_path(model_pkl_path)

                # 日尺度降尺度输出文件夹
                daily_out_path = downscaling_path + "/Daily/" + basename + "/" + method + "/" + historical + "/"
                create_path(daily_out_path)

                # 月尺度降尺度数据
                month_df_path = downscaling_path + "/Monthly/" + basename + "/" + method + "/" + historical + "/"
                month_df = pd.read_csv(month_df_path + station + ".csv", index_col="Date")

                # 降尺度模型历史校正
                obj = WeatherGenerator(daily_observe=tand, monthly_estimate=month_df,
                                       aggregator=how2hightempor, corrector=how2corrector)
                daily_df = obj.monthly2daily(model_path=model_pkl_path, precipitation_Threshold=precipitation_Threshold)
                daily_df.to_csv(daily_out_path + station + ".csv")


def future_predict(sta_location, model_path, gcm_basename, scenarios, downscaling_method, downscaling_path,
                   how2hightempor, how2corrector, precipitation_Threshold):
    for station in sta_location.index.values:
        for i, basename in enumerate(gcm_basename):
            for ssp in scenarios:
                print("未来情景 %s 时间降尺度: %s" % (ssp, basename))
                for method in downscaling_method:
                    # 日尺度降尺度输出文件夹
                    daily_out_path = downscaling_path + "/Daily/" + basename + "/" + method + "/" + ssp + "/"
                    create_path(daily_out_path)

                    # 月尺度降尺度数据
                    month_df_path = downscaling_path + "/Monthly/" + basename + "/" + method + "/" + ssp + "/"
                    month_df = pd.read_csv(month_df_path + station + ".csv", index_col="Date")

                    # 降尺度模型预测未来气候
                    model_pkl_path = model_path + "temporal_downscaling/" + basename + "/" + method + "/" + station + "/"
                    model = PredictFutureDaily(monthly_estimate=month_df, aggregator=how2hightempor,
                                               corrector=how2corrector)
                    daily_df = model.predict(model_path=model_pkl_path, precipitation_Threshold=precipitation_Threshold)
                    daily_df.to_csv(daily_out_path + station + ".csv")
