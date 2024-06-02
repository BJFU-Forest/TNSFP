# coding=utf-8
import numpy as np
import pandas as pd
from SpaCorrRebuild.ShuffleProcedure import multiSiteCorr, shuffleProcedure
from Tools.Tool import create_path, select_time


def mutilSiteCorr(sta_location, observe_path, multicorr_path, periods, gcm_basename, downscaling_path,
                  spatial_downscaling_method, tand_values, scenarios, occur_values,
                  historical="historical", threshold=0.1, rate=1e3):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical)

    for i, basename in enumerate(gcm_basename):
        for method in spatial_downscaling_method:
            post_shuffled = {}
            date = [None] * len(scenarios_self)
            date_switch = {}
            for val in tand_values:
                for month in np.arange(1, 13):
                    print("多站点相关性校正: %s-%s-%s (%s)" % (basename, val, month, method))
                    observe = pd.DataFrame()
                    generated = pd.DataFrame()
                    need_fill = val in occur_values

                    # 历史发生器结果校正及目标相关性更新
                    for station in sta_location.index.values:
                        sta_obs = pd.read_csv(observe_path + "Daily/" + station + ".csv", index_col="Date")
                        sta_obs = select_time(sta_obs, periods, leap=False)
                        date[0] = sta_obs.index.values if date[0] is None else date[0]
                        month_select = pd.to_datetime(sta_obs.index).month == month
                        if "_".join([historical, str(month)]) not in date_switch.keys():
                            date_switch["_".join([historical, str(month)])] = month_select

                        observe[station] = sta_obs[month_select][val].values
                        his_gen = pd.read_csv(
                            downscaling_path + "/".join(["Daily", basename, method, historical, station + ".csv"]),
                            index_col="Date")
                        generated[station] = select_time(his_gen, periods, leap=False)[month_select][val].values

                    post_shuffled["_".join([historical, val, str(month)])], adjust_cos = multiSiteCorr(
                        obs=observe.values,
                        gen=generated.values,
                        need_fill=need_fill,
                        threshold=threshold,
                        rate=rate)
                    # 未来发生器结果校正
                    for s, scenario in enumerate(scenarios):
                        sce_generated = pd.DataFrame()
                        for station in sta_location.index.values:
                            sce_gen = pd.read_csv(
                                downscaling_path + "/".join(["Daily", basename, method, scenario, station + ".csv"]),
                                index_col="Date")
                            date[s + 1] = sce_gen.index.values if date[s + 1] is None else date[s + 1]
                            month_select = pd.to_datetime(sce_gen.index).month == month
                            if "_".join([scenario, str(month)]) not in date_switch.keys():
                                date_switch["_".join([scenario, str(month)])] = month_select

                            sce_generated[station] = pd.read_csv(
                                downscaling_path + "/".join(["Daily", basename, method, scenario, station + ".csv"]),
                                index_col="Date")[month_select][val].values
                        post_shuffled["_".join([scenario, val, str(month)])] = shuffleProcedure(Cobs=adjust_cos,
                                                                                                gen=sce_generated.values,
                                                                                                need_fill=need_fill,
                                                                                                threshold=threshold,
                                                                                                rate=rate)
            # 多站点相关性校正结果导出
            for s, scenario in enumerate(scenarios_self):
                for j, station in enumerate(sta_location.index.values):
                    print("校正结果导出: %s-%s-%s (%s)" % (basename, scenario, station, method))
                    shuffled_df_path = multicorr_path + "/".join(
                        ["Daily", basename, method, scenario, station + ".csv"])
                    create_path(shuffled_df_path)
                    shuffled = pd.DataFrame({"Date": date[s]}).set_index("Date")
                    shuffled[tand_values] = np.full(np.asarray([shuffled.index.values] * len(tand_values)).T.shape,
                                                    np.nan)
                    for val in tand_values:
                        for month in np.arange(1, 13):
                            shuffled[val][date_switch["_".join([scenario, str(month)])]] = \
                                post_shuffled["_".join([scenario, val, str(month)])][:, j]
                    shuffled.to_csv(shuffled_df_path, index=True)

