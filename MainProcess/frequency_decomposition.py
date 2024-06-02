# coding=utf-8
import glob

import joblib
import numpy as np
import pandas as pd
import re
from PyEMD.EEMD import EEMD
from PyEMD.EMD import EMD
from PyEMD.visualisation import Visualisation
from statsmodels.tsa.seasonal import STL
from sklearn import preprocessing

from Tools.Tool import create_path, select_time, fill_nan

import warnings

warnings.filterwarnings("ignore")


def decomposition(tand_values, sta_location, periods, gcm_basename=None, scenarios=None,
                  decompose=True, decomp_X=False, decom_method="EEMD", max_imf=-1, figure=False,
                  predictand_path=None, predictor_path=None, decomp_tand_path=None, decomp_tor_path=None,
                  historical_label="historical", scale="Monthly",
                  matching=False, predictor_name=None, match_tand_path=None, match_tor_path=None,
                  merge=False, merge_group=None, merge_tand_path=None, merge_tor_path=None,
                  scaling=False, scaling_input_path=None, tor_scaler_path=None):
    if decompose:
        decom_method = decom_method.upper()
        method_list = ["EEMD", "EMD", "STL"]
        if decom_method not in method_list:
            raise Exception("%s method not support!" % decom_method)
        if decom_method == "EEMD":
            eemdhandler(observe_path=predictand_path,
                        tand_values=tand_values,
                        predictor_path=predictor_path,
                        eemd_observe_path=decomp_tand_path,
                        eemd_predictor_path=decomp_tor_path,
                        sta_location=sta_location,
                        periods=periods,
                        gcm_basename=gcm_basename,
                        scenarios=scenarios,
                        historical_label=historical_label,
                        scale=scale,
                        max_imf=max_imf,
                        need_eemd_X=decomp_X,
                        figure=figure)
        elif decom_method == "EMD":
            emdhandler(observe_path=predictand_path,
                       tand_values=tand_values,
                       predictor_path=predictor_path,
                       emd_observe_path=decomp_tand_path,
                       emd_predictor_path=decomp_tor_path,
                       sta_location=sta_location,
                       periods=periods,
                       gcm_basename=gcm_basename,
                       scenarios=scenarios,
                       historical_label=historical_label,
                       scale=scale,
                       max_imf=max_imf,
                       need_emd_X=decomp_X,
                       figure=figure)
        elif decom_method == "STL":
            stlhandler(observe_path=predictand_path,
                       tand_values=tand_values,
                       predictor_path=predictor_path,
                       stl_observe_path=decomp_tand_path,
                       stl_predictor_path=decomp_tor_path,
                       sta_location=sta_location,
                       periods=periods,
                       gcm_basename=gcm_basename,
                       scenarios=scenarios,
                       historical_label=historical_label,
                       scale=scale,
                       figure=figure,
                       need_stl_X=decomp_X)
    if matching:
        matching_decomposition(predictand_path=decomp_tand_path,
                               predictor_path=decomp_tor_path,
                               gcm_basename=gcm_basename,
                               tand_values=tand_values,
                               predictor_name=predictor_name,
                               scenarios=scenarios,
                               match_tand_path=match_tand_path,
                               match_tor_path=match_tor_path,
                               sta_location=sta_location,
                               max_imf=max_imf,
                               historical_label=historical_label)
    if merge:
        merge_decomposition(predictand_path=match_tand_path,
                            predictor_path=match_tor_path,
                            gcm_basename=gcm_basename,
                            tand_values=tand_values,
                            predictor_name=predictor_name,
                            scenarios=scenarios,
                            merge_tand_path=merge_tand_path,
                            merge_tor_path=merge_tor_path,
                            sta_location=sta_location,
                            merge_group=merge_group,
                            historical_label=historical_label)
    if scaling:
        scaling_predictor(predictor_path=predictor_path,
                          decomp_predictor_path=merge_tor_path if scaling_input_path is None else scaling_input_path,
                          predictor_name=predictor_name,
                          gcm_basename=gcm_basename,
                          sta_location=sta_location,
                          periods=periods,
                          tor_scaler_path=tor_scaler_path,
                          scenarios=scenarios,
                          historical_label=historical_label)


def emdhandler(observe_path, tand_values, predictor_path, emd_observe_path, emd_predictor_path, sta_location,
               periods,
               gcm_basename, scenarios, historical_label="historical", scale="Monthly", max_imf=8,
               need_emd_X=False, figure=False):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    # 创建EMD分解器
    emd = EMD(trials=2000, parallel=True, processes=12)

    for station in sta_location.index.values:
        print("集合经验模态分解: %s" % station)
        # observe EMD
        # 创建输出文件夹
        emd_Y_path = emd_observe_path + scale + "/" + station + ".csv"
        create_path(emd_Y_path)
        # 读取数据
        y_data = pd.read_csv(observe_path + scale + "/" + station + ".csv", index_col=0, parse_dates=True)
        y_data = select_time(y_data, periods, leap=True)
        y_data = fill_nan(y_data)

        emd_y_data = pd.DataFrame({"Date": y_data.index.values}).set_index("Date")
        print("      EMD Y:")
        for val in tand_values:
            # use EMD decompose signal into Intrinsic Mode Functions
            print("\r          Station: %s Predictand: %s" % (station, val), end="")
            emd.emd(S=y_data[val].values, T=y_data.index.values, max_imf=max_imf)
            imfs, res = emd.get_imfs_and_residue()
            # 生成pd.DataFrame
            Y = pd.DataFrame({"Date": y_data.index.values}).set_index("Date")
            for n in range(imfs.shape[0]):
                Y[val + "_imf" + str(n + 1)] = imfs[n, :]
            Y[val + "_residual"] = res
            emd_y_data = pd.concat([emd_y_data, Y], axis=1)
            print("     Finished. IFM numbers: %d" % Y.shape[1], end="")

            # 绘制EMD分解结果
            if figure:
                emd_fig_path = emd_observe_path + "/".join([scale, station, val]) + ".jpg"
                create_path(emd_fig_path)
                vis = Visualisation()
                emd_fig = vis.plot_imfs(imfs=imfs, residue=res, t=y_data.index.values, include_residue=True)
                emd_fig.savefig(emd_fig_path)
        emd_y_data.to_csv(emd_Y_path, index=True)

        if need_emd_X:
            # predictor EMD
            print("\n      EMD X:")
            for i, basename in enumerate(gcm_basename):
                for ssp in scenarios_self:
                    # 创建输出文件夹
                    emd_X_path = emd_predictor_path + "/".join([basename, ssp, station]) + ".csv"
                    create_path(emd_X_path)
                    # 读取数据
                    X_data = pd.read_csv(predictor_path + "/".join([basename, ssp, station]) + ".csv", index_col=0,
                                         parse_dates=True)
                    X_data = select_time(X_data, periods, leap=True) if ssp == historical_label else X_data
                    X_data = fill_nan(X_data, drop=True, drop_threshold=0.05)

                    # 分解X
                    emd_X_data = pd.DataFrame({"Date": X_data.index.values}).set_index("Date")
                    for tor in X_data.columns.values:
                        print(
                            "\r          EMD X Station: %s GCM: %s ssp: %s Predictor: %s" % (
                            station, basename, ssp, tor),
                            end="")
                        try:
                            emd.emd(S=X_data[tor].values, T=X_data.index.values, max_imf=max_imf)
                        except:
                            raise Exception("ERR")
                        imfs_t, res_t = emd.get_imfs_and_residue()

                        X = pd.DataFrame({"Date": X_data.index.values}).set_index("Date")
                        for n in range(imfs_t.shape[0]):
                            X[tor + "_imf" + str(n + 1)] = imfs_t[n, :]
                        X[tor + "_residual"] = res_t
                        emd_X_data = pd.concat([emd_X_data, X], axis=1)
                        print("     Finished. IFM numbers: %d" % X.shape[1], end="")
                    emd_X_data.to_csv(emd_X_path, index=True)
        print("\n")


def eemdhandler(observe_path, tand_values, predictor_path, eemd_observe_path, eemd_predictor_path, sta_location,
                periods,
                gcm_basename, scenarios, historical_label="historical", scale="Monthly", max_imf=8,
                need_eemd_X=False, figure=False):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    # 创建EEMD分解器
    eemd = EEMD(trials=2000, parallel=True, processes=12, separate_trends=True)
    eemd.noise_seed(12345)

    for station in sta_location.index.values:
        print("集合经验模态分解: %s" % station)
        # observe EEMD
        # 创建输出文件夹
        eemd_Y_path = eemd_observe_path + scale + "/" + station + ".csv"
        create_path(eemd_Y_path)
        # 读取数据
        y_data = pd.read_csv(observe_path + scale + "/" + station + ".csv", index_col=0, parse_dates=True)
        y_data = select_time(y_data, periods, leap=True)
        y_data = fill_nan(y_data)

        eemd_y_data = pd.DataFrame({"Date": y_data.index.values}).set_index("Date")
        print("      EEMD Y:")
        for val in tand_values:
            # use EEMD decompose signal into Intrinsic Mode Functions
            print("\r          Station: %s Predictand: %s" % (station, val), end="")
            eemd.eemd(S=y_data[val].values, T=y_data.index.values, max_imf=max_imf)
            imfs, res = eemd.get_imfs_and_residue()
            # 生成pd.DataFrame
            Y = pd.DataFrame({"Date": y_data.index.values}).set_index("Date")
            for n in range(imfs.shape[0]):
                Y[val + "_imf" + str(n + 1)] = imfs[n, :]
            Y[val + "_residual"] = res
            eemd_y_data = pd.concat([eemd_y_data, Y], axis=1)
            print("     Finished. IFM numbers: %d" % Y.shape[1], end="")

            # 绘制EEMD分解结果
            if figure:
                eemd_fig_path = eemd_observe_path + "/".join([scale, station, val]) + ".jpg"
                create_path(eemd_fig_path)
                vis = Visualisation()
                emd_fig = vis.plot_imfs(imfs=imfs, residue=res, t=y_data.index.values, include_residue=True)
                emd_fig.savefig(eemd_fig_path)
        eemd_y_data.to_csv(eemd_Y_path, index=True)

        if need_eemd_X:
            # predictor EEMD
            print("\n      EEMD X:")
            for i, basename in enumerate(gcm_basename):
                for ssp in scenarios_self:
                    # 创建输出文件夹
                    eemd_X_path = eemd_predictor_path + "/".join([basename, ssp, station]) + ".csv"
                    create_path(eemd_X_path)
                    # 读取数据
                    X_data = pd.read_csv(predictor_path + "/".join([basename, ssp, station]) + ".csv",
                                         index_col=0,
                                         parse_dates=True)
                    X_data = select_time(X_data, periods, leap=True) if ssp == historical_label else X_data
                    X_data = fill_nan(X_data, drop=True, drop_threshold=0.05)

                    # 分解X
                    eemd_X_data = pd.DataFrame({"Date": X_data.index.values}).set_index("Date")
                    for tor in X_data.columns.values:
                        print(
                            "\r          EEMD X Station: %s GCM: %s ssp: %s Predictor: %s" % (
                            station, basename, ssp, tor),
                            end="")
                        eemd.eemd(S=X_data[tor].values, T=X_data.index.values, max_imf=max_imf)
                        imfs_t, res_t = eemd.get_imfs_and_residue()

                        X = pd.DataFrame({"Date": X_data.index.values}).set_index("Date")
                        for n in range(imfs_t.shape[0]):
                            X[tor + "_imf" + str(n + 1)] = imfs_t[n, :]
                        X[tor + "_residual"] = res_t
                        eemd_X_data = pd.concat([eemd_X_data, X], axis=1)
                        print("     Finished. IFM numbers: %d" % X.shape[1], end="")
                    eemd_X_data.to_csv(eemd_X_path, index=True)
        print("\n")


def stlhandler(observe_path, tand_values, predictor_path, stl_observe_path, stl_predictor_path, sta_location,
               periods, gcm_basename, scenarios, historical_label="historical", scale="Monthly",
               figure=False, need_stl_X=False):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    for station in sta_location.index.values:
        print("STL: %s" % station)
        # observe STL
        # 创建输出文件夹
        stl_Y_path = stl_observe_path + scale + "/" + station + ".csv"
        create_path(stl_Y_path)
        # 读取数据
        y_data = pd.read_csv(observe_path + scale + "/" + station + ".csv", index_col=0, parse_dates=True)
        y_data = select_time(y_data, periods, leap=True)
        y_data = fill_nan(y_data)

        stl_y_data = pd.DataFrame({"Date": y_data.index.values}).set_index("Date")
        print("      STL Y:")
        for val in tand_values:
            # use STL decompose signal into Intrinsic Mode Functions
            print("\r          Station: %s Predictand: %s" % (station, val), end="")
            # 创建STL分解器
            stl_y = STL(y_data[val], period=12, seasonal=13, trend=121)
            res_y = stl_y.fit()

            # 获取分解后的结果
            y_trend = res_y.trend
            y_seasonal = res_y.seasonal
            y_residual = res_y.resid
            # 绘制STL分解结果
            if figure:
                stl_fig_path = stl_observe_path + "/".join([scale, station, val]) + ".jpg"
                create_path(stl_fig_path)
                stl_fig = res_y.plot()
                stl_fig.savefig(stl_fig_path)
            # 生成pd.DataFrame
            Y = pd.DataFrame({"Date": y_data.index.values,
                              val + "_imf1": y_trend,
                              val + "_imf2": y_seasonal,
                              val + "_residual": y_residual}).set_index("Date")
            stl_y_data = pd.concat([stl_y_data, Y], axis=1)
            print("     Finished. IFM numbers: %d" % Y.shape[1], end="")
        stl_y_data.to_csv(stl_Y_path, index=True)

        if need_stl_X:
            # predictor STL
            print("\n      STL X:")
            for i, basename in enumerate(gcm_basename):
                for ssp in scenarios_self:
                    # 创建输出文件夹
                    stl_X_path = stl_predictor_path + "/".join([basename, ssp, station]) + ".csv"
                    create_path(stl_X_path)
                    # 读取数据
                    X_data = pd.read_csv(predictor_path + "/".join([basename, ssp, station]) + ".csv", index_col=0,
                                         parse_dates=True)
                    X_data = select_time(X_data, periods, leap=True) if ssp == historical_label else X_data
                    X_data = fill_nan(X_data, drop=True, drop_threshold=0.05)

                    # 分解X
                    stl_X_data = pd.DataFrame({"Date": X_data.index.values}).set_index("Date")
                    for tor in X_data.columns.values:
                        print(
                            "\r          STL X Station: %s GCM: %s ssp: %s Predictor: %s" % (
                            station, basename, ssp, tor),
                            end="")
                        # 创建STL分解器
                        stl_x = STL(X_data[tor], period=12, seasonal=13, trend=121)
                        res_x = stl_x.fit()

                        # 获取分解后的结果
                        x_trend = res_x.trend
                        x_seasonal = res_x.seasonal
                        x_residual = res_x.resid
                        # 生成pd.DataFrame
                        X = pd.DataFrame({"Date": X_data.index.values,
                                          tor + "_imf1": x_trend,
                                          tor + "_imf2": x_seasonal,
                                          tor + "_residual": x_residual}).set_index("Date")
                        stl_X_data = pd.concat([stl_X_data, X], axis=1)
                        print("     Finished. IFM numbers: %d" % X.shape[1], end="")
                    stl_X_data.to_csv(stl_X_path, index=True)
        print("\n")


def matching_decomposition(predictand_path, predictor_path, gcm_basename, tand_values, predictor_name,
                           scenarios, match_tand_path, match_tor_path, sta_location, max_imf=-1,
                           historical_label="historical"):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    print("溢出IMF裁剪: ")
    matc_tand_df_path = match_tand_path + "/Monthly/"
    create_path(matc_tand_df_path)
    for station in sta_location.index:
        print("\r     气象站: %s" % station, end="")
        predictand = pd.read_csv(predictand_path + "/Monthly/" + station + r".csv", index_col=0, parse_dates=True)
        predictand = overflow_imf_cut(predictand, need_values=tand_values, max_imf=max_imf)
        predictand.to_csv(matc_tand_df_path + station + r".csv", index=True)

    for basename in gcm_basename:
        print("\n匹配EEMD预测因子: %s " % basename)
        for station in sta_location.index:
            print("\r     气象站: %s" % station, end="")
            matching_cols = None
            for scenario in scenarios_self:  # 溢出裁剪
                predictor = pd.read_csv(predictor_path + basename + "/" + scenario + "/" + station + r".csv",
                                        index_col=0, parse_dates=True)
                predictor = overflow_imf_cut(predictor, need_values=predictor_name, max_imf=max_imf)
                matching_cols = predictor.columns if scenario == historical_label else np.intersect1d(matching_cols,
                                                                                                      predictor.columns)
            for scenario in scenarios_self:
                matc_tor_df_path = match_tor_path + basename + "/" + scenario + "/"
                create_path(matc_tor_df_path)
                predictor = pd.read_csv(predictor_path + basename + "/" + scenario + "/" + station + r".csv",
                                        index_col=0, parse_dates=True)
                predictor = unmatched_imf_cut(predictor, matching_cols)
                predictor.to_csv(matc_tor_df_path + station + ".csv")
            print("\r", end="")


def merge_decomposition(predictand_path, predictor_path, gcm_basename, tand_values, predictor_name,
                        scenarios, merge_tand_path, merge_tor_path, sta_location, merge_group=None,
                        historical_label="historical"):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    merge_group = [(2, 4)] if merge_group is None else merge_group

    print("频率融合: ")
    merge_tand_df_path = merge_tand_path + "/Monthly/"
    create_path(merge_tand_df_path)
    for station in sta_location.index:
        print("    气象站: %s" % station)
        predictand = pd.read_csv(predictand_path + "/Monthly/" + station + r".csv", index_col=0, parse_dates=True)
        new_predictand = pd.DataFrame(index=predictand.index)
        print("         融合预测因子", end="")
        for tand in tand_values:
            tand_df = merge_imf(predictand[predictand.columns[predictand.columns.map(lambda x: tand in x)]],
                                merge_group=merge_group)
            new_predictand = pd.concat([new_predictand, tand_df], axis=1)
        new_predictand.to_csv(merge_tand_df_path + station + r".csv", index=True)

        for basename in gcm_basename:
            print("\r         融合预测因子: %s " % basename, end="")
            for scenario in scenarios_self:
                merge_tor_df_path = merge_tor_path + basename + "/" + scenario + "/"
                create_path(merge_tor_df_path)
                predictor = pd.read_csv(predictor_path + basename + "/" + scenario + "/" + station + r".csv",
                                        index_col=0, parse_dates=True)
                new_predictor = pd.DataFrame(index=predictor.index)
                for tor in predictor_name:
                    tor_df = merge_imf(predictor[predictor.columns[predictor.columns.map(lambda x: tor in x)]],
                                       merge_group=merge_group)
                    new_predictor = pd.concat([new_predictor, tor_df], axis=1)
                new_predictor.to_csv(merge_tor_df_path + station + ".csv")
        print("\r", end="")


def scaling_predictor(predictor_path, decomp_predictor_path, predictor_name, gcm_basename, sta_location, periods,
                      tor_scaler_path, scenarios, historical_label="historical"):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)
    print("创建预测因子缩放器: ")
    for basename in gcm_basename:
        print("\nGCM: %s " % basename)
        for station in sta_location.index:
            print("\r     气象站: %s" % station, end="")
            df = pd.DataFrame()
            for scenario in scenarios_self:
                predictor = pd.read_csv(predictor_path + basename + "/" + scenario + "/" + station + r".csv",
                                        index_col=0, parse_dates=True)
                predictor = select_time(predictor, periods=periods) if scenario == historical_label else predictor

                decomp_predictor = pd.read_csv(
                    decomp_predictor_path + basename + "/" + scenario + "/" + station + r".csv",
                    index_col=0, parse_dates=True)
                predictor = pd.concat([predictor, decomp_predictor], axis=1)
                predictor = predictor[predictor.columns[
                    predictor.columns.map(lambda x: re.search("|".join(predictor_name), x) is not None)]]
                df = predictor if scenario == historical_label else pd.concat([df, predictor], axis=0, join="inner")
            scaler_path = tor_scaler_path + "/".join([basename, station]) + "/"
            create_path(scaler_path)
            scaler_handler(df, scaler_path=scaler_path, name="predictor_all_scenario")
            print("\r", end="")


########################################################################################################################


def scaler_handler(df, scaler_path, name):
    # 创建文件夹
    create_path(scaler_path)
    # 备份标签
    df_label = df.columns.values
    np.save(scaler_path + name + "_label.npy", df_label)
    # 备份缩放器
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df.values)
    joblib.dump(scaler, scaler_path + name + "_min_max_scaler.pkl")


def overflow_imf_cut(df, need_values, max_imf=-1, imf_name="_imf", residual_name="_residual"):
    max_imf = np.inf if max_imf <= 0 else max_imf
    for col in df.columns:
        try:
            val, imf = re.findall(r"(\w+)%s(\d+)" % imf_name, col)[0]
        except:
            continue
        if (val in need_values) & (int(imf) > max_imf):
            df[val + imf_name + str(max_imf)] = df[val + imf_name + str(max_imf)] + df[val + imf_name + imf]
            df = df.drop(labels=val + imf_name + imf, axis=1)
    return df


def unmatched_imf_cut(df, matching_cols, imf_name="_imf", residual_name="_residual"):
    unmatched_cols = np.setdiff1d(df.columns, matching_cols)
    for col in unmatched_cols:
        try:
            val, imf = re.findall(r"(\w+)%s(\d+)" % imf_name, col)[0]
        except:
            continue
        imf_index = int(imf) - 1
        while True:
            try:
                df[val + imf_name + str(imf_index)] = df[val + imf_name + str(imf_index)] + df[val + imf_name + imf]
                break
            except:
                imf_index -= 1
        df = df.drop(labels=val + imf_name + imf, axis=1)
    return df


def merge_imf(df, merge_group, imf_name="_imf", residual_name="_residual"):
    merge_df = pd.DataFrame(index=df.index)
    imf_num = 0
    for col in df.columns:
        try:
            val, imf = re.findall(r"(\w+)%s(\d+)" % imf_name, col)[0]
            if any([(np.digitize(int(imf), group, right=True) != 1) or (int(imf) == group[0]) for group in
                    merge_group]):
                imf_num += 1
                merge_df[val + imf_name + str(imf_num)] = df[val + imf_name + imf]
            else:
                merge_df[val + imf_name + str(imf_num)] = merge_df[val + imf_name + str(imf_num)] + df[
                    val + imf_name + imf]
        except:
            try:
                val = re.findall(r"(\w+)%s" % residual_name, col)[0]
                merge_df[val + residual_name] = df[val + residual_name]
            except:
                continue
    return merge_df
