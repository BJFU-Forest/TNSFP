# coding=utf-8
import pandas as pd
import numpy as np
import re
import os
import glob

from DataReader.read_nc_by_xarray import get_xarray_from_nc as nc_reader
from Tools.Tool import create_path


def read_predictand(input_path, how2hightempor, output_path, nan_value=-99, temperature_label="TEM"):
    # 输出文件夹
    monthlyOutPath = output_path + "Monthly/"
    create_path(monthlyOutPath)
    dailyOutPath = output_path + "Daily/"
    create_path(dailyOutPath)
    """因子月值求和规则"""
    predictand = {}
    tand_time = open(input_path[0]).readline().split()[0]
    for tand_file in input_path:
        filename = os.path.basename(tand_file)
        print("提取预测因子数据: %s" % filename)
        variable = re.findall(r"\D+", filename)[0]  # 提取字符串非数字部分
        station = re.findall(r"\d+", filename)[0]  # 提取字符串数字部分
        variable = ["MIN" + variable, "MAX" + variable] if re.findall(temperature_label, variable) else [variable]
        data = pd.read_csv(tand_file, header=0, sep=",", names=variable)
        predictand[station] = pd.concat([predictand[station], data], axis=1,
                                        join="outer") if station in predictand.keys() else data

    print("按气象站整合预测因子数据:")
    for key in predictand.keys():
        print("     气象站: %s" % key)
        predictand[key]["Date"] = pd.date_range(tand_time, periods=predictand[key].shape[0], freq='D')
        predictand[key][predictand[key] == nan_value] = np.nan
        dailyOutput = predictand[key].set_index("Date")
        # 保存日尺度数据
        dailyOutput.to_csv(dailyOutPath + key + ".csv", index=True)
        # 月尺度升尺度
        predictand[key] = predictand[key].groupby(
            [predictand[key]["Date"].apply(lambda x: x.year), predictand[key]["Date"].apply(lambda x: x.month)]).agg(
            how2hightempor)
        predictand[key].index = ['-'.join(np.asarray(ind).astype(str)).strip() for ind in predictand[key].index.values]
        predictand[key].index = predictand[key].index.rename("Date")
        print(np.nanmax(predictand[key]["RS"].values))
        predictand[key]["RS"][predictand[key]["RS"] > 1000] = np.nan    #######删除RS异常值#########
        # 保存月尺度数据
        predictand[key].to_csv(monthlyOutPath + key + ".csv", index=True)


def read_predictand_from_csv(input_path, stations, how2hightempor, output_path, nan_value=-99, date_name="Date"):
    # 输出文件夹
    monthlyOutPath = output_path + "Monthly/"
    create_path(monthlyOutPath)
    """因子月值求和规则"""
    print("提取月尺度预测因子数据...")
    for station in stations:
        print("     气象站: %s" % station)
        data = pd.read_csv(input_path + station + ".csv")
        data[date_name] = pd.to_datetime(data[date_name]).rename("Date")
        # 剔除缺失值
        data[data == nan_value] = np.nan
        data = data.set_index("Date")
        data[data > 10000] = np.nan
        data[data < -10000] = np.nan
        data = data.reset_index()
        # 月尺度升尺度
        data = data.groupby(
            [data["Date"].apply(lambda x: x.year), data["Date"].apply(lambda x: x.month)]).agg(
            how2hightempor)
        data.index = ['-'.join(np.asarray(ind).astype(str)).strip() for ind in data.index.values]
        data.index = data.index.rename("Date")
        # 保存月尺度数据
        data.to_csv(monthlyOutPath + station + ".csv", index=True)


def read_predictor(father_path, gcm_basename, scenarios, output_path, sta_location, historical_label="historical"):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    """用xarray包读取nc4数据"""
    for basename in gcm_basename:
        for scenario in scenarios_self:
            predictor = {}
            tor_df_path = output_path + basename + "/" + scenario + "/"
            create_path(tor_df_path)
            print("提取预测因子数据: %s-%s" % (basename, scenario))
            for path in glob.glob(father_path + basename + "/" + scenario + r"/**/*.nc",
                                  recursive=True):
                if (int(re.search("(\d{6}-\d{6})", path).group().split("-")[1][:4]) < 1950) or (
                        int(re.search("(\d{6}-\d{6})", path).group().split("-")[0][:4]) > 2100):
                    continue
                try:
                    experiment, tor_name, period, tor_dict = nc_reader(nc_path=path, sta_location=sta_location)
                    for station in tor_dict.keys():
                        if station not in predictor.keys():  # 如果字典中没有气象站数据，则创建气象站的字典
                            predictor[station] = {tor_name: tor_dict[station]}
                        elif tor_name not in predictor[station].keys():  # 如果气象站字典中没有预测因子数据，则新建一个df
                            predictor[station][tor_name] = tor_dict[station]
                        else:  # 合并气象站-预测因子对应的df
                            predictor[station][tor_name] = pd.concat(
                                [predictor[station][tor_name], tor_dict[station]])
                except Exception as e:
                    with open(r"./nc_read_error", "a") as f:
                        f.write(path + " >>> Err:" + e + "\n")
                    continue
            print("按气象站整合预测因子数据:")
            for station in predictor.keys():
                print("     气象站: %s" % station)
                df = pd.concat(predictor[station].values(), axis=1)
                df.to_csv(tor_df_path + station + ".csv")


def matching_predictor(father_path, gcm_basename, predictor_name, scenarios, output_path, sta_location,
                       historical_label="historical"):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    for basename in gcm_basename:
        print("匹配预测因子: %s " % basename)
        for station in sta_location.index:
            print("\r     气象站: %s" % station, end="")
            matching_cols = predictor_name.copy()
            for scenario in scenarios_self:
                cols = pd.read_csv(father_path + basename + "/" + scenario + "/" + station + r".csv",
                                   index_col=0).dropna(axis=1, how="all").columns
                matching_cols = np.intersect1d(matching_cols, cols)
            for scenario in scenarios_self:
                matc_tor_df_path = output_path + basename + "/" + scenario + "/"
                create_path(matc_tor_df_path)
                df = pd.read_csv(father_path + basename + "/" + scenario + "/" + station + r".csv", index_col=0)[
                    matching_cols]
                df.to_csv(matc_tor_df_path + station + ".csv")
            print("\r", end="")
