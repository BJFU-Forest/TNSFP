# coding=utf-8
import sys

sys.path.append(r"F:\Future climate and hydrology")

import glob
import pandas as pd
import os
import re
import time
from Tools.Tool import create_path
from DataReader.read_nc_by_xarray import get_xarray_from_nc as nc_reader
from Tools.DateHandler import dateHandler

os.system('')


def get_all_path(rootdir, type=None):
    dir_list = []
    file_list = []
    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if (type is None) or (file.endswith(type)):
                dir_list.append(root)
                file_list.append(os.path.join(root, file))
    dir_list = list(set(dir_list))
    return dir_list, file_list


def get_info(father_path, output_path):
    dir_list, file_list = get_all_path(father_path, type=".nc")
    for dir_father in dir_list:
        # glob检索路径
        nc_glob_path = dir_father + r"\*.nc"
        # 输出路径
        info_csv_path = output_path + "\\" + "\\".join(dir_father.split("\\")[-2:]) + "\\info.csv"
        create_path(info_csv_path)
        # info.csv文件
        info_dict = {
            "Variable": [],
            "TableID": [],
            "Period": [],

            "SourceID": [],
            "ExperimentID": [],
            "VariantLabel": [],
            "GridLabel": [],
            "FilePath": [],
        }

        for nc_path in glob.glob(nc_glob_path):
            nc_name = os.path.basename(nc_path)
            if (int(re.search("(\d{6}-\d{6})", nc_name).group().split("-")[1][:4]) < 1950) or (
                    int(re.search("(\d{6}-\d{6})", nc_name).group().split("-")[1][:4]) > 2100):
                continue
            variable, tableID, sourceID, experimentID, variantLabel, gridLabel, period = nc_name.split("_")
            print("提取文件名信息: %s-%s-%s" % (sourceID, experimentID, variable))
            if (tableID == "day") or (tableID == "Amon"):
                period = period.split(".")[0]
                info_dict["Variable"].append(variable)
                info_dict["TableID"].append(tableID)
                info_dict["Period"].append(period)

                info_dict["SourceID"].append(sourceID)
                info_dict["ExperimentID"].append(experimentID)
                info_dict["VariantLabel"].append(variantLabel)
                info_dict["GridLabel"].append(gridLabel)
                info_dict["FilePath"].append(nc_path)

        info = pd.DataFrame(info_dict)
        info = info.sort_values(by=["SourceID", "TableID", "ExperimentID", "VariantLabel", "GridLabel",
                                    "Variable", "Period"])
        info.to_csv(info_csv_path, index=False)


def file_check(father_path, gcm_basename, scenarios, output_path, sta_location, historical_label="historical"):
    scenarios_self = scenarios.copy()
    scenarios_self.insert(0, historical_label)

    """用xarray包读取nc4数据"""
    for basename in gcm_basename:
        for scenario in scenarios_self:
            log_path = output_path + "\\" + basename + "\\" + scenario + "\\"
            create_path(log_path)
            print("校验文件可用性: %s-%s" % (basename, scenario))
            for path in glob.glob(father_path + "\\" + basename + "\\" + scenario + r"\**\*.nc",
                                  recursive=True):
                start_time = time.time()
                if (int(re.search("(\d{6}-\d{6})", path).group().split("-")[1][:4]) < 1950) or (
                        int(re.search("(\d{6}-\d{6})", path).group().split("-")[1][:4]) > 2100):
                    continue
                try:
                    experiment, tor_name, period, tor_dict = nc_reader(nc_path=path, sta_location=sta_location)
                    now_time = time.time()
                    print('\033[1A', end='')  # cursor up 5 lines
                    print('\033[0J', end='')  # erase from cursor to end
                    print("     校验文件可用性成功 (experiment = %s, tor_name = %s, period = %s), 用时：%s (%s)" % (
                        experiment, tor_name, period, dateHandler.uptimereport(start_time, now_time), basename))
                except:
                    nc_name = os.path.basename(path)
                    print("\033[31m     校验文件可用性失败 (文件名：%s)\033[0m" % nc_name)
                    with open(log_path + r".\error_log.txt", "a") as f:
                        f.write(nc_name + "\n")
                    continue


if __name__ == "__main__":
    ####################################################################################################################
    # 1. 输入数据路径
    # 原始日尺度观测数据：
    gauge_path = glob.glob(r"..\dataFile\Climate\*.txt")
    # 气象站站点信息：
    stationInfo = r"..\dataFile\Station\气象站.csv"
    sta_location = pd.read_csv(stationInfo, usecols=["Station", "LAT", "LONG", "ELEVATION"]).set_index("Station")
    sta_location.index = sta_location.index.astype(str)
    # 流域边界矢量：
    boundary = r"..\dataFile\Boundary\Jlboundary.shp"
    # 原始预测因子 / GCM父文件夹：
    gcm_father_path = r"\\ecohydrologylab.asirnas.top\Hydrological Group\1. IPCC\Data\Daily"
    # GCM命名：
    gcm_basename = [
        "ACCESS-CM2",
        "ACCESS-ESM1-5",
        # "AWI-CM-1-1-MR",
        "CanESM5",
        # "CESM2-WACCM",
        # "CMCC-CM2-SR5",
        "CMCC-ESM2",
        # "CNRM-ESM2-1",
        "EC-Earth3",
        # "EC-Earth3-Veg",
        # "EC-Earth3-Veg-LR",
        # "FGOALS-f3-L",
        "FGOALS-g3",
        # "GFDL-ESM4",
        "INM-CM4-8",
        "INM-CM-5-0",
        "IPSL-CM6A-LR",
        # "KACE-1-0-G",
        "MIROC6",
        "MPI-ESM1-2-HR",
        "MRI-ESM2-0",
    ]
    # 历史情景命名：
    historical = "historical"
    # 未来情景命名：
    scenarios = [
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    ]
    ####################################################################################################################
    # 2. 输出数据路径
    outPath = r"G:\IPCC"
    ####################################################################################################################
    # 3. 主进程
    # 3.1. info提取
    get_info(father_path=gcm_father_path, output_path=outPath)

    # 3.2. 文件校验
    file_check(father_path=gcm_father_path, scenarios=scenarios, gcm_basename=gcm_basename,
               sta_location=sta_location, output_path=outPath, historical_label=historical)
