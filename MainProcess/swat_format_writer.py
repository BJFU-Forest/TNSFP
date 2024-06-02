# coding=utf-8
import pandas as pd
from Tools.Tool import create_path, write_raw_index


def swat_format_writer(downscaling_path, swat_path, gcm_basename, downscaling_method, scenarios, sta_location,
                       start, significant_digits):
    """输出原始数据为DAT格式"""
    header = str(start.year) + str(start.month).zfill(2) + str(start.day).zfill(2)
    for basename in gcm_basename:
        for method in downscaling_method:
            for ssp in scenarios:
                print("输出降尺度结果为SWAT文件格式: %s-%s-%s" % (basename, method, ssp))
                # 创建输出文件夹
                swat_format_path = swat_path + "/".join([basename, method, ssp]) + "/"
                create_path(swat_format_path)
                # 降尺度结果文件
                downscaling_df_path = downscaling_path + "/".join(["Daily", basename, method, ssp]) + "/"
                for station in sta_location.index.values:
                    print("\r     气象站: %s" % station, end="")
                    # 读取数据
                    df = pd.read_csv(downscaling_df_path + station + ".csv", index_col="Date")

                    # 输出SWAT格式文件
                    for k in df.columns.values:
                        sign_digit = "%." + str(significant_digits[k]) + "f"
                        if (k != "MAXTEM") & (k != "MINTEM"):
                            df[k].to_csv(swat_format_path + k + station + ".txt", sep=" ", index=False,
                                         header=[header], float_format=sign_digit)
                        else:
                            df[["MINTEM", "MAXTEM"]].to_csv(swat_format_path + "TEM" + station + ".txt", sep=",",
                                                            index=False, header=False, float_format=sign_digit)
                            write_raw_index(swat_format_path + "TEM" + station + ".txt", header)
                print("\r", end="")
