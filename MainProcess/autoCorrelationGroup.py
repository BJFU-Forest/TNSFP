from datetime import datetime
import glob
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
from natsort import natsorted  # 文件列表按数字顺序排序库，需安装，命令行输入：pip install natsort

from ThreadProcess.StatisticsMultiProcessing import handler as stats_cal
from ThreadProcess.linearRegressionMultiProcessing import handler as linearRegression_cal
from ThreadProcess.varYMultiProcessing import handler as varY_cal
from ThreadProcess.DLMMultiProcessing import handler as dlm_cal
from ThreadProcess.NDVI_change_test import handler as ndvi_change_test
from ThreadProcess.Pearsonr_MP import handler as pearson_cal
from ThreadProcess.RemoveLongtermTrend import handler as remove_trend
from ThreadProcess.DeltaVariableCalculate_MP import handler as remove_trend_lr


from Tools.Tool import create_path
import os
import time

from multiprocessing import Pool, Process, Manager
import scipy.stats as st
from math import sqrt



if __name__ == '__main__':
    #################################################数据地址################################################################
    inputPath = r"/Volumes/FluxGroup/TNSP_Data/TNSP_16d_kNDVI_MODIS_2001-2021_500m/**/*.tif"
    luccPath = r"/Volumes/FluxGroup/TNSP_Data/TNSP_YearlyLandUse_MODIS_2001-2021_500m/"
    outputPath = r"/Volumes/home/HDD/AC-Regression-lr-diffAvg/"
    create_path(outputPath)
    group = 8  # 分组处理（内存不足时增大，默认为1）
    aggregated = 1  # 时间聚合（处理时间过长时增大，代表tif文件数，同时可减少内存占用，默认为1，*损失时间精度*,*67行修改聚合方法*）
    #################################################数据地址################################################################

    #################################################整合数据################################################################
    # miss_value = float(input("数据缺失值为："))
    # miss_value = 0.0
    # val_name = None
    # file_list = natsorted(glob.glob(inputPath, recursive=True))  # 文件列表按数字顺序排序，确认MacOS是否适用
    # for i in range(group):
    # # for i in [7]:
    #     data = None
    #     lat = None
    #     lon = None
    #     for j in range(len(file_list) // aggregated):
    #         agg_data = None
    #         for file in file_list[j * aggregated: (j + 1) * aggregated]:
    #             file_name, fileType = file.split("/")[-1].split(".")
    #             print("\rExtract %s data..." % file_name, end="\n")
    #             if fileType == "tif":
    #                 with rio.open(file) as ds:
    #                     ###################################读取TIF数据######################################################
    #                     singleImage = ds.read().astype("float32")
    #                     lat = np.linspace(ds.bounds.top, ds.bounds.bottom, ds.height).astype("float32") if lat is None else lat
    #                     lon = np.linspace(ds.bounds.left, ds.bounds.right, ds.width).astype("float32") if lon is None else lon
    #                     var, year, month, day = file_name.split("_")
    #                     ###################################读取TIF数据######################################################
    #
    #                     #################################筛选植被区域######################################################
    #                     lucc = rio.open(luccPath + "LandUse%s.tif" % year).read().astype("float32")
    #                     #if year == "2018":
    #                     #    lucc = lucc[:, ::-1,:]
    #                     vegetation_type = [1, 2, 3, 4, 5, 6, 10]
    #                     singleImage[~np.isin(lucc, vegetation_type)] = np.nan
    #                     ###################################筛选植被区域######################################################
    #                     time = pd.to_datetime(
    #                         (year=int(year), month=int(month), day=1 if int(day) == 0 else 15))
    #                     arr = xr.DataArray(
    #                         data=singleImage[:, :, i * ds.width // group: (i + 1) * ds.width // group],
    #                         coords=[[time], lat, lon[i * ds.width // group: (i + 1) * ds.width // group]],
    #                         dims=["time", "lat", "lon"]).to_dataset(name=var)
    #                     agg_data = xr.concat([agg_data, arr], dim="time") if agg_data is not None else arr
    #                     arr.close()
    #             elif (fileType == "nc") or (fileType == "nc4"):
    #                 val_name = input("netCDF数据的变量名为：") if val_name is None else val_name
    #                 with xr.open_dataset(file) as ds:
    #                     arr = ds[val_name]
    #                     arr.values = arr.values.astype("float32")
    #                     arr = arr.transpose("time", "lat", "lon").to_dataset(name=val_name)
    #                     agg_data = xr.concat([agg_data, arr], dim="time") if agg_data is not None else arr
    #                     arr.close()
    #             else:
    #                 raise ("仅支持GeoTiff或netCDF格式文件")
    #         agg_data = agg_data.mean(dim='time') if aggregated != 1 else agg_data  # 按time轴聚合，mean可替换为其他聚合形式
    #         data = xr.concat([data, agg_data], dim="time") if data is not None else agg_data
    #         agg_data.close()
    #     data = data.where(data != miss_value, other=np.nan)
    #     data["lat"].attrs = {"units": "degrees_north", "axis": "Y"}
    #     data["lon"].attrs = {"units": "degrees_east", "axis": "X"}
    #     data.to_netcdf(outputPath + r"kNDVI_%d.nc" % i)
    #     data.close()
    ################################################整合数据################################################################

    # ################################################去趋势stl################################################################
    # # 打印当前系统时间
    # print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # # 参数预处理
    # """创建输出文件夹"""
    # var_name = "NDVI"
    # for i in range(group):
    #     outFilePath = os.path.join(outputPath, f"remove_trend_ndvi_{i}.nc")
    #     print(f"Load data: group {i}")
    #     # 基于chunk读取数据
    #     ndvi = xr.open_dataset(outputPath + f"kNDVI_{i}.nc")["NDVI"]
    #     """获取时空坐标信息&数据"""
    #     lat = ndvi.lat.values
    #     lon = ndvi.lon.values
    #     time = ndvi.time.values
    #     matrix = ndvi.values
    #     ndvi.close()
    #     """使用 pool.apply_async并行运算SPEI"""
    #     print(f"Calculate {var_name} ... ")
    #     kwargs = {"period": 23, "smooth_length": 7}
    #     """
    #     period : {int, None}, optional ### 一年的数据量 ###
    #         Periodicity of the sequence. If None and endog is a pandas Series or
    #         DataFrame, attempts to determine from endog. If endog is a ndarray,
    #         period must be provided.
    #     smooth_length : int, optional
    #         Length of the seasonal smoother. Must be an odd integer, and should
    #         normally be >= 7 (default).
    #     """
    #     delta_ndvi = remove_trend(parameter_name=f"remove_trend/{i}",
    #                               matrix=matrix, process_num=None,
    #                               backup_path=r"/Volumes/home/HDD/backup/" + "/",
    #                               **kwargs)
    #     # 释放matrix所占用的内存
    #     matrix = None
    #     # 导出netcdf文件
    #     """构建nc_Dataset"""
    #     result = xr.DataArray(data=delta_ndvi, coords=[time, lat, lon], dims=["time", "lat", "lon"]).to_dataset(
    #         name=var_name)
    #     # 释放无用数据所占用的内存
    #     delta_ndvi, lat, lon, time = None, None, None, None
    #     """写入nc文件属性"""
    #     result.coords["lat"].attrs["units"] = "degrees_north"
    #     result.coords["lon"].attrs["units"] = "degrees_east"
    #     """导出为netCDF文件"""
    #     print(f"Export to: {outFilePath}")
    #     result.to_netcdf(path=outFilePath)
    #     result.close()
    #     print(
    #         f"Finish process >>>>> calculate {var_name} & export to: {outFilePath} ||"
    #         f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    #     )
    #     print("\n")
    # ################################################去趋势stl################################################################

    ################################################去趋势linear################################################################
    # # 打印当前系统时间
    # print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # # 参数预处理
    # """创建输出文件夹"""
    # var_name = "NDVI"
    # for i in range(group):
    #     outFilePath = os.path.join(outputPath, f"remove_trend_ndvi_{i}.nc")
    #     print(f"Load data: group {i}")
    #     # 基于chunk读取数据
    #     ndvi = xr.open_dataset(outputPath + f"kNDVI_{i}.nc")["NDVI"]
    #     """获取时空坐标信息&数据"""
    #     lat = ndvi.lat.values
    #     lon = ndvi.lon.values
    #     time = ndvi.time.values
    #     matrix = ndvi.values
    #     ndvi.close()
    #     """提取dayofyear数据"""
    #     # 将日期转换为 pandas 的 datetime 对象，年份统一设为一个非闰年（如2001年）
    #     time_series = pd.Series(pd.to_datetime(time))
    #     dates = pd.to_datetime({'year': 2001, 'month': time_series.dt.month, 'day': time_series.dt.day})
    #     dayofyear = dates.dt.dayofyear.values
    #     """使用 pool.apply_async并行运算SPEI"""
    #     print(f"Calculate {var_name} ... ")
    #     kwargs = {"dayofyear": dayofyear}
    #     delta_ndvi = remove_trend_lr(parameter_name=f"remove_trend/{i}",
    #                                   matrix=matrix, process_num=None,
    #                                   backup_path=r"/Volumes/home/HDD/backup/" + "/",
    #                                   **kwargs)
    #     # 释放matrix所占用的内存
    #     matrix = None
    #     # 导出netcdf文件
    #     """构建nc_Dataset"""
    #     result = xr.DataArray(data=delta_ndvi, coords=[time, lat, lon], dims=["time", "lat", "lon"]).to_dataset(
    #         name=var_name)
    #     # 释放无用数据所占用的内存
    #     delta_ndvi, lat, lon, time = None, None, None, None
    #     """写入nc文件属性"""
    #     result.coords["lat"].attrs["units"] = "degrees_north"
    #     result.coords["lon"].attrs["units"] = "degrees_east"
    #     """导出为netCDF文件"""
    #     print(f"Export to: {outFilePath}")
    #     result.to_netcdf(path=outFilePath)
    #     result.close()
    #     print(
    #         f"Finish process >>>>> calculate {var_name} & export to: {outFilePath} ||"
    #         f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    #     )
    #     print("\n")
    ################################################去趋势linear################################################################

    #############################################自相关系数################################################################
    # for j in range(group):
    # # for j in [3]:
    # #     with xr.open_dataset(outputPath + r"kNDVI_%d.nc" % j) as ds:                      # 未去趋势
    #     with xr.open_dataset(os.path.join(outputPath, f"remove_trend_ndvi_{j}.nc")) as ds:  # 去趋势
    #         # val_name = input("netCDF数据的变量名为：")
    #         val_name = "NDVI"
    #         data = ds[val_name]
    #         # data.values = data.values - np.nanmean(data)
    #         lats = ds["lat"]
    #         lons = ds["lon"]
    #         times = ds["time"]
    #
    #     """数据分组"""
    #     step = 1
    #     window_size = 60
    #
    #     """自相关系数计算"""
    #     alpha_nc = xr.Dataset(coords={"time": [], 'lat': lats, "lon": lons})
    #     residual_nc = xr.Dataset(coords={"time": [], 'lat': lats, "lon": lons})
    #     for i in range(times.shape[0] - window_size-step):
    #         print("\rautoCorrelation of %s [time: %s-%s]" % (val_name, times.values[i], times.values[i + window_size]),
    #               end="\n")
    #         alpha, residual = linearRegression_cal("autoCorrelation",
    #                                                data.sel(time=slice(times.values[i],
    #                                                                    times.values[i + window_size])).values,
    #                                                data.sel(time=slice(times.values[i + step],
    #                                                                    times.values[i + step + window_size])).values,
    #                                                process_num=None,
    #                                                backup_path=r"/Volumes/home/HDD/backup/" + "/")
    #         alpha = xr.DataArray(alpha[np.newaxis, :], coords=[[times[i].values], lats, lons],
    #                              dims=["time", "lat", "lon"]).to_dataset(name="alpha")
    #         residual = xr.DataArray(residual[np.newaxis, :], coords=[[times[i].values], lats, lons],
    #                                 dims=["time", "lat", "lon"]).to_dataset(name="residual")
    #
    #         print("\rautoCorrelation of %s results are stored in NetCDF Dataset (xarray)" % val_name, end="")
    #         if i == 0:
    #             alpha_nc = alpha
    #             residual_nc = residual
    #         else:
    #             alpha_nc = xr.concat([alpha_nc, alpha], dim="time")
    #             residual_nc = xr.concat([residual_nc, residual], dim="time")
    #     alpha_nc.to_netcdf(outputPath + r"autoCorrelation_alpha_%d.nc" % j)
    #     residual_nc.to_netcdf(outputPath + r"autoCorrelation_residual_%d.nc" % j)
    #     alpha_nc.close()
    #     residual_nc.close()
    ##############################################自相关系数################################################################


    ##############################################自相关系数ByDLM################################################################
    # for j in range(group):
    #     # with xr.open_dataset(outputPath + r"kNDVI_%d.nc" % j) as ds:  # 未去趋势
    #     with xr.open_dataset(os.path.join(outputPath, f"remove_trend_ndvi_{j}.nc")) as ds:  # 去趋势
    #         # val_name = input("netCDF数据的变量名为：")
    #         val_name = "NDVI"
    #         data = ds[val_name]
    #         # matrix = data.values - np.nanmean(data)
    #         lats = ds["lat"]
    #         lons = ds["lon"]
    #         times = ds["time"]
    #
    #     """数据分组"""
    #     step = 1
    #
    #     """自相关系数计算"""
    #     alpha_nc = xr.Dataset(coords={"time": [], 'lat': lats, "lon": lons})
    #     variance_nc = xr.Dataset(coords={"time": [], 'lat': lats, "lon": lons})
    #     print("\rautoCorrelation of %s [Group: %s time: %s - %s]" % (
    #         val_name, j + 1, times.values[0], times.values[-1]), end="\n")
    #     alpha, variance = dlm_cal("autoCorrelation",
    #                               matrix_x=matrix[:-step, :, :],
    #                               matrix_y=matrix[step:, :, :],
    #                               process_num=None,
    #                               backup_path=r"/Volumes/Focus Work/backup" + "/")
    #     alpha_nc = xr.DataArray(alpha, coords=[times[:-1], lats, lons],
    #                          dims=["time", "lat", "lon"]).to_dataset(name="alpha")
    #     variance_nc = xr.DataArray(variance, coords=[times[:-1], lats, lons],
    #                             dims=["time", "lat", "lon"]).to_dataset(name="variance")
    #     alpha_nc.to_netcdf(outputPath + r"autoCorrelation_alpha_%d.nc" % j)
    #     variance_nc.to_netcdf(outputPath + r"autoCorrelation_variance_%d.nc" % j)
    #     alpha_nc.close()
    #     variance_nc.close()
    ###############################################自相关系数ByDLM################################################################


    ################################################VarY##################################################################
    # for i in range(group):
    #     with xr.open_dataset(outputPath + r"autoCorrelation_alpha_%d.nc" % i)["alpha"] as alpha_nc:
    #         with xr.open_dataset(outputPath + r"autoCorrelation_residual_%d.nc" % i)["residual"] as variance_nc:
    #             lats = alpha_nc["lat"]
    #             lons = alpha_nc["lon"]
    #             times = alpha_nc["time"]
    #             print("计算varY：")
    #             var_y = varY_cal("autoCorrelation", alpha_nc.values, variance_nc.values, process_num=None,
    #                              backup_path=r"/Volumes/home/HDD/backup/" + "/")
    #
    #             var_y_nc = xr.DataArray(var_y, coords=[times.values, lats, lons],
    #                                     dims=["time", "lat", "lon"]).to_dataset(name="var_y")
    #             var_y_nc.to_netcdf(outputPath + r"autoCorrelation_var_y_%d.nc" % i)
    #             var_y_nc.close()
    #################################################VarY##################################################################

    #################################################结果统计################################################################
    # val_names = [
    #     # "alpha",
    #     "var_y",
    #     # "variance"
    # ]
    # print("时空动态统计：")
    # for i, val in enumerate(val_names):
    #     for j in range(group):
    #         print("     Value: %s, Group: %s >>>>> 读取 " % (val, j), end="")
    #         data = xr.open_dataset(outputPath + r"autoCorrelation_%s_%d.nc" % (val, j))[val]
    #         lats = data["lat"].values
    #         lons = data["lon"].values
    #         matrix = data.values
    #         data.close()
    #         ##########################################统计##############################################################
    #         print(">>>>> 统计 ", end="\n")
    #         avg, mk, sen = stats_cal("stats_" + val, matrix[69:, :, :], process_num=None,
    #                                  backup_path=r"/Volumes/home/HDD/backup/" + "/")
    #         stats = xr.Dataset(coords={"lon": lons, 'lat': lats})
    #         back_up_name = ["avg", "sen"]
    #         back_up_matrix = [avg, sen]
    #         print(">>>>> 保存 ", end="")
    #         for m in range(len(back_up_name)):
    #             # print("        %s_%s results are stored in NetCDF Dataset (xarray)" % ("stats_autoC", back_up_name[m]))
    #             stats[val + "_" + back_up_name[m]] = xr.DataArray(back_up_matrix[m], coords=[lats, lons],
    #                                                               dims=["lat", "lon"])
    #         data.coords["lat"].attrs["units"] = "degrees_north"
    #         data.coords["lon"].attrs["units"] = "degrees_east"
    #         stats.to_netcdf(outputPath + r"Statistics_%s_%d.nc" % (val, j))
    #         print(">>>>> 完成")
    #         ##########################################统计##############################################################
    #     print(">>>>> 整合")
    #     integrated = None
    #     for j in range(group):
    #         print(">>>>> Value: %s | Group: %s" % (val, j))
    #         stats = xr.open_dataset(outputPath + r"Statistics_%s_%d.nc" % (val, j))
    #         integrated = xr.concat([integrated, stats], dim="lon") if integrated is not None else stats
    #     data.coords["lat"].attrs["units"] = "degrees_north"
    #     data.coords["lon"].attrs["units"] = "degrees_east"
    #     integrated.to_netcdf(outputPath + r"Statistics_%s_all.nc" % val)
    #     integrated.close()
    ##################################################结果统计###############################################################

    ################################################相关性计算################################################################
    # variable = ["alpha","var_y"]
    # print("Pearson相关性统计：")
    # for var in variable:
    #     for i in range(group):
    #         print("     Group: %s >>>>> 读取 " % i, end="")
    #         alpha = xr.open_dataset(outputPath + r"autoCorrelation_%s_%d.nc" % (var, i))[var]
    #         npp = xr.open_dataset(outputPath + r"NPP_%d.nc" % i)["NPP"] # 确认前边NPP的nc文件名和变量名
    #         alpha = alpha.groupby("time.year").mean()
    #         npp = npp.groupby("time.year").sum()
    #         lats = alpha["lat"]
    #         lons = alpha["lon"]
    #         alphaM = alpha.values
    #         nppM = npp.values
    #         alpha.close()
    #         npp.close()
    #         #########################################统计##############################################################
    #         print(">>>>> 统计 ", end="\n")
    #         r, p = pearson_cal("stats_PearsonC", alphaM[3:, :, :], nppM[5:, :, :], process_num=None,
    #                            backup_path=r"/Volumes/home/HDD/backup/" + "/")
    #         pearsonR = xr.Dataset(coords={"lon": lons, 'lat': lats})
    #         back_up_name = ["r", "p"]
    #         back_up_matrix = [r, p]
    #         print(">>>>> 保存 ", end="")
    #         for m in range(len(back_up_name)):
    #             pearsonR[back_up_name[m]] = xr.DataArray(back_up_matrix[m], coords=[lats, lons], dims=["lat", "lon"])
    #         # data.coords["lat"].attrs["units"] = "degrees_north"
    #         # data.coords["lon"].attrs["units"] = "degrees_east"
    #         pearsonR.to_netcdf(outputPath + r"PearsonR_%s_%d.nc" % (var, i))
    #         print(">>>>> 完成")
    #         ##########################################统计##############################################################
    #     print(">>>>> 整合")
    #     integrated = None
    #     for i in range(group):
    #         print(">>>>> Group: %s" % i)
    #         pearsonR = xr.open_dataset(outputPath + r"PearsonR_%s_%d.nc" % (var, i))
    #         integrated = xr.concat([integrated, pearsonR], dim="lon") if integrated is not None else pearsonR
    #     # data.coords["lat"].attrs["units"] = "degrees_north"
    #     # data.coords["lon"].attrs["units"] = "degrees_east"
    #     integrated.to_netcdf(outputPath + r"PearsonR_%s_all.nc"%var)
    #     integrated.close()
        ################################################相关性计算################################################################



    ###############################################突变检测################################################################
    ndvi_change = None
    for j in range(group):
        ndvi = xr.open_dataset(outputPath + r"kNDVI_%s.nc" % j)["NDVI"]
        ac = xr.open_dataset(outputPath + r"autoCorrelation_alpha_%s.nc" % j)["alpha"]
        ac_year = ac.groupby("time.year").mean()
        ndvi_year = ndvi.groupby("time.year").mean()
        lats = ndvi_year["lat"].values
        lons = ndvi_year["lon"].values
        date = ndvi_year["year"].values
        trigger_index = np.argmax(date >= 2002)
        matrix_ndvi = ndvi_year.values.astype(np.float32)
        matrix_ac = ac_year.values.astype(np.float32)
        ndvi.close()
        ac.close()
        state, min_ac = ndvi_change_test("kNDVI_change", matrix_ndvi[5:, :, :], matrix_ac[3:, :, :], trigger_index, process_num=None,
                                         backup_path=r"/Volumes/home/HDD/backup/")
        nc = xr.Dataset(coords={"lon": lons, 'lat': lats})
        nc["state"] = xr.DataArray(state, coords=[lats, lons], dims=["lat", "lon"])
        nc["min_ac"] = xr.DataArray(min_ac, coords=[lats, lons], dims=["lat", "lon"])
        nc.to_netcdf(outputPath + r"NDVI_Change_%d.nc" % j)
        nc.close()
    ######################################整合##########################################################################
    print(">>>>> 整合")
    ndvi_change = None
    for j in range(group):
        print(">>>>> Group: %s" % j)
        data = xr.open_dataset(outputPath + r"NDVI_Change_%d.nc" % j)
        ndvi_change = xr.concat([ndvi_change, data], dim="lon") if ndvi_change is not None else data
        data.close()
    ndvi_change.to_netcdf(outputPath + r"NDVI_Change_all.nc")
    ndvi_change.close()
    ##################################################突变检测###############################################################
