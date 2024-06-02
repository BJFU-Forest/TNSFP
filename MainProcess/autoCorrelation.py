import datetime
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio

from ThreadProcess.StatisticsMultiProcessing import handler as stats_cal
from ThreadProcess.linearRegressionMultiProcessing import handler as linearRegression_cal
from ThreadProcess.varYMultiProcessing import handler as varY_cal

from Tools.Tool import create_path

if __name__ == '__main__':
    #################################################数据地址################################################################
    inputPath = r"/Volumes/Focus Work/TNSP_16d_kNDVI_MODIS_2002-2018_500m/**/*.tif"
    # inputPath = r"\\ecohydrologylab.asirnas.top\FluxGroup\svodi_v01_0\1987\**\*.nc"
    outputPath = r"/Volumes/Focus Work/Result" + "/"
    create_path(outputPath)
    group = 4
    #################################################数据地址################################################################

    #################################################整合数据################################################################
    miss_value = float(input("数据缺失值为："))
     miss_value = 0.0
     val_name = None
     for i in range(group):
         data = None
         for file in glob.glob(inputPath, recursive=True):
             file_name, fileType = file.split("/")[-1].split(".")
             print("\rExtract %s data..." % file_name, end="\n")
             if fileType == "tif":
                 with rio.open(file) as ds:
                     singleImage = ds.read().astype("float32")
                     depth, width, high = singleImage.shape
                     lat = np.linspace(ds.bounds.bottom, ds.bounds.top, ds.height)
                     lon = np.linspace(ds.bounds.left, ds.bounds.right, ds.width)
                     var, year, month, day = file_name.split("_")
                     time = pd.to_datetime(
                         datetime.datetime(year=int(year), month=int(month), day=1 if int(day) == 0 else 15))
                     arr = xr.DataArray(data=singleImage[:, :, i * high // group: (i + 1) * high // group],
                                        coords=[[time], lat, lon[i * high // group: (i + 1) * high // group]],
                                        dims=["time", "lat", "lon"]).to_dataset(name=var)
                     data = xr.concat([data, arr], dim="time") if data is not None else arr
             elif (fileType == "nc") or (fileType == "nc4"):
                 val_name = input("netCDF数据的变量名为：") if val_name is None else val_name
                 with xr.open_dataset(file) as ds:
                     arr = ds[val_name]
                     arr.values = arr.values.astype("float32")
                     arr = arr.transpose("time", "lat", "lon").to_dataset(name=val_name)
                     data = xr.concat([data, arr], dim="time") if data is not None else arr
             else:
                 raise ("仅支持GeoTiff或netCDF格式文件")

         data = data.where(data != miss_value, other=np.nan)
         data = data.sortby("time")
         data.to_netcdf(outputPath + r"kNDVI_%d.nc" % i)
         data.close()
    #################################################整合数据################################################################

    ###############################################自相关系数################################################################
    for i in range(group):
        with xr.open_dataset(outputPath + r"kNDVI_%d.nc" % i) as ds:
            # val_name = input("netCDF数据的变量名为：")
            val_name = "NDVI"
            data = ds[val_name]
            data.values = data.values - np.nanmean(data)
            lats = ds["lat"]
            lons = ds["lon"]
            times = ds["time"]

        """数据分组"""
        step = 1
        window_size = 24

        """自相关系数计算"""
        alpha_nc = xr.Dataset(coords={"time": [], 'lat': lats, "lon": lons})
        residual_nc = xr.Dataset(coords={"time": [], 'lat': lats, "lon": lons})
        for i in range(times.shape[0] - window_size):
            print("\rautoCorrelation of %s [time: %s-%s]" % (val_name, times.values[i], times.values[i + window_size]),
                  end="\n")
            alpha, residual = linearRegression_cal("autoCorrelation",
                                                   data.sel(time=slice(times.values[i],
                                                                       times.values[i + window_size])).values,
                                                   data.sel(time=slice(times.values[i + step],
                                                                       times.values[i + step + window_size])).values,
                                                   process_num=30,
                                                   backup_path=r"/Volumes/Focus Work/Result/matrix/" + "/")
            alpha = xr.DataArray(alpha[np.newaxis, :], coords=[[times[i].values], lats, lons],
                                 dims=["time", "lat", "lon"]).to_dataset(name="alpha")
            residual = xr.DataArray(residual[np.newaxis, :], coords=[[times[i].values], lats, lons],
                                    dims=["time", "lat", "lon"]).to_dataset(name="residual")

            print("\rautoCorrelation of %s results are stored in NetCDF Dataset (xarray)" % val_name, end="")
            if i == 0:
                alpha_nc = alpha
                residual_nc = residual
            else:
                alpha_nc = xr.concat([alpha_nc, alpha], dim="time")
                residual_nc = xr.concat([residual_nc, residual], dim="time")
        alpha_nc.to_netcdf(outputPath + r"autoCorrelation_alpha_%d.nc" % i)
        residual_nc.to_netcdf(outputPath + r"autoCorrelation_residual_%d.nc" % i)
        alpha_nc.close()
        residual_nc.close()
    ###############################################自相关系数################################################################

    ##################################################VarY##################################################################
    for i in range(group):
        with xr.open_dataset(outputPath + r"autoCorrelation_alpha_%d.nc" % i)["alpha"] as alpha_nc:
            with xr.open_dataset(outputPath + r"autoCorrelation_residual_%d.nc" % i)["residual"] as residual_nc:
                lats = alpha_nc["lat"]
                lons = alpha_nc["lon"]
                times = alpha_nc["time"]

                var_y = varY_cal("autoCorrelation", alpha_nc.values, residual_nc.values, process_num=30,
                                 backup_path=r"/Volumes/Focus Work/Result/matrix/" + "/")

                var_y_nc = xr.DataArray(var_y, coords=[times.values, lats, lons],
                                        dims=["time", "lat", "lon"]).to_dataset(name="var_y")
                var_y_nc.to_netcdf(outputPath + r"autoCorrelation_var_y_%d.nc" % i)
                var_y_nc.close()
    ##################################################VarY##################################################################

    #################################################结果统计################################################################
    integrated = None
    for j in range(group):
        val_names = [
            "alpha",
            "var_y"
        ]
        nc_path = [
            outputPath + r"autoCorrelation_alpha_%d.nc" % j,
            outputPath + r"autoCorrelation_var_y_%d.nc" % j
        ]
        for i, val in enumerate(val_names):
            data = xr.open_dataset(nc_path[i])[val]
            lats = data["lat"]
            lons = data["lon"]
            matrix = data.values
            data.close()

            avg, cv, mk, sen, abrupt, abr_sign, mk_b, mk_a = stats_cal("stats_" + val, matrix, process_num=30,
                                                                       backup_path=r"/Volumes/Focus Work/Result/matrix/" + "/")

            nc = xr.Dataset(coords={"lon": lons, 'lat': lats})
            back_up_name = ["avg", "cv", "mk", "sen", "abrupt", "abr_sign", "mk_b", "mk_a"]
            back_up_matrix = [avg, cv, mk, sen, abrupt, abr_sign, mk_b, mk_a]
            for j in range(len(back_up_name)):
                print("\r%s_%s results are stored in NetCDF Dataset (xarray)" % ("stats_autoC", back_up_name[j]), end="")
                nc[back_up_name[j]] = xr.DataArray(back_up_matrix[j], coords=[lats, lons], dims=["lat", "lon"])
            nc.to_netcdf(outputPath + r"Statistics_%s_%d.nc" % (val, j))
            nc.close()
            integrated = xr.concat([integrated, nc], dim="lon") if integrated is not None else nc
    integrated.close()
    ##################################################结果统计###############################################################
