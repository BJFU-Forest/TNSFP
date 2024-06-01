import numpy as np
import xarray as xr

import re
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# 计算空间上两点的距离
def spatial_distance(gcm_lat, gcm_lon, sta_lat, sta_lon, data):
    """

    :param gcm_lat: GCM所有纬度坐标
    :type gcm_lat: list or np.array
    :param gcm_lon: GCM所有经度坐标
    :type gcm_lon: list or np.array
    :param sta_lat: 观测站纬度坐标
    :type sta_lat: float
    :param sta_lon: 观测站经度坐标
    :type sta_lon: float
    :param data: GCM数据
    :type data: xr.DataArray
    :return:
    """
    distance = np.sqrt(np.square([gcm_lat - sta_lat]).T + np.square([gcm_lon - sta_lon]))
    r, c = np.where(distance == distance.min())
    lat, lon = gcm_lat[r[0]], gcm_lon[c[0]]
    data_nearest = data.sel(lat=lat, lon=lon)
    while np.isnan(data_nearest.values).all():  # 略过全为空值的坐标
        distance[r, c] = np.inf
        r, c = np.where(distance == distance.min())
        lat, lon = gcm_lat[r[0]], gcm_lon[c[0]]
        data_nearest = data.sel(lat=lat, lon=lon)
    return data_nearest, lat, lon


def get_xarray_from_nc(nc_path, sta_location):
    """
    从ncCDF4文件中读取数据,基于最邻近法选取距离气象站最近的预测因子数据
    :param nc_path: ncCDF4文件路径
    :type nc_path: str
    :param sta_location: 气象站坐标 [lat, lon] or df[index: {station names}, columns: {lat, lon}]
    :type sta_location: list or pd.DataFrame
    :return: 预测因子数据 df[index: date, columns: predictor name]
    """
    upper_atmosphere_variables = ["ta", "hur", "hus", "wap", "va", "ua", "zg"]
    pressure_level = [50000, 85000]
    try:
        """nc文件数据读取"""
        ds = xr.open_dataset(nc_path)
        """获取变量数据"""
        predictor_name = ds.variable_id
        data = ds[predictor_name]
        """获取数据时间信息"""
        nc_date = ds.time.values
        if ds.frequency == "mon":
            date_boundary = re.search("(\d{6}-\d{6})", nc_path).group()
            date = pd.date_range(date_boundary.split("-")[0] + "01", periods=nc_date.shape[0], freq='MS')
        elif ds.frequency == "day":
            date_boundary = re.search("(\d{8}-\d{8})", nc_path).group()
            date = pd.date_range(date_boundary.split("-")[0], date_boundary.split("-")[1], freq='D')
            date = date[~((date.month == 2) & (date.day == 29))] if date.shape[0] > nc_date.shape[0] else date
            if date.shape[0] != nc_date.shape[0]:
                raise Exception("清理闰年数据失败，请检查nc文件的时间格式")
        else:
            raise Exception("获取数据时间范围失败，请检查nc文件的时间格式")
        """获取数据空间坐标信息"""
        if ("lat" in ds.coords) & ("lon" in ds.coords):
            lats = ds.lat.values
            lons = ds.lon.values
        elif ("latitude" in ds.coords) & ("longitude" in ds.coords):
            lats = ds.latitude.values
            lons = ds.longitude.values
        else:
            raise Exception("获取坐标数据失败，请检查nc文件的坐标格式")

        """获取数据其他信息"""
        experiment = ds.experiment_id
        """判断变量类型，并校正气压"""
        if predictor_name in upper_atmosphere_variables:
            is_upper_atmos_var = True
            gcm_pressure_level = np.asarray(
                [ds.plev.values[np.nanargmin(abs(ds.plev.values - plev_set))] for plev_set in pressure_level])
        else:
            is_upper_atmos_var = False
            gcm_pressure_level = None
        ds.close()

        """判断气象站坐标类型,并读取最近的预测因子"""
        labels = ["Date"] + ([predictor_name + str(int(plev // 100)) for plev in
                              gcm_pressure_level] if is_upper_atmos_var else [predictor_name])
        print("     情景: %s, 因子: %s, 时期: %s " % (experiment, predictor_name, date_boundary))
        if isinstance(sta_location, list):
            print("\r     提取距离气象站 %s 最近的预测因子数据..." % sta_location, end="")
            data_nearest, lat, lon = spatial_distance(lats, lons, sta_location[0], sta_location[1], data)
            predictor = pd.DataFrame(
                index=labels,
                data=[date] + ([data_nearest.sel(plev=plev).values for plev in gcm_pressure_level]
                               if is_upper_atmos_var
                               else [data_nearest.values])).T.set_index("Date")
        elif isinstance(sta_location, pd.DataFrame):
            predictor = {}
            for i, station in enumerate(sta_location.index.values):
                print("\r     提取距离气象站 %s 最近的预测因子数据..." % station, end="")
                try:
                    data_nearest, lat, lon = spatial_distance(lats, lons, sta_location.iloc[i, 0], sta_location.iloc[i, 1],
                                                              data)
                    predictor[station] = pd.DataFrame(
                        index=labels,
                        data=[date] + ([data_nearest.sel(plev=plev).values for plev in gcm_pressure_level]
                                       if is_upper_atmos_var
                                       else [data_nearest.values])).T.set_index("Date")
                except:
                    pass
            print("\r", end="")
        else:
            raise Exception("气象站坐标需要以list：[lat, lon] 或 df[index: {station names}, columns: {lat, lon}]格式输入")
        data.close()
    except Exception as e:  # 处理无法打开的nc文件 # OSError
        # raise Exception("%s 文件读取失败！请检查文件路径或nc文件完整性" % nc_path)
        raise e
    return experiment, predictor_name, date_boundary, predictor
