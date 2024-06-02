# coding=utf-8
import glob
import numpy as np
import pandas as pd

from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import os
from osgeo import gdal, gdalconst, osr
import xarray as xr
import rasterio as rio

# 检查文件夹是否存在
def create_path(path):
    if not path.endswith('/') or path.endswith('\\'):
        fater_path = os.path.abspath(path + os.path.sep + "..")
    else:
        fater_path = path
    if not os.path.isdir(fater_path):
        os.makedirs(fater_path)


# 保留nan的求和
def sum_without_nan(df):
    if df.count() == 0:
        sum = np.nan
    else:
        sum = np.nansum(df)
    return sum


# df时间截取
def select_time(df, periods, leap=False):
    start = pd.to_datetime(periods[0])
    end = pd.to_datetime(periods[1])
    df.index = pd.to_datetime(df.index)  # 时间列格式转换
    df = df[(df.index >= start) & (df.index <= end)]
    if leap:
        df = df[~((df.index.month == 2) & (df.index.day == 29))]
    return df

def get_spatial_info(input_file: str) -> (np.ndarray, np.ndarray):
    with rio.open(input_file) as ds:
        lats = np.linspace(ds.bounds.top, ds.bounds.bottom, ds.height)
        lons = np.linspace(ds.bounds.left, ds.bounds.right, ds.width)
    return lats, lons

class RasterHandler:
    def __init__(self, outpath):
        """
        处理栅格数据
        :param outpath: 输出文件夹
        """
        self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

    def write_tiff(self, lats, lons, spatial, rename):
        # 影像的左上角和右下角坐标
        lon_min, lat_max, lon_max, lat_min = [lons.min(), lats.max(), lons.max(),
                                              lats.min()]
        # 分辨率计算
        n_lat = len(lats)
        n_lon = len(lons)
        lon_res = (lon_max - lon_min) / (float(n_lon) - 1)
        lat_res = (lat_max - lat_min) / (float(n_lat) - 1)
        # 创建.tif文件
        driver = gdal.GetDriverByName('GTiff')
        target_file = self.outpath + rename + ".tif"
        out_tif = driver.Create(target_file, n_lon, n_lat, 1, gdal.GDT_Float32)
        # 设置影像的显示范围
        out_tif.SetMetadataItem('AREA_OR_POINT', 'Point')
        geotransform = (lon_min, lon_res, 0, lat_max, 0, -lat_res)
        out_tif.SetGeoTransform(geotransform)
        # 获取地理坐标系统信息，用于选取需要的地理坐标系统
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS("WGS84")  # 定义输出的坐标系为"WGS 84"，AUTHORITY["EPSG","4326"]
        out_tif.SetProjection(srs.ExportToWkt())  # 给新建图层赋予投影信息
        # 数据写出
        out_tif.GetRasterBand(1).WriteArray(spatial)  # 将数据写入内存，此时没有写入硬盘
        out_tif.FlushCache()  # 将数据写入硬盘
        # out_tif.ComputeBandStats(True)  # 计算统计信息
        del driver


if __name__ == '__main__':
    """************************************************参数设置*******************************************************"""
    # 1. 输入数据路径
    # 降尺度数据：
    downscaling_path = r"\\ecohydrologylab.asirnas.top\FluxGroup\GCM_P\Downscaling\Monthly" + "\\"
    # 气象站站点信息：
    stationInfo = r"F:\3. Project\SuperResolution\dataFile\Station\TNR_756.csv"
    sta_location = pd.read_csv(stationInfo, usecols=["Station", "LAT", "LONG", "ELEVATION"]).set_index("Station")
    sta_location.index = sta_location.index.astype(str)
    ####################################################################################################################
    # 2. 输出数据路径
    # 输出结果文件夹
    result_path = r"\\ecohydrologylab.asirnas.top\FluxGroup\GCM_P" + "\\"
    # 插值结果输出：
    inter_path = result_path + r"Interpolation" + "\\"
    # 创建文件夹
    paths = [
        result_path, inter_path,
    ]
    for path in paths:
        create_path(path=path)
    ####################################################################################################################
    # 3. 数据处理参数
    # 气候模式
    gcm_basename = [
        # "ACCESS-CM2",
        # "CESM2-WACCM",
        # "CMCC-ESM2",
        # "FGOALS-f3-L",
        "MMEA"
    ]
    # 未来情景命名：
    scenarios = [
        # "historical",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp585",
    ]
    variables = [
        "PRE"
    ]
    # 时间升尺度方法：
    how2hightempor = {
        "PRE": sum_without_nan,
        # "RHU": np.nanmean,
        # "RS": sum_without_nan,
        # "MINTEM": np.nanmean,  # np.nanmin,
        # "MAXTEM": np.nanmean,  # np.nanmax,
        # "WIN": np.nanmean
    }
    # 插值时间段：
    periods = ["2022/1/1", "2050/12/31"]
    # 插值时间分辨率
    temporal = "multi-year" # 仅支持"mulit-year/year"/"monthly"/"daily", 其他时间尺度需单独修改156行代码
    # 插值空间分辨率
    resolution = 0.01   # 分辨率过高时会出现圆斑
    chunk_size = 100  # 内存溢出时减小,越小插值速度越慢，对效果无影响
    # 插值范围：
    boundary = [70, 55, 135, 30]  # left, top, right, bottom
    """************************************************参数设置*******************************************************"""
    # 插值信息
    data_type = "float32"
    lons = sta_location["LONG"].values.astype(data_type)
    lats = sta_location["LAT"].values.astype(data_type)
    eve = sta_location["ELEVATION"].values.astype(data_type)
    # grid_lon = np.arange(boundary[0], boundary[2], resolution).astype(data_type)
    # grid_lat = np.arange(boundary[3], boundary[1], resolution).astype(data_type)[::-1]
    # 采样范围：
    grid_lat, grid_lon = get_spatial_info(r"\\ecohydrologylab.asirnas.top\FluxGroup\TNSP_Data\AC\AC_avg_2001-2021.tif")
    grid_lat = grid_lat[(grid_lat <= boundary[1]) & (grid_lat >= boundary[3])]
    grid_lon = grid_lon[(grid_lon <= boundary[2]) & (grid_lon >= boundary[0])]

    # 提取站点数据（CMA)
    for gcm in gcm_basename:
        for ssp in scenarios:
            for var in variables:
                df = pd.DataFrame()
                for station in sta_location.index.values:
                    print(f"\rGCM: {gcm}, SSP: {ssp}, Variable: {var}, Station: {station}", end="")
                    data = pd.read_csv(f"{downscaling_path}{gcm}\\MLP\\{ssp}\\{station}.csv", index_col="Date")
                    data = select_time(data, periods).reset_index()
                    data["Date"] = pd.to_datetime(data["Date"])
                    if temporal == "multi-year":
                        data = data.groupby(data["Date"].apply(lambda x: x.year)).agg(how2hightempor).mean(axis=0)
                        data.index = data.index.rename("Date")
                        data = pd.DataFrame(data=data[var], index=["Avg"], columns=[station])
                        df = pd.concat([df, data], axis=1)
                    else:
                        if temporal == "year":
                            data = data.groupby(data["Date"].apply(lambda x: x.year)).agg(how2hightempor)

                        elif temporal == "monthly":
                            data = data.groupby(
                                [data["Date"].apply(lambda x: x.year), data["Date"].apply(lambda x: x.month)]).agg(
                                how2hightempor)
                            data.index = ['-'.join(np.asarray(ind).astype(str)).strip() for ind in data.index.values]
                        elif temporal == "daily":
                            data = data.set_index("Date")
                        else:
                            raise Exception("仅支持年、月、日尺度聚合，其他时间尺度需修改代码")
                        data.index = data.index.rename("Date")
                        df = pd.concat([df, data[var]], axis=1)
                        df = df.rename(columns={var: station})
                if df.shape[0] == 0:
                    break
                df.to_csv(f"\\\\ecohydrologylab.asirnas.top\\FluxGroup\\GCM_P\\Interpolation\\{gcm}_{ssp}_{var}.csv")
                print("\n")
                # 协克里金插值
                df = df.astype(data_type)
                for date in df.index.values:
                    inter_all = np.full([grid_lat.shape[0], grid_lon.shape[0]], np.nan)
                    for chunk in range(0, grid_lon.shape[0], chunk_size):
                        print(
                            f"\rCokrigingInterpolation >>> Date = {date} || Chunks = {chunk} : {chunk + chunk_size}"
                            f" || Total = {grid_lon.shape[0]}",
                            end="")
                        inter_var = df.loc[date]
                        # 创建 UniversalKriging 对象
                        oK = OrdinaryKriging(lons, lats, inter_var, variogram_model='spherical')
                        # 执行插值，仍内存溢出则增加参数backend="loop", n_closest_points=12，但速度较慢（12为插值点采样数量，按需调整，插值效果类似ArcGIS）
                        z, ss = oK.execute('grid', grid_lon[chunk: chunk + chunk_size], grid_lat)
                        inter_all[:, chunk: chunk + chunk_size] = z
                    inter_all[inter_all < 0] = 0
                    obj = RasterHandler(inter_path)
                    obj.write_tiff(grid_lat, grid_lon, inter_all, f"{gcm}_{ssp}_{var}_{date}")
                    print("\n")
