# coding=utf-8
from osgeo import gdal, gdalconst, osr
import os


class RasterHandler:
    def __init__(self, outpath):
        """
        处理栅格数据
        :param outpath: 输出文件夹
        """
        self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

    def resampling(self, source_file, rename, resolution=1.0, refer_raster=None, cols=None, rows=None):
        """
        影像重采样
        :param source_file: 源文件
        :param rename: 输出文件名
        :param resolution: 像元大小(m)
        :param refer_raster: 参考栅格(重采样为与参考栅格相同分辨率)
        :param cols: 列数
        :param rows: 行数
        :return:
        """
        dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)
        band_count = dataset.RasterCount  # 波段数

        if refer_raster is not None:
            refer = gdal.Open(refer_raster, gdalconst.GA_ReadOnly)
            resolution = refer.GetGeoTransform()[1]

        if band_count == 0:
            raise Exception("输入栅格为空")
        elif resolution <= 0 and refer_raster is None and cols is None and rows is None:
            raise Exception("空间分辨率不能为负")

        geotrans = list(dataset.GetGeoTransform())

        if cols is None and rows is None:
            ori_res_x = geotrans[1]
            ori_res_y = -geotrans[5]
            geotrans[1] = resolution  # 修改x轴分辨率
            geotrans[5] = -resolution  # 修改y轴分辨率(由于Gdal读取数据是倒置，所以y轴分辨率为负数)

            scale_x = ori_res_x / resolution
            scale_y = ori_res_y / resolution
            cols = dataset.RasterXSize  # 列数
            rows = dataset.RasterYSize  # 行数
            cols = int(cols * scale_x)  # 计算新的行列数
            rows = int(rows * scale_y)

        target_file = self.outpath + rename + ".tif"
        if os.path.exists(target_file) and os.path.isfile(target_file):  # 如果已存在同名影像
            os.remove(target_file)  # 则删除之

        band1 = dataset.GetRasterBand(1)
        data_type = band1.DataType
        target = dataset.GetDriver().Create(target_file, xsize=cols, ysize=rows, bands=band_count,
                                            eType=data_type)
        target.SetProjection(dataset.GetProjection())  # 设置投影坐标
        target.SetGeoTransform(geotrans)  # 设置地理变换参数
        total = band_count + 1
        for index in range(1, total):
            # 读取波段数据
            data = dataset.GetRasterBand(index).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
            out_band = target.GetRasterBand(index)
            out_band.SetNoDataValue(dataset.GetRasterBand(index).GetNoDataValue())
            out_band.WriteArray(data)  # 写入数据到新影像中
            out_band.FlushCache()
            out_band.ComputeBandStats(False)  # 计算统计信息
        del dataset
        del target

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


if __name__ == "__main__":
    source = r"G:\data\Climate\KoppenClassify\world_koppen.tif"
    outpath = r"D:\Work\气候分析\statistics" + "\\"
    obj = RasterHandler(outpath)
    data = obj.resampling(source, "test", rows=1800, cols=900)
