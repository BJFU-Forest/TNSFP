# coding=utf-8
from osgeo import gdal, gdalconst


def read_as_numpy(source_file, band_num=1):
    """
    读取栅格数据为numpy格式
    :param source_file: 源文件
    :param band_num: 波段序号
    :return:
    """
    dataset = gdal.Open(source_file, gdalconst.GA_ReadOnly)

    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    band_count = dataset.RasterCount

    if band_count < band_num:
        raise Exception("波段序号错误")

    band = dataset.GetRasterBand(band_num)
    data = band.ReadAsArray(0, 0, cols, rows)
    del dataset
    return data
