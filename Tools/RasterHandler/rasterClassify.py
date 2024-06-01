# coding=utf-8
from RasterHandler import rasterHandler, rasterReader
import os


def get_classify_ndarray(classify_file, cols, rows):
    """
    读取ndarray格式的分类信息
    :param classify_file: 原始分类栅格位置
    :param cols: 数据的列数
    :param rows: 数据的行数
    :return:
    """
    Ras_handler = rasterHandler.RasterHandler("./")
    Ras_handler.resampling(classify_file, "classify", cols=cols, rows=rows)
    classify = rasterReader.read_as_numpy(r".\classify.tif")
    os.remove(r".\classify.tif")
    return classify
