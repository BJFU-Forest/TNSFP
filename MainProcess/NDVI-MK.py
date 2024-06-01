import os
import numpy as np
from osgeo import gdal
import pymannkendall as mk
from gdalconst import *


def mk_test(tif_list):
    sample = gdal.Open(tif_list[0])
    band = sample.GetRasterBand(1)
    rows, cols = band.YSize, band.XSize
    geo_transform = sample.GetGeoTransform()
    projection = sample.GetProjection()
    driver = gdal.GetDriverByName('GTiff')

    z_output = driver.Create('z.tif', cols, rows, 1, GDT_Float64)
    z_output.SetGeoTransform(geo_transform)
    z_output.SetProjection(projection)

    tau_output = driver.Create('tau.tif', cols, rows, 1, GDT_Float64)
    tau_output.SetGeoTransform(geo_transform)
    tau_output.SetProjection(projection)

    z_band = z_output.GetRasterBand(1)
    tau_band = tau_output.GetRasterBand(1)

    for i in range(rows):
        z_row = []
        tau_row = []
        for j in range(cols):
            cell_values = []
            for tif in tif_list:
                dataset = gdal.Open(tif)
                band = dataset.GetRasterBand(1)
                cell_values.append(band.ReadAsArray(j, i, 1, 1)[0][0])
            result = mk.original_test(cell_values)
            z_row.append(result.z)
            tau_row.append(result.Tau)
        z_band.WriteArray(np.array([z_row]), 0, i)
        tau_band.WriteArray(np.array([tau_row]), 0, i)

    z_output.FlushCache()
    tau_output.FlushCache()

if __name__ == '__main__':
    dir_path = '/Volumes/FluxGroup/TNSP_Data/TNSP_16d_kNDVI_MODIS_2001-2021_500m'
    tif_files = sorted([os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith('.tif')])
    mk_test(tif_files)