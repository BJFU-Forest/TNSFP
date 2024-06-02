import os.path
from typing import Dict
import numpy as np
import xarray as xr
import rasterio as rio
from Tools.Tool import create_path
from ThreadProcess.VCC_MultiProcessing import handler as vcc_cal
from  CokrigingInterpolation import RasterHandler

if __name__ == '__main__':
    #################################################数据地址################################################################
    hisAcPath = r"/Users/hx/Desktop/12th/Data/LF_alpha_avg_nochange2.tif"
    hisNPPPath = r"/Volumes/FluxGroup/TNSP_Data/TNSP_YearlyNPPsum_MODIS_2001-2021_500m/NPPsum2021.tif"
    hisLUCPath = r"/Users/hx/Desktop/12th/Geotif影像/LandUseNoChange.tif"
    hisPPath = r"/Volumes/FluxGroup/TNSP_Data/AC/P_avg_2001-2021.tif"
    hisETPath = r"/Volumes/FluxGroup/TNSP_Data/AC/ET2001-2021.tif"
    futurePPath = r"/Volumes/FluxGroup/GCM_P/Interpolation" + "/"
    additionPPath = r"/Users/hx/Desktop/12th/Geotif影像/AdditionWaterWest.tif"
    outputPath = r"/Volumes/home/HDD/AC-Regression/"
    create_path(outputPath)
    data_type = "float32"
    scenario = [
        # "ssp126",
        # "ssp245",
        # "ssp370",
        "ssp585"
    ]
    #################################################数据地址################################################################
    with rio.open(hisAcPath) as ds:
        hisAc = ds.read().astype("float32")[0]
        lats = np.linspace(ds.bounds.top, ds.bounds.bottom, ds.height)[::-1]
        lons = np.linspace(ds.bounds.left, ds.bounds.right, ds.width)

    with rio.open(hisNPPPath) as ds:
        hisNPP = ds.read().astype("float32")[0]

    with rio.open(hisLUCPath) as ds:
        hisLUC = ds.read().astype("float32")[0]

    with rio.open(hisPPath) as ds:
        hisP = ds.read().astype("float32")[0]

    with rio.open(additionPPath) as ds:
        adP = ds.read().astype("float32")[0]

    with rio.open(hisETPath) as ds:
        hisET = ds.read().astype("float32")[0]
    print("适宜性&稳定性评价：")
    for ssp in scenario:
        print(f">>>>> 提取数据 {ssp}", end="")
        with rio.open(os.path.join(futurePPath, f"MMEA_{ssp}_PRE_Avg.tif")) as ds:
            futureP = ds.read().astype("float32")[0]
        ##########################################统计##############################################################
        print(">>>>> 统计 ", end="\n")
        classify = vcc_cal("AC_evaluation", hisAc, hisNPP, hisLUC, hisP, hisET, futureP, adP, process_num=8,
                           backup_path=r"/Volumes/home/HDD/backup/" + "/")
        obj = RasterHandler(outpath=outputPath)
        obj.write_tiff(lats, lons, classify, "classify585")
        # evaluation_nc = xr.DataArray(classify, coords=[lats, lons], dims=["lat", "lon"]).to_dataset(name="classify")
        # print(">>>>> 保存 ", end="")
        # evaluation_nc.to_netcdf(outputPath + r"classify.nc")
        # print(">>>>> 完成")
        ##########################################统计##############################################################




