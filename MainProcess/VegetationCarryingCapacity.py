import os.path
from typing import Dict
import numpy as np
import xarray as xr
import rasterio as rio
from Tools.Tool import create_path
from ThreadProcess.VCC_MultiProcessing import handler as vcc_cal


if __name__ == '__main__':
    #################################################数据地址################################################################
    hisAcPath = r"\\ecohydrologylab.asirnas.top\FluxGroup\TNSP_Data\AC\AC_avg_2001-2021.tif"
    hisNPPPath = r"\\ecohydrologylab.asirnas.top\FluxGroup\TNSP_Data\TNSP_YearlyNPPsum_MODIS_2001-2021_500m\NPPsum2021.tif"
    hisLUCPath = r"\\ecohydrologylab.asirnas.top\FluxGroup\TNSP_Data\TNSP_YearlyLandUse_MODIS_2001-2021_500m\LandUse2021.tif"
    futurePPath = r"\\ecohydrologylab.asirnas.top\FluxGroup\GCM_P\Interpolation" + "\\"
    outputPath = r"\\ecohydrologylab.asirnas.top\FluxGroup\TNSP_Data\AC" + "\\"
    create_path(outputPath)
    data_type = "float32"
    scenario = [
        "ssp126",
        # "ssp245",
        # "ssp370",
        # "ssp585"
    ]
    #################################################数据地址################################################################
    with rio.open(hisAcPath) as ds:
        hisAc = ds.read().astype("float32")[0]
        lats = np.linspace(ds.bounds.top, ds.bounds.bottom, ds.height)
        lons = np.linspace(ds.bounds.left, ds.bounds.right, ds.width)

    with rio.open(hisNPPPath) as ds:
        hisNPP = ds.read().astype("float32")[0]

    with rio.open(hisLUCPath) as ds:
        hisLUC = ds.read().astype("float32")[0]
    print("适宜性&稳定性评价：")
    for ssp in scenario:
        print(f">>>>> 提取数据 {ssp}", end="")
        with rio.open(os.path.join(futurePPath, f"MMEA_{ssp}_PRE_Avg.tif")) as ds:
            futureP = ds.read().astype("float32")[0]
        ##########################################统计##############################################################
        print(">>>>> 统计 ", end="\n")
        classify = vcc_cal("AC_evaluation", hisAc, hisNPP, hisLUC, futureP, process_num=8,
                           backup_path=r"/Volumes/home/HDD/backup/" + "/")
        evaluation_nc = xr.DataArray(classify, coords=[lats, lons], dims=["lat", "lon"]).to_dataset(name="classify")
        print(">>>>> 保存 ", end="")
        evaluation_nc.to_netcdf(outputPath + r"classify.nc")
        print(">>>>> 完成")
        ##########################################统计##############################################################




