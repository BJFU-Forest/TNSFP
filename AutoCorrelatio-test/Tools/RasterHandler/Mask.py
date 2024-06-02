import numpy as np
import xarray as xr
from rasterio import features
from affine import Affine
import geopandas as gpd


def transform_from_latlon(lat, lon):
    """ input 1D array of lat / lon and output an Affine transformation
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    """Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.

    usage:
    -----
    1. read shapefile to geopandas.GeoDataFrame
          `states = gpd.read_file(shp_dir+shp_file)`
    2. encode the different shapefiles that capture those lat-lons as different
        numbers i.e. 0.0, 1.0 ... and otherwise np.nan
          `shapes = (zip(states.geometry, range(len(states))))`
    3. Assign this to a new coord in your original xarray.DataArray
          `ds['states'] = rasterize(shapes, ds.coords, longitude='X', latitude='Y')`

    arguments:
    ---------
    : **kwargs (dict): passed to `rasterio.rasterize` function

    attrs:
    -----
    :transform (affine.Affine): how to translate from latlon to ...?
    :raster (numpy.ndarray): use rasterio.features.rasterize fill the values
      outside the .shp file with np.nan
    :spatial_coords (dict): dictionary of {"X":xr.DataArray, "Y":xr.DataArray()}
      with "X", "Y" as keys, and xr.DataArray as values

    returns:
    -------
    :(xr.DataArray): DataArray with `values` of nan for points outside shapefile
      and coords `Y` = latitude, 'X' = longitude.


    """
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def trans_shape_as_xarray(shp_path, field_name, xr_coords, latitude='latitude', longitude='longitude'):
    """
    根据shp和坐标系生成xr
    :param shp_path: shp文件位置 <os path>
    :param field_name: 字段名 (str)
    :param xr_coords: 坐标系 (xarray, xr.coords)
    :param latitude: 坐标系纬度名 (str)
    :param longitude: 坐标系经度名 (str)
    :return:
    """
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in zip(shp_gpd[field_name], shp_gpd.geometry)]
    # shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]
    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    shp_xr = rasterize(shapes, xr_coords, latitude=latitude, longitude=longitude)
    return shp_xr


if __name__ == "__main__":
    shp_dir = r"G:\ForSarwan\DataFile\Pak SHP\National_Boundary.shp"
    nc_dir = r"H:\SarwanData\GPCP\gpcp_v01r03_daily_d19961001_c20170530.nc"

    # 数据读取及时间平均处理
    ds = xr.open_dataset(nc_dir)
    print(ds)
    lat = ds.latitude
    lon = ds.longitude
    time = ds.time
    prec = ds["precip"]  # 把温度转换为℃
    # 区域选择
    lon_range = lon[(lon > 60) & (lon < 80)]
    lat_range = lat[(lat > 20) & (lat < 40)]
    prec = prec.sel(longitude=lon_range, latitude=lat_range)

    # 统一分辨率为0.25°
    lat25 = np.asarray([i for i in np.arange(20, 40, 0.05)])
    lon25 = np.asarray([i for i in np.arange(60, 80, 0.05)])
    prec = prec.interp(latitude=lat25, longitude=lon25)

    precip_da = add_shape_coord_from_data_array(prec, shp_dir, "Pak")
    mask = precip_da.Pak
    mask = np.asarray(mask)
    awash_da = precip_da.where(precip_da.Pak == precip_da.Pak, other=np.nan)
    import matplotlib.pyplot as plt

    awash_da.mean(dim="time").plot()

    plt.show()
