import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import json
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point


def voronoi_finite_polygons(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def clip_voronoi(regions, vertices, box):
    poly_list = []
    area_list = []
    for region in regions:
        polygon = vertices[region]
        poly = Polygon(polygon)
        poly = poly.intersection(box)
        poly_list.append(poly)
        area_list.append(poly.area)
    voro_df = gpd.GeoDataFrame(gpd.GeoSeries(poly_list), columns=['geometry'])
    voro_df.crs = "epsg:4326"
    voro_df["area"] = voro_df.to_crs("epsg:2381").applymap(lambda p: p.area / 10 ** 6)
    return voro_df


def getVoronoiAverage(data, location_info, boundary):
    # 匹配相同站点
    stationIDs = location_info.min_ac.values
    stationIDs = [_id for _id in data.columns.values if _id in stationIDs]
    data = data[stationIDs]
    # 绘制泰森多边形, 计算站点控制面积(权重)
    coords = location_info[["LONG", "LAT"]].values
    vor = Voronoi(coords)
    regions, vertices = voronoi_finite_polygons(vor)
    box = gpd.read_file(boundary).to_crs(epsg=4326).unary_union
    voro_gdf = clip_voronoi(regions, vertices, box)
    voro_df = pd.DataFrame(
        {"Station": stationIDs, "Area": voro_gdf["area"], "Weight": voro_gdf["area"] / voro_gdf["area"].sum()})
    # 计算流域泰森平均
    voro_avg = np.multiply(data.values, voro_df["Weight"].values).sum(axis=1)
    return voro_avg


if __name__ == "__main__":
    location_info = pd.read_csv(r"../../dataFile/Station/气象站.csv")
    boundary = r"..\dataFile\Boundary\Jlboundary.shp"
    data = None
    getVoronoiAverage(data, location_info, boundary)
