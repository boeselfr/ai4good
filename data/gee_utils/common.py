import ee
import json
from typing import Union

import geopandas as gpd
import shapely
from shapely.geometry import Polygon, MultiPolygon, polygon

GEE_DEFAULT_CRS = "EPSG:4326"


def shapely_to_gpd(geometry_obj):
    geometry_obj_gpd_gs = gpd.GeoSeries([geometry_obj])
    footprint_gpd_df = gpd.GeoDataFrame({'geometry': geometry_obj_gpd_gs})
    return footprint_gpd_df


def shapely_to_geojson(geometry_obj):
    # Convert to GeoJson
    geometry_obj_geo_j = json.dumps(shapely.geometry.mapping(geometry_obj))
    geometry_obj_geo_j = json.loads(geometry_obj_geo_j)
    return geometry_obj_geo_j


def shapely_to_ee(polygon: Union[Polygon, MultiPolygon]):
    polygon_geo_j = shapely_to_geojson(polygon)
    if type(polygon) is Polygon:
        return ee.Geometry.Polygon(polygon_geo_j["coordinates"])
    elif type(polygon) is MultiPolygon:
        return ee.Geometry.MultiPolygon(polygon_geo_j["coordinates"])
