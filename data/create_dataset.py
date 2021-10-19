"""this file takes a large polygon as input that represents the are that data should be collected from. Then
collects prodes data. Intersects asll polygons with the desired large polygon and collects the sentinel-1 data
for the large polygon as well"""

import argparse
import requests, zipfile, io, os
from shapely.geometry import Polygon
import shapely
import matplotlib.pyplot as plt
import geopandas as gpd
from pathlib import Path
from typing import List
import ee
import json
import folium

_MIN_VALID_YEAR = 2014   # ee sentinel-1 data starts in 2014
_MAX_VALID_YEAR = 2018
AREA_POlY = Polygon([(-52.4646, -3.6592),(-51.7779, -3.6592),
                     (-51.7779, -3.2863),(-52.4646, -3.2863),(-52.4646, -3.6592)])  # this is fixed for now to experiment


def get_dataset(year: int):
    assert year >= _MIN_VALID_YEAR and year <= _MAX_VALID_YEAR
    area = get_area_dataset(AREA_POlY)
    prodes_shp_area = intersect_prodes_with_area(area, year)
    sentineal_area = get_sentinel_imagery(AREA_POlY)
    return prodes_shp_area, sentineal_area


def get_sentinel_imagery(area, year, prodes_polygons):
    ee.Authenticate()
    ee.Initialize()

    # took this from the tutorial its probably faster to do it without the geojson, didnt look into it though
    geoJSON = json.dumps(shapely.geometry.mapping(area))
    coords = geoJSON['features'][0]['geometry']['coordinates']
    aoi = ee.Geometry.Polygon(coords)
    ffa_db = ee.Image(ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filterBounds(aoi)
                      .filterDate(ee.Date(f'{year}-08-01'), ee.Date(f'{year}-08-31'))
                      .first()
                      .clip(aoi))

    # visual check
    rgb = ee.Image.rgb(ffa_db.select('VV'),
                       ffa_db.select('VH'),
                       ffa_db.select('VV').divide(ffa_db.select('VH')))

    # Create the map object.
    location = ee.Geometry.Polygon(coords).centroid().coordinates().getInfo()[
               ::-1]  # Folium expects coordinate as lat long, but gee uses long lat (or the opposite?)
    m = folium.Map(location=location, zoom_start=12)

    # Add the S1 rgb composite to the map object.
    m.add_ee_layer(rgb, {'min': [-20, -20, 0], 'max': [0, 0, 2]}, 'FFA')

    # Add polygon to the map
    folium.GeoJson(data=json.dumps(shapely.geometry.mapping(prodes_polygons)), style_function=lambda x: {'fillColor': 'blue'}).add_to(m)

    # Add a layer control panel to the map.
    m.add_child(folium.LayerControl())

    # Display the map.
    display(m)

    return ffa_db



# polygon is too annoying for argparse: easier to create tiles if this is a rectangle
def get_area_dataset(poly):
    area_polygon = gpd.GeoSeries([poly])
    area = gpd.GeoDataFrame({'geometry': area_polygon, 'area': [1]})
    print(area)
    return area


def intersect_prodes_with_area(area, year):
    """
    :param area: area that all plygons of prodes data get intersected with
    :param year: year of prodes data, this has to be donloaded before
    :return: dataset of polygons that are in the prodes data and intersect the area of interest
    """
    prodes_shp_path = os.path.abspath(f'{str(year)}/PDigital{str(year)}_AMZ_pol.shp')
    print(prodes_shp_path)
    prodes_shp = gpd.read_file(prodes_shp_path)
    prodes_shp_area = area.overlay(prodes_shp, how='intersection')
    print(prodes_shp_area)
    return prodes_shp_area


def _parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Creating a dataset from downloaded shapefiles and sentinel-1 data for a specified area"
    )

    parser.add_argument(
        "-y",
        "--year",
        required=True,
        type=int
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # get_dataset(year=args.year)
    area = get_area_dataset(AREA_POlY)
    prodes_polygons = intersect_prodes_with_area(area, year=args.year)
    ffa = get_sentinel_imagery(area, args.year, prodes_polygons)