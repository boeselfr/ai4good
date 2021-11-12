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
import PIL
from PIL import Image, ImageDraw
import regionmask
import numpy as np

import rasterio
from rasterio.plot import reshape_as_image
import rasterio.mask
from rasterio.features import rasterize

import pandas as pd
from shapely.geometry import mapping, Point, Polygon
from shapely.ops import cascaded_union


_MIN_VALID_YEAR = 2014   # ee sentinel-1 data starts in 2014
_MAX_VALID_YEAR = 2018
AREA_POlY = Polygon([(-52.4646, -3.6592),(-51.7779, -3.6592),
                     (-51.7779, -3.2863),(-52.4646, -3.2863),(-52.4646, -3.6592)])  # this is fixed for now to experiment


def get_dataset(year: int):
    assert year >= _MIN_VALID_YEAR and year <= _MAX_VALID_YEAR
    area = get_area_dataset(AREA_POlY)
    prodes_shp_area = intersect_prodes_with_area(area, year)
    mask = create_segmentation_map(prodes_shp_area, year)
    #sentineal_area = get_sentinel_imagery(AREA_POlY)
    return prodes_shp_area, sentineal_area



# ignore this for now and focus on the segmentation mask
"""def get_sentinel_imagery(area, year, prodes_polygons):
    #ee.Authenticate()
    #ee.Initialize()

    # took this from the tutorial its probably faster to do it without the geojson, didnt look into it though
    geoJSON = json.dumps(shapely.geometry.mapping(area))
    print(geoJSON[0])
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

    return ffa_db"""


# polygon is too annoying for argparse: easier to create tiles if this is a rectangle
def get_area_dataset(poly):
    area_polygon = gpd.GeoSeries([poly])
    area = gpd.GeoDataFrame({'geometry': area_polygon, 'area': [1]})
    #print(area)
    return area


def intersect_prodes_with_area(area, year):
    """
    :param area: area that all plygons of prodes data get intersected with
    :param year: year of prodes data, this has to be donloaded before
    :return: dataset of polygons that are in the prodes data and intersect the area of interest
    """
    prodes_shp_path = os.path.abspath(f'{str(year)}/PDigital{str(year)}_AMZ_pol.shp')
    #2015:
    #prodes_shp_path = os.path.abspath(f'{str(year)}/PDigital{str(year)}_AMZ_Agregado_pol.shp')
    #print(prodes_shp_path)
    prodes_shp = gpd.read_file(prodes_shp_path)
    # need to adjust crs for both polygons:
    epsg = prodes_shp.crs.to_epsg()
    print(f'epsg: {epsg}')
    area = area.set_crs(epsg, allow_override=True)
    area = area.to_crs(epsg=epsg)

    prodes_shp_area = area.overlay(prodes_shp, how='intersection')
    print(prodes_shp_area)
    return prodes_shp_area


def test_img():
    img = Image.open('prodes_polygons.png')
    img.show()
    return


def create_segmentation_map(prodes_shp_area, year):
    # define raster on area:
    # defin resolution in decimal places
    prodes_shp_area.plot()
    plt.axis('off')
    plt.savefig(f'prodes_polygons_{year}.png', bbox_inches='tight', pad_inches=0, transparent=True, dpi=1200)
    mask = Image.open(f'prodes_polygons_{year}.png')
    thresh = 100
    fn = lambda x: 0 if x > thresh else 255
    mask = mask.convert('L').point(fn, mode='1')
    mask.show()
    mask.save(f'prodes_polygons_{year}.png')


    return mask


def generate_mask(raster_path, shape_path, output_path, file_name):
    """Function that generates a binary mask from a vector file (shp or geojson)

    raster_path = path to the .tif;

    shape_path = path to the shapefile or GeoJson.

    output_path = Path to save the binary mask.

    file_name = Name of the file.

    """

    # load raster

    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta

    # load o shapefile ou GeoJson
    train_df = gpd.read_file(shape_path)
    # optinioal desmatamento instead
    train_df = train_df[train_df['sprclasse'] == 'desmatamento_total']
    print(len(train_df))
    # Verify crs
    if train_df.crs != src.crs:
        print(" Raster crs : {}, Vector crs : {}.\n Convert vector and raster to the same CRS.".format(src.crs,
                                                                                                       train_df.crs))

    # Function that generates the mask
    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            poly_pts.append(~transform * tuple(i))

        new_poly = Polygon(poly_pts)
        return new_poly

    poly_shp = []
    im_size = (src.meta['height'], src.meta['width'])
    for num, row in train_df.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    mask = rasterize(shapes=poly_shp,
                     out_shape=im_size)
    print(im_size)

    # Salve
    mask = mask.astype("uint16")

    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    os.chdir(output_path)
    with rasterio.open(file_name, 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)


def create_segmentation_map_aligned(prodes_shp_area, area):
    raster_path = '/Users/fredericboesel/Documents/Data Science Master/Herbstsemester 2021/AI4Good/ai4good/data/partitioned/2015/Landsat8_00160_16092015.tif'
    with rasterio.open(raster_path, "r") as src:
        raster_img = src.read()
        raster_meta = src.meta

    print("CRS Raster: {}, CRS Vector {}".format(prodes_shp_area.crs, src.crs))

    def poly_from_utm(polygon, transform):
        poly_pts = []

        poly = cascaded_union(polygon)
        for i in np.array(poly.exterior.coords):
            # Convert polygons to the image CRS
            poly_pts.append(~transform * tuple(i))

        # Generate a polygon object
        new_poly = Polygon(poly_pts)
        return new_poly

    # Generate Binary maks

    poly_shp = []

    poly = list(area['geometry'][0].exterior.coords)
    x, y = map(list, zip(*poly))
    width = len(np.arange(min(x), max(x), 0.0001))
    height = len(np.arange(min(y), max(y), 0.0001))
    print(width, height)
    im_size = (width, height)
    for num, row in prodes_shp_area.iterrows():
        if row['geometry'].geom_type == 'Polygon':
            poly = poly_from_utm(row['geometry'], src.meta['transform'])
            poly_shp.append(poly)
        else:
            for p in row['geometry']:
                poly = poly_from_utm(p, src.meta['transform'])
                poly_shp.append(poly)

    print(np.shape(poly_shp))
    """mask = rasterize(shapes=prodes_shp_area['geometry'],
                     out_shape=im_size)"""

    print(src.meta['transform'])
    mask = rasterio.features.geometry_mask(prodes_shp_area['geometry'], im_size, transform=rasterio.transform.from_origin(min(x), max(y), width, height))
    print(mask)
    print(np.count_nonzero(1*mask))
    # Plot the mask

    mask = mask.astype("uint16")

    bin_mask_meta = src.meta.copy()
    bin_mask_meta.update({'count': 1})
    with rasterio.open('maks_aligned.png', 'w', **bin_mask_meta) as dst:
        dst.write(mask * 255, 1)

    return mask



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
    #area = get_area_dataset(AREA_POlY)
    #prodes_polygons = intersect_prodes_with_area(area, year=args.year)
    #mask = create_segmentation_map(prodes_polygons, args.year)
    #mask = create_segmentation_map_aligned(prodes_polygons, area)
    #ffa = get_sentinel_imagery(area, args.year, prodes_polygons)
    raster_path = '/Users/fredericboesel/Documents/Data Science Master/Herbstsemester 2021/AI4Good/ai4good/data/partitioned/2015/TM_00160_06102005.tif'
    shape_path =  '/Users/fredericboesel/Documents/Data Science Master/Herbstsemester 2021/AI4Good/ai4good/data/partitioned/2015/PDigital2005_00160_shp/PDigital2005_00160_368_pol.shp'
    output_path = 'partitioned/2015'
    file_name = 'aligned_mask.png'

    generate_mask(raster_path, shape_path, output_path, file_name)

