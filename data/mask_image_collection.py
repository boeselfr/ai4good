#### this is a function that takes an image colleciton from ee and masks it based on a given shp file with polygons
import geopandas as gpd
import ee
from shapely.geometry import Polygon,MultiPolygon
import pprint
import matplotlib.pyplot as plt
import IPython.display as disp
from shapely.ops import unary_union
import shapely
import json

GEE_DEFAULT_CRS = "EPSG:4326"



def get_dummy_collection():
    geoJSON = {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {},
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [
                  -52.4646,
                  -3.6592
                ],
                [
                  -51.7779,
                  -3.6592
                ],
                [
                  -51.7779,
                  -3.2863
                ],
                [
                  -52.4646,
                  -3.2863
                ],
                [
                  -52.4646,
                  -3.6592
                ]
              ]
            ]
          }
        }
      ]
    }

    coords = geoJSON['features'][0]['geometry']['coordinates']
    aoi = ee.Geometry.Polygon(coords)
    ffa_db = ee.Image(ee.ImageCollection('COPERNICUS/S1_GRD')
                      .filterBounds(aoi)
                      .filterDate(ee.Date('2020-08-01'), ee.Date('2020-08-31'))
                      .first()
                      .clip(aoi))
    ffa_fl = ee.Image(ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
                      .filterBounds(aoi)
                      .filterDate(ee.Date('2020-08-01'), ee.Date('2020-08-31'))
                      .first()
                      .clip(aoi))

    ffa_db.bandNames().getInfo()

    coords = ffa_db.getInfo()['properties']['system:footprint']['coordinates'][0]
    footprint = Polygon(coords)
    print(f'footprint of im: {footprint}')
    return ffa_db


# polygon is too annoying for argparse: easier to create tiles if this is a rectangle
def get_footprint_dataset(poly):
    footprint_polygon = gpd.GeoSeries([poly])
    footprint = gpd.GeoDataFrame({'geometry': footprint_polygon})
    return footprint


def intersect_polygons_with_footprint(footprint, polygons):
    # need to adjust crs for both polygons:
    polygons_image = footprint.overlay(polygons, how='intersection')
    return polygons_image


def mask_collection(image, polygons):
    # get footprint from image:
    coords = image.getInfo()['properties']['system:footprint']['coordinates'][0]
    footprint_poly = Polygon(coords)
    footprint = get_footprint_dataset(footprint_poly)
    # convert all polys to same epsg
    footprint = footprint.set_crs(GEE_DEFAULT_CRS, allow_override=True)
    footprint = footprint.to_crs(GEE_DEFAULT_CRS)
    polygons = polygons.set_crs(GEE_DEFAULT_CRS, allow_override=True)
    polygons = polygons.to_crs(GEE_DEFAULT_CRS)

    # intersect the footprint with all polygons from polygons:
    polygons_image = intersect_polygons_with_footprint(footprint, polygons)

    # make one large multipolygon out of the deforested areas
    geomlist = list(polygons_image['geometry'][:])
    multipoly = unary_union(geomlist)
    # convert to ee.multipolygon
    geojson = json.dumps(shapely.geometry.mapping(multipoly))
    geojson = json.loads(geojson)
    multipoly = ee.Geometry.MultiPolygon(geojson['coordinates'])

    # polygon to feature
    feature = ee.FeatureCollection([ee.Feature(multipoly, {'deforestation': 1})])

    # feature to image: (this image is global again so need to clip
    deforestimg = feature.reduceToImage(**{'properties': ['deforestation'],
                                           'reducer': ee.Reducer.first()})


    geojson = json.dumps(shapely.geometry.mapping(footprint_poly))
    geojson = json.loads(geojson)
    footprint_poly = ee.Geometry.Polygon(geojson['coordinates'])

    # need to be clipped to area of interest:
    # two not()'s as first one does conversion to black second one to white
    deforestimg = deforestimg.clip(footprint_poly).Not().Not()

    # add band to original image
    segmented_image = image.addBands(deforestimg).clip(footprint_poly)

    return segmented_image



if __name__ == '__main__':
    poly = '/Users/fredericboesel/Documents/Data Science Master/Herbstsemester 2021/AI4Good/ai4good/data/alertas_2020_com_municipio_e_coordenadas/alertas_2020_com_municipio_e_coordenadas.shp'
    polygons = gpd.read_file(poly)
    print(polygons)

    # Trigger the authentication flow.
    #ee.Authenticate()

    # Initialize the library.
    ee.Initialize()

    im = get_dummy_collection()

    seg_im = mask_collection(im,polygons)
