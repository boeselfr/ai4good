import ee
import geopandas as gpd
import json
from pathlib import Path
from .common import GEE_DEFAULT_CRS

BRAZILIAN_LEGAL_AMAZON_SHAPEFILE_PATH = Path(
    __file__).parents[1] / "brazilian_legal_amazon/brazilian_legal_amazon.shp"


def get_brazilian_legal_amazon_polygon() -> ee.Geometry.Polygon:
    bla_gpd_df = gpd.read_file(str(BRAZILIAN_LEGAL_AMAZON_SHAPEFILE_PATH))
    # Convert to EPSG:4326 to avoid errors when generating the GEE polygon later
    bla_gpd_df = bla_gpd_df.to_crs(GEE_DEFAULT_CRS)
    # Convert the geometry to GeoJSON dict
    bla_geom_geo_j = json.loads(
        gpd.GeoSeries(bla_gpd_df.iloc[0]["geometry"]).to_json())
    # Creat the EE polygon
    return ee.Geometry.Polygon(
        bla_geom_geo_j["features"][0]["geometry"]["coordinates"])