import argparse
import datetime
import ee
import json

from typing import Union

import geopandas as gpd

from pathlib import Path

from googleapiclient.http import RequestMockBuilder
from data.gee_utils.common import GEE_DEFAULT_CRS, shapely_to_ee
from data.gee_utils import sentinel, export

ee.Initialize()

_IMAGE_COLLECTION_BUILDERS = {
    "s1": sentinel.sample_s1_grd_images,
    "s2": sentinel.sample_s2_sr_images
}


def _validate_date(date: str):
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid data format, should be: YYYY-MM-DD")


def _import_polygons_from_file(polygons_file_path: Path):
    """Import polygons from geodata file
    """
    gdf = gpd.read_file(str(polygons_file_path))
    # Explode multi part geometries if present
    gdf = gdf.explode(index_parts=True)
    # Convert to the default CRS
    gdf = gdf.to_crs(GEE_DEFAULT_CRS)
    return gdf


def _convert_polygons_to_feature_collection_with_date(
    polygons: gpd.GeoDataFrame,
    date_column_name: str,
):
    # Convert to feature collection
    feat_coll = ee.FeatureCollection(json.loads(polygons.to_json()))

    # Add property "system:time_start" so later we can use filterDate
    def set_time_start_fn(feat: ee.Feature) -> ee.Feature:
        return feat.set("system:time_start",
                        ee.Date(feat.get(date_column_name)))

    return feat_coll.map(set_time_start_fn)


def _get_image_collection(
    image_collection_id: str,
    start_date: ee.Date,
    end_date: ee.Date,
    aoi: Union[ee.Geometry.Polygon, ee.FeatureCollection],
) -> ee.ImageCollection:
    # FIXME: each collection has its own specific parameters
    # and preprocessing steps which should be read from a config file but
    # for now we just use the default parameters
    builder = _IMAGE_COLLECTION_BUILDERS[image_collection_id]

    # Build the image collection
    return builder(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
    )


def _create_binary_image_from_polygons(
        aoi: ee.Geometry.Polygon, polygons: ee.FeatureCollection) -> ee.Image:
    """ From:
    https://gis.stackexchange.com/questions/304923/creating-binary-image-from-featurecollection-in-google-earth-engine
    """
    aoi_image = ee.Image(0).clip(aoi)
    polygons_image = ee.Image(1).clip(polygons.geometry())
    return aoi_image.where(test=polygons_image, value=polygons_image)


def create_collection_with_masks(
    image_collection_id: str,
    start_date: str,
    end_date: str,
    deforestation_polygons_file_path: Path,
    date_column_name: str,
    aoi_polygon_file_path: Path,
    # TODO: use these parameters to filter the collection
    before_acq_window: str,
    after_acq_window: str,
):
    assert deforestation_polygons_file_path.is_file()
    assert aoi_polygon_file_path.is_file()

    _validate_date(start_date)
    _validate_date(end_date)

    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)

    # Load aoi polygons
    aoi_polygons = _import_polygons_from_file(aoi_polygon_file_path)
    # Simplify geometry to speed up calcs
    aoi_polygons["geometry"] = aoi_polygons.simplify(tolerance=0.01)
    aoi_polygons_feat_coll = ee.FeatureCollection(
        json.loads(aoi_polygons.to_json()))

    # Load the deforestation polygons and convert to FeatureCollection adding system:time_start
    deforest_polygons = _import_polygons_from_file(
        deforestation_polygons_file_path)
    assert date_column_name in deforest_polygons.columns
    deforest_polygons_ee = _convert_polygons_to_feature_collection_with_date(
        deforest_polygons,
        date_column_name,
    ).filterBounds(aoi_polygons_feat_coll.geometry()).filterDate(
        start_date_ee, end_date_ee)

    # Fetch the image collection from GEE and filter it by date and aoi
    # TODO: implement collection builders to pass more filters and / or params
    ic = _get_image_collection(
        image_collection_id=image_collection_id,
        start_date=start_date_ee,
        end_date=end_date_ee,
        aoi=aoi_polygons_feat_coll.geometry(),
    )

    def add_deforestation_mask_fn(image: ee.Image) -> ee.Image:
        # Get all the polygons detected before the acquisition date of the image
        # and inside the image footprint
        acquisition_date = ee.Date(image.get("system:time_start"))
        footprint = ee.Geometry.Polygon(
            ee.Geometry(image.get('system:footprint')).coordinates())
        polygons = deforest_polygons_ee.filterDate(
            start_date_ee, acquisition_date).filterBounds(footprint)
        # Create binary deforestation mask from polygons
        deforestation_mask = ee.Algorithms.If(
            polygons.size().eq(ee.Number(0)),
            ee.Image(0).clip(footprint),  # empty mask if no polygons
            _create_binary_image_from_polygons(footprint, polygons))
        # Add the mask as a band and return
        return image.addBands(deforestation_mask)

    # Add a binary segmentation mask create using the polygons detected
    # before the acquisition date that lie inside the footprint of the image
    ic_with_deforestation_map = ic.map(add_deforestation_mask_fn)

    export.export_image_collection_to_gdrive(
        ic_with_deforestation_map,
        "ai4good",
    )


def _parse_args():
    desc = """
    Create a deforestation detection dataset from a GEE
    ImageCollection and a set of groundtruth polygons.
    """

    # FIXME: the script has too many required options - instead
    # we should configure dataset creation params through a config
    # file, also to improve reproducibility

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '-ic',
        '--image-collection-id',
        type=str,
        choices=["s1", "s2"],
        required=True,
        help=
        "The image collection to sample data from: s1 (Sentinel-1) or s2 (Sentinel-2)"
    )

    parser.add_argument("-sd",
                        "--start-date",
                        required=True,
                        type=str,
                        help="The start date to filter the collection.")

    parser.add_argument("-ed",
                        "--end-date",
                        required=True,
                        type=str,
                        help="The end date.")

    parser.add_argument(
        '-dp',
        '--deforestation-polygons-file-path',
        type=lambda p: Path(p).absolute(),
        required=True,
        help="Path to a file containing the groundtruth polygons description.")

    parser.add_argument(
        '-dc',
        '--date-column-name',
        type=str,
        required=True,
        help=
        "Name of the polygon detection date column in the poligons .shp file.")

    parser.add_argument(
        '-aoi',
        '--aoi-polygon-file-path',
        type=lambda p: Path(p).absolute(),
        required=True,
        help=
        "Path to a geodata file containing the polygon of the area of interest",
    )

    parser.add_argument(
        "-b",
        "--before",
        type=str,
        help=
        "To generate the segmentation mask of each image consider the polygons "
        "that were detected within a time window of this length before the acquisition time."
    )

    parser.add_argument(
        "-a",
        "--after",
        type=str,
        help=
        "To generate the segmentation mask of each image consider the polygons "
        "that were detected within a time window of this length after the acquisition time."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_collection_with_masks(
        image_collection_id=args.image_collection_id,
        start_date=args.start_date,
        end_date=args.end_date,
        deforestation_polygons_file_path=args.deforestation_polygons_file_path,
        date_column_name=args.date_column_name,
        aoi_polygon_file_path=args.aoi_polygon_file_path,
        before_acq_window=args.before,
        after_acq_window=args.after,
    )
