import argparse
import datetime
import functools
import math
from typing import Union

import ee
import tensorflow as tf
from data.gee_utils import sentinel, export

ee.Initialize()

_IMAGE_COLLECTION_BUILDERS = {
    "s1": sentinel.build_sentinel_1_image_collection,
    "s2": sentinel.build_sentinel_2_image_collection,
}

_AOI_ASSET_IDS = {
    "para": "users/albanesegiuliano97/para_state_simple",
    "brazil_simple": "users/albanesegiuliano97/brazil_simple",
    "sample_area_1": "users/albanesegiuliano97/sample_area_1",
}

_MAPBIOMAS_2020_SIMPLE_ASSET_ID = "users/albanesegiuliano97/mapbiomas_2020_simplified"
_MAPBIOMAS_2020_DETECTION_DATE_COLUMN_NAME = "DataDetec"

_USE_PRECOMPUTED_MULTISPECTRAL_OPTICAL = False
_MULTISPECTRAL_OPTICAL_ASSET_ID = (
    "users/albanesegiuliano97/para_simple_s2_composite_202012_202110_100pxm"
)

_SAR_BANDS = ["VV", "VH"]
_MULTISPECTRAL_OPTICAL_BANDS = ["B4", "B3", "B2"]  # Bands for composite visualization


def _validate_date(date: str):
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid data format, should be: YYYY-MM-DD")


def _get_footprint_as_polygon(image: ee.Image) -> ee.Geometry.Polygon:
    return ee.Geometry.Polygon(ee.Geometry(image.get("system:footprint")).coordinates())


def _get_image_collection(
    image_collection_id: str,
    start_date: ee.Date,
    end_date: ee.Date,
    aoi: Union[ee.FeatureCollection, ee.Geometry.Polygon],
) -> ee.ImageCollection:
    # TODO: read config params from file
    builder = _IMAGE_COLLECTION_BUILDERS[image_collection_id]

    # Build the image collection
    return builder(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
    )


def _get_simplified_mapbiomas_2020():
    fc = ee.FeatureCollection(_MAPBIOMAS_2020_SIMPLE_ASSET_ID)

    def add_time_start_fn(feat: ee.Feature):
        return feat.set(
            "system:time_start",
            ee.Date(feat.get(_MAPBIOMAS_2020_DETECTION_DATE_COLUMN_NAME)).format(),
        )

    # Add system:time_start to every polygon so we can filter the collection by date
    return fc.map(add_time_start_fn)


def _get_multispectral_optical_composite():
    if _USE_PRECOMPUTED_MULTISPECTRAL_OPTICAL:
        return ee.Image(_MULTISPECTRAL_OPTICAL_ASSET_ID)
    aoi = ee.FeatureCollection(_AOI_ASSET_IDS["brazil_simple"])
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate("2020-12-01", "2021-10-31")
        .filterBounds(aoi)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    return s2.select(_MULTISPECTRAL_OPTICAL_BANDS).median().divide(10000)


def _append_multispectral_optical_composite(
    image: ee.Image,
    multispectral_optical_composite: ee.Image,
):
    return image.addBands(multispectral_optical_composite)


def _min_date(date_a: ee.Date, date_b: ee.Date):
    return ee.Algorithms.If(
        date_a.difference(date_b, "day").lt(ee.Number(0)), date_a, date_b
    )


def _make_segmentation_mask_from_polygons(polygons: ee.FeatureCollection):
    # Note 1: Here we use clipToCollection because it's faster when clipping to a large
    # collection with a complex geometry - .clip(polygons.geometry()) would be much slower
    # Note 2: we do .Not() twice to generate a mask with white foreground
    return ee.Image(1).clipToCollection(polygons).mask().Not().Not()


def _append_deforestation_mask(
    image: ee.Image,
    deforestation_polygons: ee.FeatureCollection,
    start_date: ee.Date,
    months_before_acquisition: int,
):
    """
    Append a binary segmentation mask created using the polygons detected
    before the acquisition date that lie inside the footprint of the image
    """
    # Get all the polygons detected before the acquisition date of the image
    acquisition_date = ee.Date(image.get("system:time_start"))
    polygons_start_date = _min_date(
        start_date, acquisition_date.advance(months_before_acquisition, "month")
    )
    # Filter by the the footprint of the image
    footprint = _get_footprint_as_polygon(image)
    polygons = deforestation_polygons.filterDate(
        polygons_start_date, acquisition_date
    ).filterBounds(footprint)
    # Create binary deforestation mask from polygons
    mask = ee.Image(
        ee.Algorithms.If(
            polygons.size().gt(ee.Number(0)),
            _make_segmentation_mask_from_polygons(polygons),
            ee.Image(0),
        )
    )
    # Add the mask as a band and return
    return image.addBands(mask.clip(footprint).rename("deforestation_mask"))


def create_deforestation_detection_dataset(
    image_collection_id: str,
    start_date: str,
    end_date: str,
    area_of_interest: str,
    months_before_acquisition: int,
):
    _validate_date(start_date)
    _validate_date(end_date)

    assert months_before_acquisition > -1

    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)

    # Load aoi polygons from assets
    aoi_polygons = ee.FeatureCollection(_AOI_ASSET_IDS[area_of_interest])

    # convert to FeatureCollection adding system:time_start
    deforest_polygons_ee = _get_simplified_mapbiomas_2020()

    # Fetch the image collection for GEE
    ic = _get_image_collection(
        image_collection_id=image_collection_id,
        start_date=start_date_ee,
        end_date=end_date_ee,
        aoi=aoi_polygons,
    )

    # Append multispectral composite for visualization
    multispectral_optical_composite = _get_multispectral_optical_composite()
    append_multispectral_optical_map_fn = functools.partial(
        _append_multispectral_optical_composite,
        multispectral_optical_composite=multispectral_optical_composite,
    )
    ic = ic.map(append_multispectral_optical_map_fn)

    # Append deforestation mask to each image
    append_deforestation_mask_map_fn = functools.partial(
        _append_deforestation_mask,
        deforestation_polygons=deforest_polygons_ee,
        start_date=ee.Date("2020-01-01"),
        months_before_acquisition=months_before_acquisition,
    )
    ic = ic.map(append_deforestation_mask_map_fn)

    # Sort the collection by date
    ic = ic.sort("system:time_start")

    num_images = ic.size().getInfo()
    ic_list = ic.toList(ic.size())

    # TODO: these should be all CLI-configurable parameters
    kernel_size = 256
    total_samples_per_image = 100
    num_shards_per_image = 10
    num_samples_per_shard = total_samples_per_image / num_shards_per_image
    scale = 40  # meters per pixel
    bands = _SAR_BANDS + _MULTISPECTRAL_OPTICAL_BANDS + ["deforestation_mask"]
    shard_basename = "deforest-training-patches"
    bucket = "ai4good-3b"
    export_folder = (
        "deforest-training_"
        + area_of_interest
        + "_"
        + start_date.replace("-", "")
        + "_"
        + end_date.replace("-", "")
        + "_"
        + f"scale-{scale}"
    )

    lists = ee.List.repeat(ee.List.repeat(1, kernel_size), kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)
    for i in range(num_images):
        image = ee.Image(ic_list.get(i)).float()
        array_image = image.neighborhoodToArray(kernel)

        # Export all the training data (in many pieces), with one task per image
        geomSample = ee.FeatureCollection([])
        for j in range(num_shards_per_image):
            sample = array_image.sample(
                region=None,  # Use default image footprint
                scale=scale,
                numPixels=num_samples_per_shard,  # Size of the shard.
                seed=j,
                tileScale=8,
            )
            geomSample = geomSample.merge(sample)

        desc = shard_basename + "_" + str(i)
        task = ee.batch.Export.table.toCloudStorage(
            collection=geomSample,
            description=desc,
            bucket=bucket,
            fileNamePrefix=export_folder + "/" + desc,
            fileFormat="TFRecord",
            selectors=bands,
        )
        task.start()


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
        "-ic",
        "--image-collection-id",
        type=str,
        choices=["s1", "s2"],
        required=True,
        help="The image collection to sample data from: s1 (Sentinel-1) or s2 (Sentinel-2)",
    )

    parser.add_argument(
        "-sd",
        "--start-date",
        required=True,
        type=str,
        help="The start date to filter the collection.",
    )

    parser.add_argument(
        "-ed",
        "--end-date",
        required=True,
        type=str,
        help="The end date.",
    )

    parser.add_argument(
        "-aoi",
        "--area-of-interest",
        type=str,
        choices=list(_AOI_ASSET_IDS.keys()),
        default="para",
        help="Name of the area of interest to sample images from.",
    )

    parser.add_argument(
        "--months-before-acquisition",
        type=int,
        default=3,
        help="To generate the segmentation mask of each image consider the polygons "
        "that were detected within a time window of this length before the acquisition time.",
    )

    parser.add_argument(
        "--min-num-deforest-polygons",
        type=int,
        default=10,
        help="The minimum number of polygons an image should have to be added to the dataset",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_deforestation_detection_dataset(
        image_collection_id=args.image_collection_id,
        start_date=args.start_date,
        end_date=args.end_date,
        area_of_interest=args.area_of_interest,
        months_before_acquisition=args.months_before_acquisition,
    )
