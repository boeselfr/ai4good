import argparse
import datetime
import math

import ee

from data.globals import SAR_EXPORT_BANDS, MSO_EXPORT_BANDS, RESPONSES
from data.patch_sampler import PatchSampler

ee.Initialize()


## Assets ##
_AOI_ASSET_IDS = {
    "para": "users/albanesegiuliano97/para_state_simple",
    "brazil_simple": "users/albanesegiuliano97/brazil_simple",
    "sample_area_1": "users/albanesegiuliano97/sample_area_1",
    "sample_area_2": "users/albanesegiuliano97/sample_area_2",
    "sampling_rectangles_1": "users/albanesegiuliano97/ai4good/sampling_rectangles_1",
    "sampling_rectangles_2": "users/albanesegiuliano97/ai4good/sampling_rectangles_2",
}

_MAPBIOMAS_2020_ASSET_ID = "users/albanesegiuliano97/mapbiomas_2020"
_MAPBIOMAS_2020_DETECTION_DATE_COLUMN_NAME = "DataDetec"
_MAX_SAMPLES_PER_SHARD = 20


def _validate_date(date: str):
    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid data format, should be: YYYY-MM-DD")


def _get_footprint_as_polygon(image: ee.Image) -> ee.Geometry.Polygon:
    return ee.Geometry.Polygon(ee.Geometry(image.get("system:footprint")).coordinates())


def _get_mapbiomas_2020():
    fc = ee.FeatureCollection(_MAPBIOMAS_2020_ASSET_ID)

    def add_time_start_fn(feat: ee.Feature):
        return feat.set(
            "system:time_start",
            ee.Date(feat.get(_MAPBIOMAS_2020_DETECTION_DATE_COLUMN_NAME)).format(),
        )

    # Add system:time_start to every polygon so we can filter the collection by date
    return fc.map(add_time_start_fn)


def _get_s1(start_date, end_date, log_scale=True):
    ic = "COPERNICUS/S1_GRD" if log_scale else "COPERNICUS/S1_GRD_FLOAT"
    return (
        ee.ImageCollection(ic)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
    )


def _make_multispectral_optical_composite(aoi=None):
    if aoi is None:
        aoi = ee.FeatureCollection(_AOI_ASSET_IDS["brazil_simple"])
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR")
        .filterDate("2020-12-01", "2021-10-31")
        .filterBounds(aoi)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 40))
    )
    return s2.select(MSO_EXPORT_BANDS).median().divide(10000)


def _min_date(date_a: ee.Date, date_b: ee.Date):
    return ee.Algorithms.If(
        date_a.difference(date_b, "day").lt(ee.Number(0)), date_a, date_b
    )


def _make_segmentation_mask_from_polygons(polygons: ee.FeatureCollection):
    # Note 1: Here we use clipToCollection because it's faster when clipping to a large
    # collection with a complex geometry - .clip(polygons.geometry()) would be much slower
    # Note 2: we do .Not() twice to generate a mask with white foreground
    return ee.Image(1).clipToCollection(polygons).mask().Not().Not()


def _make_deforestation_mask(
    image: ee.Image,
    deforestation_polygons: ee.FeatureCollection,
    months_before_acquisition: int,
) -> ee.Image:
    acquisition_date = ee.Date(image.get("system:time_start"))
    polygons_start_date = acquisition_date.advance(-months_before_acquisition, "month")
    footprint = _get_footprint_as_polygon(image)
    polygons = deforestation_polygons.filterDate(
        polygons_start_date, acquisition_date
    ).filterBounds(footprint)
    mask = _make_segmentation_mask_from_polygons(polygons)
    return mask.rename(RESPONSES[0])


def _make_time_series_with_groundtruth(
    before_image,
    after_image,
    multispectral_composite,
    difference_mask,
    aoi,
):
    image = before_image.rename(SAR_EXPORT_BANDS[:2])

    image = image.addBands(after_image.rename(SAR_EXPORT_BANDS[2:]))

    image = image.addBands(multispectral_composite)

    image = image.addBands(difference_mask)

    return image


def create_deforestation_detection_dataset(
    name: str,
    start_date: str,
    end_date: str,
    log_scale: bool,
    area_of_interest: str,
    months_before_acquisition: int,
    scale: int,
    kernel_size: int,
    num_samples_per_area: int,
):
    _validate_date(start_date)
    _validate_date(end_date)

    assert months_before_acquisition > -1

    start_date_ee = ee.Date(start_date)
    end_date_ee = ee.Date(end_date)

    # Load aoi polygons from assets
    train_aoi_polygons = ee.FeatureCollection(_AOI_ASSET_IDS[area_of_interest])

    # convert to FeatureCollection adding system:time_start
    deforest_polygons_ee = _get_mapbiomas_2020()

    # Configure export params
    num_shards_per_area = math.ceil(num_samples_per_area / _MAX_SAMPLES_PER_SHARD)
    num_samples_per_shard = min(
        math.ceil(num_samples_per_area / num_shards_per_area),
        _MAX_SAMPLES_PER_SHARD,
    )

    # Initialize patch sampler
    patch_sampler = PatchSampler(
        kernel_size=kernel_size,
        scale=scale,
        num_shards=num_shards_per_area,
        num_samples_per_shard=num_samples_per_shard,
    )

    # Define export parameters
    bands = SAR_EXPORT_BANDS + MSO_EXPORT_BANDS + RESPONSES
    bucket = "ai4good-3b"
    export_folder = "_".join(
        [
            name,
            area_of_interest,
            start_date.replace("-", ""),
            end_date.replace("-", ""),
            f"scale-{scale}",
            "db" if log_scale else "float",
            f"{months_before_acquisition}mba",
            f"{num_samples_per_area}spa",
        ]
    )

    # Get Sentinel-1
    before_end_date = start_date_ee.advance(-(months_before_acquisition + 1), "month")
    before_start_date = before_end_date.advance(-1, "month")
    s1_before = _get_s1(before_start_date, before_end_date, log_scale).sort(
        "system:time_start"
    )
    s1_after = _get_s1(start_date_ee, end_date_ee, log_scale).sort("system:time_start")

    # Now loop over the aoi polygons
    train_aoi_polygons_list = train_aoi_polygons.toList(train_aoi_polygons.size())
    for g in range(train_aoi_polygons.size().getInfo()):
        sample_geometry = ee.Feature(train_aoi_polygons_list.get(g)).geometry()
        # Get before and after image
        before_image = s1_before.filter(
            ee.Filter.contains(".geo", sample_geometry)
        ).first()
        after_image = s1_after.filter(
            ee.Filter.contains(".geo", sample_geometry)
        ).first()

        # Create the difference mask
        difference_mask = _make_deforestation_mask(
            after_image,
            deforest_polygons_ee,
            months_before_acquisition,
        )

        # Make the multispectral optical composite
        multispectral_optical = _make_multispectral_optical_composite(sample_geometry)

        # Now merge everything
        time_series = _make_time_series_with_groundtruth(
            before_image,
            after_image,
            multispectral_optical,
            difference_mask,
            sample_geometry,
        )

        # Sample patches from the time series
        samples = patch_sampler.sample_image(time_series, sample_geometry)

        # Export as TFRecord
        desc = name + "_" + str(g)
        task = ee.batch.Export.table.toCloudStorage(
            collection=samples,
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

    # TODO: instead of using cli options read the params below from a config

    parser = argparse.ArgumentParser(
        description=desc,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="The name to prepend to the dataset directory",
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
        "--log-scale",
        action="store_true",
        help="Use log scale backscattering values.",
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
        "--scale",
        type=int,
        default=40,
        choices=[10, 30, 40],
        help="Export scale in meters per pixel",
    )

    parser.add_argument(
        "--kernel-size",
        type=int,
        default=256,
        help="Size of the exported patches.",
    )

    parser.add_argument(
        "--num-samples-per-area",
        type=int,
        default=100,
        help="Number of patches generated from each sampling area.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    create_deforestation_detection_dataset(
        name=args.name,
        start_date=args.start_date,
        end_date=args.end_date,
        log_scale=args.log_scale,
        area_of_interest=args.area_of_interest,
        months_before_acquisition=args.months_before_acquisition,
        scale=args.scale,
        kernel_size=args.kernel_size,
        num_samples_per_area=args.num_samples_per_area,
    )
