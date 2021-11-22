from typing import List, Union

import ee


def build_sentinel_1_image_collection(
    aoi: Union[ee.FeatureCollection, ee.Geometry.Polygon],
    start_date: ee.Date,
    end_date: ee.Date,
    instrument_mode: str = 'IW',
    bands: List[str] = ['VV', 'VH'],
    platform_number: str = None,
    orbit_pass: str = None,
):
    # First get the collection and filter by date
    coll = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate(
        start_date, end_date)

    # Filter by instrument mode
    assert instrument_mode in ["IW"]
    coll = coll.filter(ee.Filter.eq("instrumentMode", instrument_mode))

    for band in bands:
        assert band in ['VV', 'VH', 'HH', 'HV']
        coll = coll.filter(
            ee.Filter.listContains('transmitterReceiverPolarisation', band))

    # Optionally filter by metadata
    if platform_number is not None:
        assert platform_number in ['A', 'B'], "Platform number must be A or B"
        coll = coll.filter(ee.Filter.eq("platform_number", platform_number))
    if orbit_pass is not None:
        assert orbit_pass in ['ASCENDING', 'DESCENDING'
                              ], "Orbit pass must ASCENDING or DESCENDING"
        coll = coll.filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    # Filter by bounds
    coll = coll.filterBounds(aoi)

    return coll


def build_sentinel_2_image_collection(
    aoi: ee.Geometry.Polygon,
    start_date: ee.Date,
    end_date: ee.Date,
    max_cloud_percent: int = 60,
):
    assert max_cloud_percent >= 0 and max_cloud_percent <= 100
    return (ee.ImageCollection("COPERNICUS/S2_SR").filterDate(
        start_date, end_date).filterDate(start_date, end_date).filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',
                         max_cloud_percent)).filterBounds(aoi))
