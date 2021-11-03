import ee
import numpy as np

def make_linearly_spaced_grid(
    top_left: ee.Geometry.Point,
    bottom_right: ee.Geometry.Point,
    n_lon: int,
    n_lat: int,
):
    lon_start, lat_start = top_left.coordinates().getInfo()
    lon_end, lat_end = bottom_right.coordinates().getInfo()

    grid_points_lon = np.linspace(lon_start, lon_end, n_lon)
    grid_points_lat = np.linspace(lat_start, lat_end, n_lat)

    cells = []
    cell_id = 0
    for tl_lon, br_lon in zip(grid_points_lon, grid_points_lon[1:]):
        for tl_lat, br_lat in zip(grid_points_lat, grid_points_lat[1:]):
            tl = ee.Geometry.Point(tl_lon, tl_lat)
            br = ee.Geometry.Point(br_lon, br_lat)
            cell_id = cell_id + 1
            cells.append(
                ee.Feature(ee.Geometry.Rectangle([tl, br]),
                           {"label": cell_id}))

    return ee.FeatureCollection(cells)


def make_grid_over_roi(roi: ee.Geometry.Polygon, n_lon: int, n_lat: int):
    # Get the bounds of the roi polygon
    roi_bounds_coords = roi.bounds().coordinates().get(0).getInfo()
    # Get top-left and bottom-right corners of the grid
    grid_top_left = ee.Geometry.Point(roi_bounds_coords[0])
    grid_bottom_right = ee.Geometry.Point(roi_bounds_coords[2])
    # Generate the grid
    return make_linearly_spaced_grid(grid_top_left, grid_bottom_right, n_lon,
                                     n_lat)