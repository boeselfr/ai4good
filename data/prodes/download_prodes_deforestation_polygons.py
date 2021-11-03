import argparse
import requests, zipfile, io, os

from pathlib import Path
from typing import List

_MIN_VALID_YEAR = 2005
_MAX_VALID_YEAR = 2018


class ParseYearsList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        years = []
        for y in values.split(','):
            year = int(y)
            assert year >= _MIN_VALID_YEAR and year <= _MAX_VALID_YEAR
            years.append(year)
        setattr(namespace, self.dest, years)


def get_url(year):
    # url for shapefile of amazon for a specific year (accumulated)
    url = "http://www.dpi.inpe.br/prodesdigital/dadosn/mosaicos/" + year + "/PDigital2000_" + year + "_AMZ_shp.zip"
    return url


def download_prodes_deforestation_polygons(
    years: List[int],
    output_dir_path: Path,
):
    for year in years:
        url = get_url(str(year))
        print(f'downloading data from: {url}')
        dl_dir_path = output_dir_path / str(year)
        dl_dir_path.mkdir(exist_ok=False, parents=True)
        r = requests.get(url, allow_redirects=True, stream=True)
        # save zip file
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(str(dl_dir_path) + '/')


def _parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Donwload PRODES deforestation warning polygons .shp files for the selected years"
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        type=lambda p: Path(p).absolute(),
        help="Path to the dir where the data should be downloaded.")

    parser.add_argument(
        "-ys",
        "--years",
        required=True,
        type=str,
        action=ParseYearsList,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_prodes_deforestation_polygons(
        years=args.years,
        output_dir_path=args.output_dir,
    )