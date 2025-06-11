import os

env = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_REQUEST_PAYER="requester",
    AWS_DEFAULT_REGION="us-west-2",
    AWS_ACCESS_KEY_ID=os.environ["AWS_ACCESS_KEY_ID"],
    AWS_SECRET_ACCESS_KEY=os.environ["AWS_SECRET_ACCESS_KEY"],
    AWS_PROFILE="ieee",
)
os.environ.update(env)

import argparse
import multiprocessing as mp
import time
from functools import partial

import geopandas as gpd
import pandas as pd
import pystac_client
from odc.stac import stac_load
from pystac.extensions.eo import EOExtension as eo
from tqdm import tqdm

RASTERIO_BEST_PRACTICES = dict(  # See https://github.com/pangeo-data/cog-best-practices
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE="200000000",
    GDAL_SWATH_SIZE="200000000",
    VSI_CURL_CACHE_SIZE="200000000",
)

os.environ.update(RASTERIO_BEST_PRACTICES)


url = "https://landsatlook.usgs.gov/stac-server/"
catalog = pystac_client.Client.open(url)


def query(geometry, dates="2018-02-01/2018-10-30"):
    query = catalog.search(
        collections=["landsat-c2l2-sr"],
        datetime=dates,
        intersects=geometry,
        query={"eo:cloud_cover": {"lt": 20}, "platform": {"eq": "LANDSAT_8"}},
    )

    items = list(query.items())

    if len(items) == 0:
        return 0

    if eo.ext(items[0]).cloud_cover is not None:
        # Sort images by cloud cover
        items = sorted(query.items(), key=lambda x: x.properties["eo:cloud_cover"])
    else:
        # If cloud cover is not an option, sample most recent
        items = reversed(items)

    item = items[0]
    for asset in item.assets:
        item.assets[asset].href = item.assets[asset].href.replace(
            "https://landsatlook.usgs.gov/data/", "s3://usgs-landsat/"
        )

    return item


def worker(inputs, out):
    geometry, image, date = inputs
    output_path = os.path.join(out, f"{image}.tif")

    if os.path.exists(output_path):
        return image, None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Query data
    try:
        item = query(geometry, dates=date)
    except Exception:
        # typically a request error causes failure, try waiting and querying again
        time.sleep(5)
        item = query(geometry, dates=date)

    # If no images available then skip
    if item == 0:
        return image, "No images available for area"
    elif item == -1:
        return image, "No images met min size requirement"

    # Read data from urls to array
    bands = ["coastal", "blue", "green", "red", "nir08", "swir16", "swir22"]
    x = stac_load([item], bands=bands, resolution=30, intersects=geometry)
    x = x.isel(time=0).to_array("band")

    # Write to geotiff
    x.rio.to_raster(
        output_path, driver="GTiff", transform=x.rio.transform(), dtype="uint16", compression="LZW"
    )
    return image, None


def main(args):
    months = 2
    df = gpd.read_parquet(args.data_file)
    df["date"] = pd.to_datetime(df["date"])
    date_start = (df["date"] - pd.DateOffset(months=months)).dt.strftime("%Y-%m-%d")
    date_end = (df["date"] + pd.DateOffset(months=months)).dt.strftime("%Y-%m-%d")
    df["date_range"] = date_start + "/" + date_end

    os.makedirs(args.output, exist_ok=True)

    with open(args.log_file, "w") as f:
        f.write("filename,error\n")

    values = zip(df["geometry"], df["image"], df["date_range"], strict=False)
    func = partial(worker, out=args.output)
    with mp.Pool(processes=args.workers) as p:
        pool = p.imap(func, values)
        for filename, error in tqdm(pool, total=len(df), position=0, leave=False):
            if error is not None:
                with open(args.log_file, "a") as f:
                    f.write(f"{filename},{error}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/bigearthnet.parquet",
        help="File containing dataset metadata.",
    )
    parser.add_argument(
        "--output", type=str, default="data/bigearthnet-l", help="Folder to save dataset to."
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="bigearthnet-logs.csv",
        help="Log file to save images with errors.",
    )
    parser.add_argument("--workers", type=int, default=4, help="Num parallel download workers.")
    args = parser.parse_args()
    main(args)
