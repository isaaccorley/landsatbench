import argparse
import multiprocessing as mp
import os
import sys
import time
from functools import partial

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

sys.path.append(".")
from landsatbench.utils import RASTERIO_BEST_PRACTICES, query, read_band, write

os.environ.update(RASTERIO_BEST_PRACTICES)


def worker(inputs, out, min_size):
    geometry, folder, date = inputs
    output_path = os.path.join(out, folder, f"{folder}.tif")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Query data
    try:
        hrefs = query(geometry, dates=date, min_size=min_size)
    except Exception:
        # typically a request error causes failure, try waiting and querying again
        time.sleep(5)
        hrefs = query(geometry, dates=date, min_size=min_size)

    # If no images available then skip
    if hrefs == 0:
        return folder, "No images available for area"
    elif hrefs == -1:
        return folder, "No images met min size requirement"

    # Read data from urls to array
    x = np.stack([read_band(href, geometry, src_crs="EPSG:4326") for _, href in hrefs.items()])

    # Write to geotiff
    with rasterio.open(hrefs["SR_B1"]) as f:
        profile = f.profile.copy()

    profile["count"], profile["width"], profile["height"] = x.shape
    metadata = {band: href.split("?")[0] for band, href in hrefs.items()}
    write(output_path, x, profile, metadata)
    return folder, None


def main(args):
    months = 6
    df = gpd.read_parquet(args.data_file)
    df["date"] = pd.to_datetime(df["date"])
    date_start = (df["date"] - pd.DateOffset(months=months)).dt.strftime("%Y-%m-%d")
    date_end = (df["date"] + pd.DateOffset(months=months)).dt.strftime("%Y-%m-%d")
    df["date_range"] = date_start + "/" + date_end

    """
    import pandas as pd
    logs = pd.read_csv(args.log_file)["filename"].tolist()
    indices = df["index"].isin(logs)
    df = df[indices]
    print(f"Found {len(df)} images to download.")
    """

    with open(args.log_file, "w") as f:
        f.write("filename,error\n")

    values = zip(df["geometry"], df["image"], df["date_range"], strict=False)
    func = partial(worker, out=args.output, min_size=(38, 38))
    with mp.Pool(processes=16) as p:
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
        "--output",
        type=str,
        default="/workspace/storage/data/bigearthnet-l",
        help="Folder to save dataset to.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="bigearthnet-logs.csv",
        help="Log file to save images with errors.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Num parallel download workers.")
    args = parser.parse_args()
    main(args)
