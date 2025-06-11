import planetary_computer as pc
import pystac_client
import rasterio
import rasterio.features
import rasterio.warp
from pystac.extensions.eo import EOExtension as eo

RASTERIO_BEST_PRACTICES = dict(  # See https://github.com/pangeo-data/cog-best-practices
    CURL_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_MAX_RAW_BLOCK_CACHE_SIZE="200000000",
    GDAL_SWATH_SIZE="200000000",
    VSI_CURL_CACHE_SIZE="200000000",
)


CATALOG = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace
)


def read_band(href, polygon, src_crs="EPSG:4326"):
    with rasterio.open(href) as f:
        bounds = rasterio.features.bounds(polygon)
        bounds = rasterio.warp.transform_bounds(src_crs, f.crs, *bounds)
        window = rasterio.windows.from_bounds(transform=f.transform, *bounds)
        x = f.read(1, window=window)
    return x


def write(path, x, profile, metadata=None):
    with rasterio.open(path, "w", **profile) as f:
        if metadata:
            f.update_tags(**metadata)
        f.write(x)
    return


def query(
    geometry,
    dates="2018-02-01/2018-10-30",
    keys=["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"],
    min_size=(20, 20),
):
    collection = "landsat-8-c2-l2"
    search = CATALOG.search(
        collections=[collection],
        intersects=geometry,
        datetime=dates,
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.items())

    if len(items) == 0:
        return 0

    if eo.ext(items[0]).cloud_cover is not None:
        # Sort images by cloud cover
        items = sorted(items, key=lambda item: eo.ext(item).cloud_cover)
    else:
        # If cloud cover is not an option, sample most recent
        items = reversed(items)

    hrefs = [{k: pc.sign(item.assets[k].href) for k in keys} for item in items]

    # Find 1st images which satisfies min height/width requirement
    valid = []
    for href in hrefs:
        b = read_band(href["SR_B2"], geometry, src_crs="EPSG:4326")
        valid.append(b.shape[0] >= min_size[0] and b.shape[1] >= min_size[1])

    if not any(valid):
        return -1
    else:
        idx = valid.index(True)
        href = hrefs[idx]

    return href
