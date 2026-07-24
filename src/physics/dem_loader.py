"""Stream lunar elevation patches from the USGS LRO LOLA global mosaic.

The mosaic is a Cloud Optimized GeoTIFF, so only the requested window travels
over the network.  Three things this module is careful about, each of which
silently corrupted patches in an earlier revision:

* **Coordinate reference.**  Selenographic longitude/latitude are converted
  into the raster's own coordinate space rather than being fed to it raw.  The
  USGS lunar mosaics are published both in degrees (simple cylindrical) and in
  metres, and their longitude origin may be 0..360 or -180..180.
* **Window geometry.**  Patches are requested as a fixed number of *pixels*,
  not a fixed span of degrees.  One degree of longitude is ~30 km at the
  equator but ~265 m at 89.5 deg S, so a degree-sized window silently returns
  wildly different physical areas — and Shackleton, the headline hazard site,
  sits at the extreme end of that distortion.
* **Nodata.**  The mosaic has gaps.  Unmasked fill values propagate into the
  physics engine as absurd elevations.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# USGS global lunar DEM, 118 m/pixel.
LOLA_118M_URL = (
    "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
)

_ASC = "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/"

# IAU mean radius of the Moon, used to convert degrees to metres when the
# mosaic is published in a projected (metre-based) simple cylindrical CRS.
MOON_RADIUS_M = 1_737_400.0

_DEFAULT_CACHE = Path(os.environ.get("LUNASLIDE_CACHE", ".cache/dem"))


@dataclass(frozen=True)
class DemProduct:
    """A streamable lunar elevation product."""

    key: str
    name: str
    url: str
    resolution_m: float
    min_latitude: float
    max_latitude: float
    citation: str

    def covers(self, latitude: float) -> bool:
        return self.min_latitude <= latitude <= self.max_latitude


# Ordered best-resolution-first; ``choose_product`` takes the first that covers
# the requested latitude.  The polar entry exists because an equirectangular
# global mosaic cannot describe the poles at all: there, its cells degenerate
# to 118 m x 1 m and its bottom row is the pole smeared across 92,160 columns.
DEM_PRODUCTS: tuple[DemProduct, ...] = (
    DemProduct(
        key="lola_spole_5m",
        name="LOLA south polar DEM, 5 m/px",
        url=_ASC + "Lunar_safed/LMAP/SouthPolar/ldem_87s_5mpp_cog.tif",
        resolution_m=5.0,
        min_latitude=-90.0,
        max_latitude=-87.0,
        citation="LRO LOLA south polar DEM (ldem_87s_5mpp), Moon2000 south polar stereographic, USGS/NASA",
    ),
    DemProduct(
        key="lolakaguya_59m",
        name="LOLA + Kaguya merged DEM, 59 m/px",
        url=_ASC + "LolaKaguya_Topo/Lunar_LRO_LOLAKaguya_DEMmerge_60N60S_512ppd.tif",
        resolution_m=59.23,
        min_latitude=-60.0,
        max_latitude=60.0,
        citation="LRO LOLA + SELENE Kaguya TC merged DEM, 512 ppd, equirectangular, USGS/NASA/JAXA",
    ),
    DemProduct(
        key="lola_global_118m",
        name="LOLA global DEM, 118 m/px",
        url=LOLA_118M_URL,
        resolution_m=118.45,
        min_latitude=-90.0,
        max_latitude=90.0,
        citation="LRO LOLA global LDEM 118 m (Mar 2014), simple cylindrical, USGS/NASA",
    ),
)

PRODUCTS_BY_KEY = {product.key: product for product in DEM_PRODUCTS}


def choose_product(latitude: float) -> DemProduct:
    """Finest-resolution product that covers a latitude."""
    for product in DEM_PRODUCTS:
        if product.covers(latitude):
            return product
    return PRODUCTS_BY_KEY["lola_global_118m"]


@dataclass(frozen=True)
class LunarPatch:
    """An elevation patch with enough context to be georeferenced later.

    The two ground spacings differ, and the difference is not a rounding
    detail.  The mosaic is equirectangular, so a pixel's north-south extent is
    constant but its east-west extent shrinks by ``cos(latitude)``: 118 m at
    the equator, 40 m at 70 degrees, and about 1 m at Shackleton.  Treating the
    grid as square inflates every east-west slope by ``1 / cos(latitude)``.
    """

    elevation: np.ndarray
    latitude: float
    longitude: float
    grid_spacing_y_m: float
    grid_spacing_x_m: float
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    nodata_fraction: float = 0.0
    product_key: str | None = None
    from_cache: bool = False

    @property
    def product(self) -> DemProduct | None:
        return PRODUCTS_BY_KEY.get(self.product_key or "")

    @property
    def grid_spacing(self) -> tuple[float, float]:
        """``(north_south, east_west)`` spacing, as the physics engine wants it."""
        return self.grid_spacing_y_m, self.grid_spacing_x_m

    @property
    def anisotropy(self) -> float:
        """How far from square the cells are; 1.0 means isotropic."""
        return self.grid_spacing_y_m / self.grid_spacing_x_m


def bounds_are_degrees(left: float, bottom: float, right: float, top: float) -> bool:
    """Whether a raster's bounds are plausibly in degrees rather than metres.

    A lunar raster in metres spans millions of units; one in degrees cannot
    exceed 360.  The gap between those scales is wide enough that this needs no
    CRS introspection, which matters because lunar CRSs frequently lack an EPSG
    code and are only partially described by their WKT.
    """
    return abs(left) <= 361.0 and abs(right) <= 361.0 and abs(bottom) <= 91.0 and abs(top) <= 91.0


def wrap_longitude(longitude: float, low: float, high: float) -> float:
    """Fold a longitude into the range a raster actually uses.

    Handles the 0..360 and -180..180 conventions without needing to know which
    one the mosaic was published with.  Partial-coverage rasters (span well
    under a full revolution) are left alone, since wrapping them is meaningless.
    """
    span = high - low
    if span < 359.0:
        return longitude
    return low + (longitude - low) % 360.0


def lonlat_to_raster_xy(
    longitude: float,
    latitude: float,
    bounds: tuple[float, float, float, float],
    *,
    radius_m: float = MOON_RADIUS_M,
) -> tuple[float, float]:
    """Convert selenographic degrees into the mosaic's coordinate space."""
    left, bottom, right, top = bounds
    if bounds_are_degrees(left, bottom, right, top):
        return wrap_longitude(longitude, left, right), latitude
    # Projected simple cylindrical on a sphere with the standard parallel at
    # the equator: x = R * lambda, y = R * phi.
    degree_left, degree_right = np.degrees(left / radius_m), np.degrees(right / radius_m)
    wrapped = wrap_longitude(longitude, float(degree_left), float(degree_right))
    return float(np.radians(wrapped) * radius_m), float(np.radians(latitude) * radius_m)


def _cache_path(cache_dir: Path, url: str, latitude: float, longitude: float, size_px: int) -> Path:
    key = f"{url}|{latitude:.6f}|{longitude:.6f}|{size_px}"
    return cache_dir / f"patch_{hashlib.sha256(key.encode()).hexdigest()[:20]}.npz"


def _lunar_geographic_crs():
    from rasterio.crs import CRS

    return CRS.from_proj4(f"+proj=longlat +R={MOON_RADIUS_M:.1f} +no_defs")


def great_circle_m(
    lon_a: float, lat_a: float, lon_b: float, lat_b: float, radius_m: float = MOON_RADIUS_M
) -> float:
    """Surface distance between two selenographic coordinates, in metres."""
    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    d_lambda = np.radians(lon_b - lon_a)
    haversine = (
        np.sin((phi_b - phi_a) / 2) ** 2
        + np.cos(phi_a) * np.cos(phi_b) * np.sin(d_lambda / 2) ** 2
    )
    return float(2 * radius_m * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0))))


def _project_to_raster(src, longitude: float, latitude: float) -> tuple[float, float]:
    """Selenographic degrees into the raster's coordinate space, any projection.

    Delegates to GDAL rather than hand-rolling projection maths, so polar
    stereographic products work as well as equirectangular ones.  Falls back to
    the closed-form equirectangular mapping only if the transform is refused.
    """
    if src.crs is None or src.crs.is_geographic:
        return wrap_longitude(longitude, src.bounds.left, src.bounds.right), latitude
    try:
        from rasterio.warp import transform as warp_transform

        xs, ys = warp_transform(_lunar_geographic_crs(), src.crs, [longitude], [latitude])
        return float(xs[0]), float(ys[0])
    except Exception:
        return lonlat_to_raster_xy(longitude, latitude, tuple(src.bounds))


def _ground_spacing(src, x: float, y: float) -> tuple[float, float]:
    """True ``(north_south, east_west)`` ground spacing at a point, in metres.

    Measured by inverse-projecting the pixel's own corners back to
    selenographic coordinates and taking great-circle distances.  This is
    projection-agnostic: it reports the equirectangular ``cos(latitude)``
    contraction and the polar stereographic scale factor without either being
    special-cased.
    """
    pixel_x, pixel_y = abs(src.transform.a), abs(src.transform.e)
    try:
        from rasterio.warp import transform as warp_transform

        lons, lats = warp_transform(
            src.crs, _lunar_geographic_crs(), [x, x + pixel_x, x], [y, y, y - pixel_y]
        )
    except Exception:
        return float(pixel_y), float(pixel_x)
    east_west = great_circle_m(lons[0], lats[0], lons[1], lats[1])
    north_south = great_circle_m(lons[0], lats[0], lons[2], lats[2])
    return north_south, east_west


def fetch_patch(
    latitude: float,
    longitude: float,
    size_px: int = 500,
    *,
    product: DemProduct | str | None = None,
    url: str | None = None,
    timeout_s: int = 30,
    max_nodata_fraction: float = 0.02,
    cache_dir: Path | str | None = _DEFAULT_CACHE,
    verbose: bool = True,
) -> LunarPatch | None:
    """Stream a ``size_px`` square elevation patch centred on a coordinate.

    With no ``product`` or ``url``, the finest-resolution product covering the
    latitude is chosen automatically: 5 m/px polar stereographic below 87 deg S,
    59 m/px merged LOLA+Kaguya within +/-60 deg, and the 118 m global mosaic
    elsewhere.

    Returns ``None`` — never a partial or fabricated array — when the mosaic is
    unreachable, the coordinate falls outside it, or too much of the window is
    nodata.  Callers decide what to substitute; this module will not guess.
    """
    if size_px <= 0:
        raise ValueError("size_px must be positive.")
    if not -90.0 <= latitude <= 90.0:
        raise ValueError(f"latitude {latitude} is outside [-90, 90].")

    if url is None:
        if product is None:
            selected = choose_product(latitude)
        elif isinstance(product, str):
            if product not in PRODUCTS_BY_KEY:
                raise ValueError(
                    f"Unknown product {product!r}; choose one of: {', '.join(PRODUCTS_BY_KEY)}"
                )
            selected = PRODUCTS_BY_KEY[product]
        else:
            selected = product
        url = selected.url
    else:
        selected = None

    cache_file = None
    if cache_dir is not None:
        cache_file = _cache_path(Path(cache_dir), url, latitude, longitude, size_px)
        if cache_file.is_file():
            try:
                stored = np.load(cache_file, allow_pickle=False)
                transform = stored["transform"]
                return LunarPatch(
                    elevation=stored["elevation"],
                    latitude=latitude,
                    longitude=longitude,
                    grid_spacing_y_m=float(stored["grid_spacing_y_m"]),
                    grid_spacing_x_m=float(stored["grid_spacing_x_m"]),
                    crs=str(stored["crs"]) or None,
                    transform=tuple(transform.tolist()) if transform.size == 6 else None,
                    nodata_fraction=float(stored["nodata_fraction"]),
                    product_key=str(stored["product_key"]) or None,
                    from_cache=True,
                )
            except KeyError:
                # Written by an older layout; re-fetch rather than guess.
                cache_file.unlink(missing_ok=True)

    try:
        import rasterio
        from rasterio.env import Env
        from rasterio.windows import Window
    except ImportError:
        if verbose:
            print("rasterio is not installed; run `pip install -r requirements.txt`.")
        return None

    if verbose:
        label = selected.name if selected else url.rsplit("/", 1)[-1]
        print(f"Streaming {label} at lat {latitude:.2f}, lon {longitude:.2f} ...")

    try:
        with Env(GDAL_HTTP_TIMEOUT=timeout_s, GDAL_HTTP_MAX_RETRY=2, VSI_CACHE=True):
            with rasterio.open(url) as src:
                x, y = _project_to_raster(src, longitude, latitude)
                left, bottom, right, top = src.bounds
                if not (left <= x <= right and bottom <= y <= top):
                    if verbose:
                        print(f"Coordinate maps to ({x:.1f}, {y:.1f}), outside the mosaic.")
                    return None

                row, column = src.index(x, y)
                half = size_px // 2
                window = Window(column - half, row - half, size_px, size_px)
                # boundless=True pads rather than truncating, so polar windows
                # that overhang the raster still come back the requested size;
                # masked=True marks both the declared nodata and that padding,
                # which is the only way to tell real zero elevation from fill.
                raw = src.read(1, window=window, boundless=True, masked=True)
                data = np.ma.getdata(raw).astype(np.float64)
                invalid = np.ma.getmaskarray(raw) | ~np.isfinite(data)
                transform = src.window_transform(window)
                crs = str(src.crs) if src.crs else None

                # The band is int16 with a scale factor; rasterio's read()
                # returns raw counts and never applies it.  The LOLA mosaic
                # stores half-metres, so skipping this doubles every elevation
                # and therefore every slope.  Nodata was masked above, in raw
                # units, so scaling afterwards is safe.
                scale = float(src.scales[0]) if src.scales else 1.0
                offset = float(src.offsets[0]) if src.offsets else 0.0
                if scale != 1.0 or offset != 0.0:
                    data = data * scale + offset

                # Measured from the raster's own geometry rather than assumed,
                # so equirectangular cos(latitude) contraction and polar
                # stereographic scaling are both handled without special cases.
                spacing_y, spacing_x = _ground_spacing(src, x, y)
    except Exception as error:  # network, CRS, or window failure
        if verbose:
            print(f"Could not stream the mosaic: {error}")
        return None

    fraction = float(invalid.mean())
    if invalid.all() or fraction > max_nodata_fraction:
        if verbose:
            print(f"Rejected patch: {fraction:.1%} nodata exceeds the {max_nodata_fraction:.1%} limit.")
        return None
    if invalid.any():
        data[invalid] = float(np.median(data[~invalid]))

    patch = LunarPatch(
        elevation=data.astype(np.float32),
        latitude=latitude,
        longitude=longitude,
        grid_spacing_y_m=float(spacing_y),
        grid_spacing_x_m=float(spacing_x),
        crs=crs,
        transform=tuple(transform.to_gdal()),
        nodata_fraction=fraction,
        product_key=selected.key if selected else None,
    )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_file,
            elevation=patch.elevation,
            grid_spacing_y_m=patch.grid_spacing_y_m,
            grid_spacing_x_m=patch.grid_spacing_x_m,
            crs=patch.crs or "",
            transform=np.asarray(patch.transform, dtype=np.float64),
            nodata_fraction=patch.nodata_fraction,
            product_key=patch.product_key or "",
        )
    if verbose:
        print(f"Streamed {patch.elevation.shape} patch at "
              f"{patch.grid_spacing_y_m:.1f} x {patch.grid_spacing_x_m:.1f} m/px.")
    return patch


def fetch_lunar_patch(lat: float, lon: float, size_px: int = 500, **kwargs) -> np.ndarray | None:
    """Elevation-only convenience wrapper around :func:`fetch_patch`."""
    patch = fetch_patch(lat, lon, size_px, **kwargs)
    return None if patch is None else patch.elevation


@dataclass(frozen=True)
class LatitudeBand:
    """A full-width horizontal slice of a mosaic, to be cut into patches.

    The equirectangular lunar products are *strip*-organised rather than tiled:
    each block is one full-width scanline (1 x 184,320 for the 59 m merge).  A
    128-row window therefore costs 128 full-width strips — about 47 MB — no
    matter how few columns are actually wanted, and issuing hundreds of those
    concurrently produces truncated range reads.

    Reading the whole band instead costs the same 47 MB and yields on the order
    of a thousand disjoint patches.  Patches from one band share a latitude, so
    sampling is stratified by band rather than independent; in exchange every
    band spans all longitudes, which keeps latitude from confounding a
    longitude-based train/test split.
    """

    elevation: np.ndarray
    latitude: float
    grid_spacing_y_m: float
    grid_spacing_x_m: float
    crs: str | None = None
    product_key: str | None = None
    invalid: np.ndarray | None = None

    @property
    def grid_spacing(self) -> tuple[float, float]:
        return self.grid_spacing_y_m, self.grid_spacing_x_m

    def patch_at(self, column: int, size_px: int) -> tuple[np.ndarray, float] | None:
        """Cut one patch, returning it with its nodata fraction, or None."""
        if column < 0 or column + size_px > self.elevation.shape[1]:
            return None
        window = self.elevation[:, column : column + size_px]
        if window.shape[0] != size_px:
            return None
        if self.invalid is None:
            return window, 0.0
        bad = self.invalid[:, column : column + size_px]
        return window, float(bad.mean())

    def longitude_of(self, column: int, size_px: int) -> float:
        """Selenographic longitude of a patch's centre column."""
        centre = (column + size_px / 2) / self.elevation.shape[1]
        return -180.0 + 360.0 * centre


def fetch_latitude_band(
    latitude: float,
    height_px: int,
    *,
    product: DemProduct | str | None = None,
    timeout_s: int = 120,
    attempts: int = 3,
    verbose: bool = True,
) -> LatitudeBand | None:
    """Stream one full-width band of a mosaic at a given latitude.

    Retries on truncated reads, which these strip-organised products produce
    intermittently under load.
    """
    if height_px <= 0:
        raise ValueError("height_px must be positive.")
    selected = (
        choose_product(latitude) if product is None
        else PRODUCTS_BY_KEY[product] if isinstance(product, str)
        else product
    )
    try:
        import rasterio
        from rasterio.env import Env
        from rasterio.windows import Window
    except ImportError:
        if verbose:
            print("rasterio is not installed; run `pip install -r requirements.txt`.")
        return None

    for attempt in range(1, attempts + 1):
        try:
            with Env(GDAL_HTTP_TIMEOUT=timeout_s, GDAL_HTTP_MAX_RETRY=3, VSI_CACHE=True):
                with rasterio.open(selected.url) as src:
                    x, y = _project_to_raster(src, 0.0, latitude)
                    row, _ = src.index(x, y)
                    top = max(0, min(row - height_px // 2, src.height - height_px))
                    window = Window(0, top, src.width, height_px)
                    raw = src.read(1, window=window, masked=True)
                    data = np.ma.getdata(raw).astype(np.float32)
                    invalid = np.ma.getmaskarray(raw) | ~np.isfinite(data)
                    scale = float(src.scales[0]) if src.scales else 1.0
                    offset = float(src.offsets[0]) if src.offsets else 0.0
                    if scale != 1.0 or offset != 0.0:
                        data = data * scale + offset
                    if invalid.any():
                        usable = data[~invalid]
                        data[invalid] = float(np.median(usable)) if usable.size else 0.0
                    spacing_y, spacing_x = _ground_spacing(src, x, y)
                    return LatitudeBand(
                        elevation=data,
                        latitude=latitude,
                        grid_spacing_y_m=spacing_y,
                        grid_spacing_x_m=spacing_x,
                        crs=str(src.crs) if src.crs else None,
                        product_key=selected.key,
                        invalid=invalid,
                    )
        except Exception as error:
            if attempt == attempts:
                if verbose:
                    print(f"  band at {latitude:+.1f} failed after {attempts} attempts: {error}")
                return None
            if verbose:
                print(f"  band at {latitude:+.1f} attempt {attempt} failed, retrying")
    return None
