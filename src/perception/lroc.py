"""Acquire and tile real LROC NAC imagery for Stage 1 training.

The USGS/WUSTL Orbital Data Explorer (ODE) REST API indexes LROC Narrow Angle
Camera products.  Each calibrated product (``pt=CDRNAC4``) ships a pyramided
GeoTIFF (``*_pyr.tif``) that GDAL reads directly, so no ISIS/SPICE processing is
needed to get usable pixels.

Two honest constraints:

* **These strips are not map-projected.**  A NAC CDR ``_pyr.tif`` has no CRS —
  only a centre latitude/longitude.  That is fine for training the enhancer,
  which learns appearance, not position.  Georeferenced evidence for Stage 3
  needs map-projected products and is a separate concern.
* **ODE's spatial box filter is loose.**  It returns products whose footprint
  merely overlaps a wide region, so targeting by coordinate is done here,
  client-side, on each product's reported centre.

Pure logic (product parsing, tiling, content filtering) is separated from the
network so it can be tested offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ODE_URL = "https://oderest.rsl.wustl.edu/live2/"
NAC_PRODUCT_TYPE = "CDRNAC4"


@dataclass(frozen=True)
class NacProduct:
    """A LROC NAC product with a directly-readable pyramided GeoTIFF."""

    product_id: str
    center_lat: float
    center_lon: float
    pyr_tif_url: str
    kbytes: float

    def distance_deg(self, lat: float, lon: float) -> float:
        """Great-circle distance from a target, in degrees of arc."""
        d_lon = min(abs(self.center_lon - lon), 360 - abs(self.center_lon - lon))
        return math.hypot(self.center_lat - lat, d_lon)


def parse_products(ode_json: dict) -> list[NacProduct]:
    """Extract NAC products carrying a ``_pyr.tif`` from an ODE response.

    Tolerates ODE's habit of collapsing single-element lists into a bare object.
    """
    results = ode_json.get("ODEResults", {})
    if results.get("Status") == "ERROR":
        return []
    raw = results.get("Products", {}).get("Product", [])
    if isinstance(raw, dict):
        raw = [raw]

    products: list[NacProduct] = []
    for entry in raw:
        files = entry.get("Product_files", {}).get("Product_file", [])
        if isinstance(files, dict):
            files = [files]
        pyr = next(
            (f for f in files if str(f.get("URL", "")).lower().endswith("_pyr.tif")),
            None,
        )
        if pyr is None:
            continue
        try:
            products.append(
                NacProduct(
                    product_id=str(entry.get("pdsid", "unknown")),
                    center_lat=float(entry.get("Center_latitude", "nan")),
                    center_lon=float(entry.get("Center_longitude", "nan")),
                    pyr_tif_url=str(pyr["URL"]),
                    kbytes=float(pyr.get("KBytes", 0) or 0),
                )
            )
        except (TypeError, ValueError):
            continue
    return products


def is_informative(tile: np.ndarray, min_std: float = 12.0, max_fill_fraction: float = 0.15) -> bool:
    """Whether a tile carries real surface texture rather than margin or fill.

    NAC strips have black borders and occasional saturated bands.  A tile is
    kept only if little of it is fill (0 or 255) and it has enough contrast to
    contain structure — training an enhancer on flat black teaches it nothing.
    """
    fill = np.mean((tile == 0) | (tile == 255))
    return fill <= max_fill_fraction and float(tile.std()) >= min_std


def iter_tiles(
    image: np.ndarray, size_px: int, stride: int | None = None
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield ``(tile, row, col)`` windows across an image.

    Partial windows at the right and bottom edges are dropped, so every tile is
    exactly ``size_px`` square.
    """
    if size_px <= 0:
        raise ValueError("size_px must be positive.")
    if stride is None:
        stride = size_px
    if stride <= 0:  # an explicit stride of 0 is an error, not "use the default"
        raise ValueError("stride must be positive.")
    height, width = image.shape[:2]
    for row in range(0, height - size_px + 1, stride):
        for col in range(0, width - size_px + 1, stride):
            yield image[row : row + size_px, col : col + size_px], row, col


def build_ode_params(
    lat: float | None,
    lon: float | None,
    half_deg: float,
    limit: int,
    product_type: str = NAC_PRODUCT_TYPE,
) -> dict[str, str]:
    """ODE query parameters, with an optional (loose) spatial box."""
    params = {
        "query": "product",
        "results": "fmp",
        "output": "json",
        "ihid": "LRO",
        "iid": "LROC",
        "pt": product_type,
        "limit": str(limit),
    }
    if lat is not None and lon is not None:
        params.update(
            minlatitude=str(lat - half_deg),
            maxlatitude=str(lat + half_deg),
            westernlon=str((lon - half_deg) % 360),
            easternlon=str((lon + half_deg) % 360),
        )
    return params


def query_nac_products(
    lat: float | None = None,
    lon: float | None = None,
    *,
    half_deg: float = 2.0,
    limit: int = 20,
    max_kbytes: float | None = None,
    timeout_s: int = 60,
) -> list[NacProduct]:
    """Query ODE for NAC products, nearest-first when a target is given."""
    import json
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    params = build_ode_params(lat, lon, half_deg, limit * 3 if lat is not None else limit)
    request = Request(ODE_URL + "?" + urlencode(params), headers={"User-Agent": "Lunaslide/1.0"})
    with urlopen(request, timeout=timeout_s) as response:
        products = parse_products(json.loads(response.read()))

    if max_kbytes is not None:
        products = [p for p in products if 0 < p.kbytes <= max_kbytes]
    if lat is not None and lon is not None:
        products.sort(key=lambda p: p.distance_deg(lat, lon))
    return products[:limit]


def download_pyr_tif(product: NacProduct, cache_dir: Path | str, timeout_s: int = 300) -> Path | None:
    """Download a product's pyramided GeoTIFF, cached by product id."""
    from urllib.request import Request, urlopen

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{product.product_id.replace('.', '_')}_pyr.tif"
    if target.is_file() and target.stat().st_size > 0:
        return target
    try:
        request = Request(product.pyr_tif_url, headers={"User-Agent": "Lunaslide/1.0"})
        with urlopen(request, timeout=timeout_s) as response:
            payload = response.read()
        if not payload:
            return None
        target.write_bytes(payload)
        return target
    except Exception:
        return None
