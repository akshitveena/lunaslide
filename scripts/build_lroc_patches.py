"""Build a folder of real LROC NAC image patches for Stage 1 training.

Queries ODE for calibrated NAC products, downloads each pyramided GeoTIFF, tiles
it, and keeps only tiles with real surface texture (dropping black margins and
saturated bands).  The output is unlabelled grayscale patches — exactly what the
self-supervised CurveEnhancer needs, and the base pool for later boulder/debris
labelling.

    # 400 patches of diverse real LROC imagery
    python3 -m scripts.build_lroc_patches --products 8 --max-patches 400

    # patches near a specific site
    python3 -m scripts.build_lroc_patches --lat 26.13 --lon 3.63 --products 4

Writes PNGs to data/stage1/lroc_patches/ plus a manifest recording, per patch,
the source product id and pixel window, so any patch is traceable to its origin.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.perception.lroc import (
    download_pyr_tif,
    is_informative,
    iter_tiles,
    query_nac_products,
)

OUTPUT = Path("data/stage1/lroc_patches")
CACHE = Path(".cache/lroc")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, help="target latitude (optional)")
    parser.add_argument("--lon", type=float, help="target longitude (optional)")
    parser.add_argument("--products", type=int, default=8, help="NAC products to tile")
    parser.add_argument("--size-px", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256, help="tile stride (=size for no overlap)")
    parser.add_argument("--max-patches", type=int, default=400)
    parser.add_argument("--per-product", type=int, default=80, help="cap patches kept per product")
    parser.add_argument("--min-std", type=float, default=12.0, help="min contrast to keep a tile")
    parser.add_argument("--max-kbytes", type=float, default=120000, help="skip products larger than this")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()

    try:
        import rasterio
    except ImportError:
        print("rasterio is required; run: pip install -r requirements.txt")
        return 1

    where = f"near lat {args.lat}, lon {args.lon}" if args.lat is not None else "globally diverse"
    print(f"Querying ODE for {args.products} NAC products ({where}) ...")
    products = query_nac_products(
        args.lat, args.lon, limit=args.products, max_kbytes=args.max_kbytes
    )
    if not products:
        print("No NAC products returned.")
        return 1
    print(f"Got {len(products)} products; downloading and tiling.\n")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    kept = 0

    for index, product in enumerate(products, start=1):
        if kept >= args.max_patches:
            break
        tif = download_pyr_tif(product, args.cache)
        if tif is None:
            print(f"  [{index}/{len(products)}] {product.product_id}: download failed")
            continue
        with rasterio.open(tif) as src:
            image = src.read(1)  # single-band uint8 NAC strip
        kept_here = 0
        for tile, row, col in iter_tiles(image, args.size_px, args.stride):
            if kept >= args.max_patches or kept_here >= args.per_product:
                break
            if not is_informative(tile, min_std=args.min_std):
                continue
            name = f"{product.product_id.replace('.', '_')}_r{row}_c{col}.png"
            cv2.imwrite(str(args.output / name), tile)
            manifest.append({
                "patch": name,
                "product_id": product.product_id,
                "center_lat": product.center_lat,
                "center_lon": product.center_lon,
                "row": row,
                "col": col,
                "size_px": args.size_px,
                "source_url": product.pyr_tif_url,
            })
            kept += 1
            kept_here += 1
        print(f"  [{index}/{len(products)}] {product.product_id}: "
              f"kept {kept_here} (total {kept})", flush=True)

    if not manifest:
        print("\nNo informative tiles found; nothing written.")
        return 1

    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps({
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product_type": "CDRNAC4",
        "size_px": args.size_px,
        "count": len(manifest),
        "note": "Real LROC NAC CDR imagery, not map-projected; for appearance training.",
        "patches": manifest,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {kept} patches -> {args.output}")
    print(f"  from {len({m['product_id'] for m in manifest})} products")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
