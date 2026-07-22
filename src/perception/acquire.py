"""Acquire Stage 1 visual inputs for locations already used by Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .sites import get_stage2_site


def download_stage1_site(site_key: str, output_dir: str | Path) -> tuple[Path, Path]:
    """Download the registered optical preview and write immutable provenance.

    Shackleton intentionally has no default visible-light image: a permanently
    shadowed target needs a specifically selected alternate-sensor product,
    rather than an arbitrary brightened optical raster.
    """
    site = get_stage2_site(site_key)
    if not site.image_url:
        raise ValueError(
            f"{site.name} has no registered visible-light Stage 1 asset. "
            "Register a suitable LROC/ShadowCam product before inference."
        )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = Path(site.image_url).suffix.lower() or ".bin"
    image_path = root / f"{site_key}{suffix}"
    request = Request(site.image_url, headers={"User-Agent": "Lunaslide Stage1/1.0"})
    with urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Downloaded empty response from {site.image_url}")
    image_path.write_bytes(payload)
    metadata = {
        "site": site.to_dict(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "stage2_coordinate_match": True,
        "note": "Stage 2 is untouched. This is an independent Stage 1 input acquisition record.",
    }
    metadata_path = root / f"{site_key}.metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return image_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a registered Stage 1 input at a Stage 2 location.")
    parser.add_argument("site", choices=["apollo15", "shackleton"])
    parser.add_argument("output_dir")
    args = parser.parse_args()
    image, metadata = download_stage1_site(args.site, args.output_dir)
    print(f"Downloaded {image}\nWrote provenance {metadata}")


if __name__ == "__main__":
    main()
