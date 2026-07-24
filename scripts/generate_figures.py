"""Regenerate the Stage 2 figure set from real LRO LOLA topography.

Every figure produced here carries a provenance footer naming the data source,
the coordinate, the ground sample distance on both axes, and whether the terrain
is real or synthetic.  The previous figure set could not be audited after the
fact: three maps were captioned with real NASA site names while showing
procedurally generated terrain, and nothing in the image said so.

    python3 -m scripts.generate_figures

Writes to ``figures/``.  ``assets/`` is left alone; it holds the historical
debugging screenshots, which remain valid records of real bugs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource

from src.physics.craters import generate_crater_terrain
from src.physics.dem_loader import DEM_PRODUCTS, fetch_patch
from src.physics.relaxation import compute_slope, simulate_mass_wasting

OUTPUT_DIR = Path("figures")
CRIT = 0.577  # tan(30 deg), dry angle of repose for lunar regolith
MAX_ITER = 500
SIZE_PX = 300


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    latitude: float
    longitude: float
    note: str = ""
    size_px: int = SIZE_PX


SITES = (
    Site("apollo15", "Apollo 15 Landing Site", 26.13, 3.63, "Hadley-Apennine; Stage 2 reference site"),
    Site("tycho", "Tycho Crater", -43.31, -11.36, "85 km young impact crater, steep terraced walls"),
    Site("copernicus", "Copernicus Crater", 9.62, -20.08, "93 km crater, terraced rim"),
    Site("serenitatis", "Mare Serenitatis", 28.00, 17.50, "Flat mare basin; low-relief control"),
    # A 300 px window at -89.5 deg overhangs the mosaic's south edge, which sits
    # only ~128 px away. 200 px fits -- but see the diagnostic figure for why
    # fitting is not the same as being analysable.
    Site("shackleton", "Shackleton Crater", -89.5, 0.0, "Lunar south pole", size_px=200),
)


def _provenance(site: Site, patch, stats: dict) -> str:
    # Must come from the patch, not a constant: the product is selected per
    # site by latitude, so a hardcoded citation silently misattributes the
    # data -- the exact failure this footer exists to prevent.
    product = patch.product
    citation = product.citation if product else "source raster (unregistered)"
    return (
        f"Source: {citation}  |  REAL TOPOGRAPHY\n"
        f"{site.name}  lat {site.latitude:+.2f}  lon {site.longitude:+.2f}  |  "
        f"window {patch.elevation.shape[0]}x{patch.elevation.shape[1]} px  |  "
        f"ground sample {patch.grid_spacing_y_m:.0f} m (N-S) x {patch.grid_spacing_x_m:.0f} m (E-W), "
        f"anisotropy {patch.anisotropy:.2f}x\n"
        f"relief {stats['relief']:.0f} m  |  mean slope {stats['mean_deg']:.1f}deg  |  "
        f"p99 slope {stats['p99_deg']:.1f}deg  |  "
        f"cells exceeding {np.degrees(np.arctan(CRIT)):.0f}deg repose: {stats['above_repose']:.2f}%\n"
        f"CA: {stats['iters']} iterations, "
        f"{'converged' if stats['converged'] else f'CENSORED at max_iter={MAX_ITER} (still relaxing)'}  |  "
        f"cells that shed material: {stats['toppled']:.2f}%  |  "
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
    )


def render_site(site: Site) -> dict | None:
    patch = fetch_patch(site.latitude, site.longitude, size_px=site.size_px, verbose=False)
    if patch is None:
        print(f"  {site.name}: REJECTED by the loader (nodata or outside the mosaic)")
        return None

    H = patch.elevation.astype(np.float64)
    dy, dx = patch.grid_spacing
    slope = compute_slope(H, (dy, dx))
    slope_deg = np.degrees(np.arctan(slope))
    relaxed, toppled, iters = simulate_mass_wasting(
        H, grid_spacing=(dy, dx), crit=CRIT, max_iter=MAX_ITER
    )
    change = relaxed - H

    stats = {
        "relief": float(H.max() - H.min()),
        "slope_deg": slope_deg,
        "mean_deg": float(slope_deg.mean()),
        "p99_deg": float(np.percentile(slope_deg, 99)),
        "above_repose": float((slope > CRIT).mean() * 100),
        "toppled": float(toppled.mean() * 100),
        "iters": iters,
        "converged": iters < MAX_ITER,
        "site": site.name,
        "latitude": site.latitude,
        "longitude": site.longitude,
        "anisotropy": patch.anisotropy,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))
    fig.suptitle(f"{site.name} — real LRO LOLA topography", fontsize=15, fontweight="bold", y=0.98)

    light = LightSource(azdeg=315, altdeg=45)
    shaded = light.hillshade(H, vert_exag=1.0, dx=dx, dy=dy)
    axes[0].imshow(shaded, cmap="gray")
    axes[0].set_title(f"Hillshaded topography\n{stats['relief']:.0f} m relief", fontsize=10)

    # Slope with the repose angle drawn on, so "how close to failing" is legible
    # rather than implied.
    slope_image = axes[1].imshow(slope_deg, cmap="magma", vmin=0, vmax=max(35.0, slope_deg.max()))
    repose_deg = float(np.degrees(np.arctan(CRIT)))
    if slope_deg.max() > repose_deg:
        axes[1].contour(slope_deg, levels=[repose_deg], colors="cyan", linewidths=0.8)
        contour_note = f"cyan = {repose_deg:.0f}deg repose contour"
    else:
        contour_note = f"nowhere reaches the {repose_deg:.0f}deg repose angle"
    axes[1].set_title(f"Slope magnitude\n{contour_note}", fontsize=10)
    fig.colorbar(slope_image, ax=axes[1], fraction=0.046, pad=0.04, label="degrees")

    limit = float(np.abs(change).max())
    if limit == 0:
        limit = 1.0
    change_image = axes[2].imshow(change, cmap="seismic_r", vmin=-limit, vmax=limit)
    axes[2].set_title(
        f"Simulated mass movement (red = eroded, blue = deposited)\n"
        f"{stats['toppled']:.2f}% of cells shed material (peak {limit:.1f} m)",
        fontsize=10,
    )
    fig.colorbar(change_image, ax=axes[2], fraction=0.046, pad=0.04, label="metres changed")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.text(0.01, 0.005, _provenance(site, patch, stats), fontsize=7.5,
             family="monospace", va="bottom", color="#333333")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    target = OUTPUT_DIR / f"site_{site.key}.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"  {site.name}: wrote {target}  (toppled {stats['toppled']:.2f}%, "
          f"{'converged' if stats['converged'] else 'CENSORED'})")
    return stats


def render_polar_diagnostic(site: Site) -> None:
    """Why the global mosaic cannot answer questions about the poles."""
    patch = fetch_patch(site.latitude, site.longitude, size_px=site.size_px, verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle(
        "Shackleton Crater — why this mosaic cannot be used at the pole",
        fontsize=14, fontweight="bold",
    )

    latitudes = np.linspace(0, 89.9, 500)
    east_west = 118.45 * np.cos(np.radians(latitudes))
    axes[0].plot(latitudes, east_west, color="#c1440e", linewidth=2, label="east-west spacing")
    axes[0].axhline(118.45, color="#1f77b4", linestyle="--", linewidth=1.5, label="north-south spacing")
    axes[0].axvline(89.5, color="black", linestyle=":", linewidth=1.2)
    axes[0].annotate(
        f"Shackleton (89.5°S)\neast-west spacing {118.45*np.cos(np.radians(89.5)):.1f} m",
        xy=(89.5, 118.45 * np.cos(np.radians(89.5))), xytext=(52, 46),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1), fontsize=9,
    )
    axes[0].set_xlabel("latitude (degrees)")
    axes[0].set_ylabel("ground distance between adjacent pixels (m)")
    axes[0].set_title("Equirectangular cells stop being square", fontsize=11)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].axis("off")
    if patch is None:
        body = "The loader rejected this window:\ntoo much of it falls outside the mosaic."
    else:
        body = (
            f"A {patch.elevation.shape[0]}x{patch.elevation.shape[1]} px window does fetch here, but\n"
            f"its cells are {patch.grid_spacing_y_m:.0f} m x {patch.grid_spacing_x_m:.1f} m "
            f"— {patch.anisotropy:.0f}:1.\n\n"
            f"Columns {patch.grid_spacing_x_m:.1f} m apart carry no independent\n"
            f"information: the mosaic's true resolution is 118 m, so\n"
            f"east-west detail here is interpolation, not measurement."
        )
    axes[1].text(
        0.02, 0.97,
        "Finding\n"
        "-------\n"
        f"{body}\n\n"
        "In a simple-cylindrical projection the pole is a singularity:\n"
        "the mosaic's entire bottom row is the single south pole point,\n"
        "stretched across 92,160 columns.\n\n"
        "Consequence\n"
        "-----------\n"
        "No slope-stability result at Shackleton computed from this\n"
        "product is meaningful, including every earlier figure that\n"
        "carried its name.\n\n"
        "Required instead\n"
        "----------------\n"
        "An LOLA polar stereographic DEM, which is projected about the\n"
        "pole and keeps cells square where it matters.",
        fontsize=9.5, family="monospace", va="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff8e7", edgecolor="#c1440e"),
    )
    fig.text(
        0.01, 0.01,
        f"Source: LRO LOLA LDEM 118 m/px global mosaic (USGS), simple cylindrical  |  "
        f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        fontsize=7.5, family="monospace", color="#333333",
    )
    fig.subplots_adjust(bottom=0.13, top=0.88)
    target = OUTPUT_DIR / "diagnostic_shackleton_projection.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"  Shackleton diagnostic: wrote {target}")


def render_reality_check(real_stats: list[dict]) -> None:
    """Real terrain against the synthetic proxies that stood in for it."""
    synthetic = {
        "safe proxy": generate_crater_terrain(size=300, crater_depth=1.0, crater_radius=1.0, noise_scale=2.0, seed=15),
        "moderate proxy": generate_crater_terrain(size=300, crater_depth=3500.0, crater_radius=100.0, noise_scale=15.0, seed=42),
        "extreme proxy": generate_crater_terrain(size=300, crater_depth=6000.0, crater_radius=150.0, noise_scale=25.0, seed=89),
    }
    synthetic_rows = []
    for label, H in synthetic.items():
        slope_deg = np.degrees(np.arctan(compute_slope(H, 118.0)))
        _, toppled, _ = simulate_mass_wasting(H, grid_spacing=118.0, crit=CRIT, max_iter=MAX_ITER)
        synthetic_rows.append((label, slope_deg, float(toppled.mean() * 100)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    fig.suptitle(
        "Real lunar terrain vs. the synthetic proxies that replaced it",
        fontsize=14, fontweight="bold",
    )

    ordered_real = sorted(real_stats, key=lambda row: row["toppled"])
    names = [row["site"] for row in ordered_real] + [row[0] for row in synthetic_rows]
    values = [row["toppled"] for row in ordered_real] + [row[2] for row in synthetic_rows]
    colours = ["#1f77b4"] * len(ordered_real) + ["#c1440e"] * len(synthetic_rows)
    bars = axes[0].barh(names, values, color=colours)
    axes[0].bar_label(bars, fmt="%.2f%%", fontsize=8, padding=3)
    axes[0].axvline(2.5, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_xlabel("cells that shed material (%)")
    axes[0].set_title(
        "Blue = real LOLA topography, red = synthetic proxy\n"
        "dashed line = the 2.5% 'grade 2' threshold",
        fontsize=11,
    )
    axes[0].set_xscale("symlog", linthresh=0.01)
    axes[0].grid(axis="x", alpha=0.3)

    for row in ordered_real:
        axes[1].hist(row["slope_deg"].ravel(), bins=80, histtype="step", linewidth=1.6,
                     density=True, label=f"{row['site']} (real)")
    for label, slope_deg, _ in synthetic_rows[1:]:
        axes[1].hist(slope_deg.ravel(), bins=80, histtype="step", linewidth=1.3,
                     linestyle="--", density=True, label=f"{label} (synthetic)")
    axes[1].axvline(np.degrees(np.arctan(CRIT)), color="black", linestyle="--",
                    linewidth=1.5, label="30° angle of repose")
    axes[1].set_xlabel("slope (degrees)")
    axes[1].set_ylabel("density")
    axes[1].set_xlim(0, 45)
    axes[1].set_title("Slope distributions — solid = real, dashed = synthetic", fontsize=11)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    steep = [row for row in ordered_real if row["toppled"] >= 2.5]
    reachable = (
        f"Real terrain does reach grade 2: {', '.join(r['site'] for r in steep)} "
        f"({max(r['toppled'] for r in steep):.2f}%). But it takes a deliberately steep target -- "
        f"{len(ordered_real) - len(steep)} of {len(ordered_real)} real sites here stay below it,\n"
        f"and five uniformly-random lunar patches produced none at all. Grade 2 is rare, not absent, "
        f"so uniform sampling will not find it."
        if steep else
        f"None of the {len(ordered_real)} real sites sampled reached the 2.5% grade 2 threshold."
    )
    fig.text(
        0.01, 0.01,
        "Synthetic proxies were selected by the hazard label they were meant to predict, so any score "
        "measured over them is circular.\n"
        f"{reachable}\n"
        f"The 'extreme' proxy at {synthetic_rows[2][2]:.2f}% is "
        f"{synthetic_rows[2][2] / max(r['toppled'] for r in ordered_real):.0f}x more active than the "
        f"most hazardous real site sampled.  |  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
        fontsize=7.5, family="monospace", color="#333333",
    )
    fig.subplots_adjust(bottom=0.20, top=0.88)
    target = OUTPUT_DIR / "reality_check_real_vs_synthetic.png"
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"  Reality check: wrote {target}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The product is chosen per site by latitude, so naming one here would be
    # wrong for most of them; each figure states its own source in its footer.
    print("Regenerating Stage 2 figures. Product auto-selected per site:")
    for product in DEM_PRODUCTS:
        print(f"  {product.resolution_m:6.1f} m/px  {product.min_latitude:+.0f}.."
              f"{product.max_latitude:+.0f} deg  {product.name}")
    print()
    collected = []
    for site in SITES:
        if site.key == "shackleton":
            continue
        stats = render_site(site)
        if stats is not None:
            collected.append(stats)
    print()
    render_polar_diagnostic(next(s for s in SITES if s.key == "shackleton"))
    if collected:
        render_reality_check(collected)
    print(f"\n{len(collected)} real sites rendered into {OUTPUT_DIR}/")
    return 0 if collected else 1


if __name__ == "__main__":
    sys.exit(main())
