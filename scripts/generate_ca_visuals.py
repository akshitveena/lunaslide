"""Stage 2 visualisation suite centred on the cellular automaton itself.

Four views per site, all from real LRO topography:

1. **Evolution** — the avalanche propagating, iteration by iteration.  This is
   the automaton actually running, rather than only its fixed point.
2. **Redistribution** — how the slope distribution moves relative to the angle
   of repose, plus the convergence curve.
3. **Vibration sensitivity** — the landing-relevant view.  A descent engine
   shakes regolith, which lowers its effective friction angle; sweeping ``crit``
   downward shows how far the failure footprint spreads under that load.
4. **Relief in three dimensions** with the hazard footprint draped on it.

Snapshots come from re-running the simulator with increasing ``max_iter``.  The
automaton is deterministic (``test_repeated_runs_are_identical``), so a shorter
run is exactly a prefix of a longer one, and the total extra cost is only ~1.6x
a single full run.

    python3 -m scripts.generate_ca_visuals

Writes to ``figures/``.
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
from matplotlib.colors import LightSource, TwoSlopeNorm

from src.physics.dem_loader import fetch_patch
from src.physics.relaxation import compute_slope, simulate_mass_wasting

OUTPUT_DIR = Path("figures")
REPOSE_DEG = 30.0
CRIT = float(np.tan(np.radians(REPOSE_DEG)))
MAX_ITER = 500
CHECKPOINTS = (1, 5, 20, 60, 200, MAX_ITER)
# Effective friction angle under increasing vibrational load. Shaking mobilises
# regolith, lowering the angle at which it will hold; 30 deg is the quiescent
# dry value and the sweep spans plausible descent-engine loading.
VIBRATION_ANGLES_DEG = (30.0, 27.0, 24.0, 21.0, 18.0)


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    latitude: float
    longitude: float
    context: str
    size_px: int = 500


SITES = (
    Site("apollo15", "Apollo 15 Landing Site", 26.13, 3.63,
         "Hadley-Apennine; the Apollo 15 touchdown point and its mountain front"),
    Site("shackleton", "Shackleton Crater", -89.5, 0.0,
         "Lunar south pole; Artemis candidate region, permanently shadowed interior"),
    Site("faustini", "Faustini Crater", -87.1, 77.0,
         "South polar permanently shadowed region (PSR); a volatile cold trap"),
)


def _footer(site: Site, patch, extra: str = "") -> str:
    product = patch.product
    return (
        f"{product.citation if product else 'source raster'}  |  REAL TOPOGRAPHY\n"
        f"{site.name}  lat {site.latitude:+.2f}  lon {site.longitude:+.2f}  |  "
        f"{patch.elevation.shape[0]}x{patch.elevation.shape[1]} px  "
        f"({patch.elevation.shape[1] * patch.grid_spacing_x_m / 1000:.1f} km across)  |  "
        f"cell {patch.grid_spacing_y_m:.1f} x {patch.grid_spacing_x_m:.1f} m "
        f"(anisotropy {patch.anisotropy:.3f})  |  nodata {patch.nodata_fraction:.1%}\n"
        f"angle of repose {REPOSE_DEG:.0f} deg (crit={CRIT:.3f}), relax_factor 0.2, "
        f"4-neighbour synchronous update, mass conserved to float64"
        f"{extra}  |  generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}"
    )


def _hillshade(H, dy, dx):
    return LightSource(azdeg=315, altdeg=45).hillshade(H, vert_exag=1.0, dx=dx, dy=dy)


def run_checkpoints(H, spacing, crit=CRIT):
    """Simulator state at each checkpoint, exploiting run determinism."""
    states = []
    for ceiling in CHECKPOINTS:
        relaxed, toppled, iters = simulate_mass_wasting(
            H, grid_spacing=spacing, crit=crit, max_iter=ceiling
        )
        states.append({"ceiling": ceiling, "H": relaxed, "toppled": toppled, "iters": iters})
        if iters < ceiling:  # converged; later checkpoints are identical
            break
    return states


def figure_evolution(site: Site, patch, states) -> None:
    H0 = patch.elevation.astype(np.float64)
    dy, dx = patch.grid_spacing
    panels = states[:6]
    columns = len(panels) + 1
    fig, axes = plt.subplots(1, columns, figsize=(3.05 * columns, 4.5))
    fig.suptitle(
        f"{site.name} — cellular automaton evolution of the avalanche",
        fontsize=14, fontweight="bold", y=0.99,
    )

    axes[0].imshow(_hillshade(H0, dy, dx), cmap="gray")
    axes[0].set_title(f"Initial relief\n{H0.max() - H0.min():.0f} m", fontsize=9)

    limit = max(1e-6, float(np.abs(states[-1]["H"] - H0).max()))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    for axis, state in zip(axes[1:], panels):
        change = state["H"] - H0
        axis.imshow(_hillshade(H0, dy, dx), cmap="gray", alpha=0.55)
        image = axis.imshow(change, cmap="seismic_r", norm=norm, alpha=0.85)
        moved = float(state["toppled"].mean() * 100)
        axis.set_title(
            f"iteration {state['iters']}\n{moved:.2f}% shed, peak {np.abs(change).max():.1f} m",
            fontsize=9,
        )
        axis.set_facecolor("black")
    fig.colorbar(image, ax=axes[1:].tolist(), fraction=0.02, pad=0.01,
                 label="cumulative elevation change (m)  —  red eroded, blue deposited")

    for axis in axes:
        axis.set_xticks([]); axis.set_yticks([])
    converged = states[-1]["iters"] < MAX_ITER
    fig.text(0.005, 0.005, _footer(
        site, patch,
        f"  |  {'converged' if converged else f'CENSORED at max_iter={MAX_ITER}'}"),
        fontsize=7, family="monospace", va="bottom", color="#333333")
    fig.subplots_adjust(bottom=0.16, top=0.86, left=0.01, right=0.94)
    fig.savefig(OUTPUT_DIR / f"ca_evolution_{site.key}.png", dpi=140)
    plt.close(fig)


def figure_redistribution(site: Site, patch, states) -> None:
    H0 = patch.elevation.astype(np.float64)
    spacing = patch.grid_spacing
    before = np.degrees(np.arctan(compute_slope(H0, spacing)))
    after = np.degrees(np.arctan(compute_slope(states[-1]["H"], spacing)))

    fig = plt.figure(figsize=(15.5, 5.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.1], wspace=0.28)
    fig.suptitle(
        f"{site.name} — slope redistribution toward the angle of repose",
        fontsize=14, fontweight="bold",
    )

    ax0 = fig.add_subplot(grid[0, 0])
    bins = np.linspace(0, max(45.0, before.max()), 90)
    ax0.hist(before.ravel(), bins=bins, histtype="stepfilled", alpha=0.45,
             color="#c1440e", label="before", density=True)
    ax0.hist(after.ravel(), bins=bins, histtype="step", linewidth=1.8,
             color="#1f77b4", label="after relaxation", density=True)
    ax0.axvline(REPOSE_DEG, color="black", linestyle="--", linewidth=1.5,
                label=f"{REPOSE_DEG:.0f}° repose")
    ax0.set_xlabel("slope (degrees)"); ax0.set_ylabel("density")
    ax0.set_yscale("log")
    ax0.set_title("Slope distribution", fontsize=11)
    ax0.legend(fontsize=8); ax0.grid(alpha=0.3)

    ax1 = fig.add_subplot(grid[0, 1])
    iterations = [0] + [state["iters"] for state in states]
    unstable = [float((before > REPOSE_DEG).mean() * 100)] + [
        float((np.degrees(np.arctan(compute_slope(state["H"], spacing))) > REPOSE_DEG).mean() * 100)
        for state in states
    ]
    ax1.plot(iterations, unstable, marker="o", color="#c1440e", linewidth=1.8)
    ax1.set_xlabel("CA iteration"); ax1.set_ylabel("cells above repose (%)")
    ax1.set_title("Convergence", fontsize=11)
    ax1.grid(alpha=0.3)
    if unstable[0] > 0:
        ax1.set_ylim(0, unstable[0] * 1.15)

    ax2 = fig.add_subplot(grid[0, 2])
    image = ax2.imshow(before, cmap="magma", vmin=0, vmax=max(35.0, before.max()))
    if before.max() > REPOSE_DEG:
        ax2.contour(before, levels=[REPOSE_DEG], colors="cyan", linewidths=0.7)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"Initial slope (cyan = {REPOSE_DEG:.0f}° contour)", fontsize=11)
    fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.04, label="degrees")

    fig.text(0.005, 0.005, _footer(site, patch), fontsize=7,
             family="monospace", va="bottom", color="#333333")
    fig.subplots_adjust(bottom=0.24, top=0.87)
    fig.savefig(OUTPUT_DIR / f"ca_redistribution_{site.key}.png", dpi=140)
    plt.close(fig)


def figure_vibration(site: Site, patch) -> dict:
    """How far failure spreads as a descent engine lowers the friction angle."""
    H0 = patch.elevation.astype(np.float64)
    spacing = patch.grid_spacing
    dy, dx = spacing
    results = []
    for angle in VIBRATION_ANGLES_DEG:
        crit = float(np.tan(np.radians(angle)))
        relaxed, toppled, iters = simulate_mass_wasting(
            H0, grid_spacing=spacing, crit=crit, max_iter=MAX_ITER
        )
        results.append({
            "angle": angle, "toppled": float(toppled.mean() * 100),
            "mask": toppled, "peak": float(np.abs(relaxed - H0).max()), "iters": iters,
        })

    columns = len(results)
    fig = plt.figure(figsize=(3.1 * columns, 6.9))
    grid = fig.add_gridspec(2, columns, height_ratios=[1.25, 1.0], hspace=0.32)
    fig.suptitle(
        f"{site.name} — failure footprint vs. effective friction angle "
        f"(descent-engine vibration)",
        fontsize=14, fontweight="bold",
    )

    shade = _hillshade(H0, dy, dx)
    for index, result in enumerate(results):
        axis = fig.add_subplot(grid[0, index])
        axis.imshow(shade, cmap="gray")
        overlay = np.zeros((*H0.shape, 4))
        overlay[result["mask"]] = [1.0, 0.1, 0.0, 0.85]
        axis.imshow(overlay)
        axis.set_xticks([]); axis.set_yticks([])
        axis.set_title(
            f"{result['angle']:.0f}° repose\n{result['toppled']:.2f}% shed",
            fontsize=10,
            fontweight="bold" if index == 0 else "normal",
        )

    ax = fig.add_subplot(grid[1, :])
    angles = [r["angle"] for r in results]
    toppled = [r["toppled"] for r in results]
    ax.plot(angles, toppled, marker="o", linewidth=2.2, color="#c1440e")
    for angle, value in zip(angles, toppled):
        ax.annotate(f"{value:.2f}%", (angle, value), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("effective friction angle (degrees) — decreasing vibrational load →")
    ax.set_ylabel("cells that shed material (%)")
    ax.set_title(
        f"Sensitivity: a {angles[0] - angles[-1]:.0f}° reduction in effective friction angle "
        f"multiplies the failure footprint {toppled[-1] / max(toppled[0], 1e-9):.1f}x",
        fontsize=11,
    )
    ax.grid(alpha=0.3)

    fig.text(0.005, 0.005, _footer(
        site, patch,
        "  |  vibration is modelled as a reduced effective friction angle, "
        "not as an explicit seismic forcing term"),
        fontsize=7, family="monospace", va="bottom", color="#333333")
    fig.subplots_adjust(bottom=0.13, top=0.90)
    fig.savefig(OUTPUT_DIR / f"ca_vibration_{site.key}.png", dpi=140)
    plt.close(fig)
    return {"site": site.name, "angles": angles, "toppled": toppled}


def figure_three_d(site: Site, patch, states) -> None:
    H0 = patch.elevation.astype(np.float64)
    dy, dx = patch.grid_spacing
    toppled = states[-1]["toppled"]
    step = max(1, H0.shape[0] // 220)
    H = H0[::step, ::step]
    mask = toppled[::step, ::step]
    rows, columns = H.shape
    X, Y = np.meshgrid(np.arange(columns) * dx * step / 1000.0,
                       np.arange(rows) * dy * step / 1000.0)

    normalised = (H - H.min()) / max(float(np.ptp(H)), 1e-9)
    colours = plt.get_cmap("terrain")(normalised)
    colours[mask] = [1.0, 0.1, 0.0, 1.0]

    fig = plt.figure(figsize=(12, 8))
    axis = fig.add_subplot(111, projection="3d")
    axis.plot_surface(X, Y, H, facecolors=colours, rstride=1, cstride=1,
                      linewidth=0, antialiased=True, shade=True)
    axis.set_title(
        f"{site.name} — relief with simulated failure footprint (red)\n"
        f"{float(mask.mean() * 100):.2f}% of cells shed material",
        fontsize=13, fontweight="bold",
    )
    axis.set_xlabel("east (km)"); axis.set_ylabel("north (km)")
    axis.set_zlabel("elevation (m)")
    axis.view_init(elev=42, azim=-125)
    fig.text(0.01, 0.01, _footer(site, patch), fontsize=7,
             family="monospace", color="#333333")
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(OUTPUT_DIR / f"ca_relief3d_{site.key}.png", dpi=140)
    plt.close(fig)


def figure_summary(rows: list[dict], vibration: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    fig.suptitle(
        "Stage 2 across the three target sites — real LRO topography",
        fontsize=14, fontweight="bold",
    )

    names = [row["site"] for row in rows]
    bars = axes[0].barh(names, [row["toppled"] for row in rows], color="#1f77b4")
    axes[0].bar_label(bars, fmt="%.2f%%", fontsize=9, padding=3)
    axes[0].set_xlabel("cells that shed material (%) at the 30° quiescent repose angle")
    axes[0].set_title("Baseline hazard", fontsize=11)
    axes[0].grid(axis="x", alpha=0.3)

    for entry in vibration:
        axes[1].plot(entry["angles"], entry["toppled"], marker="o",
                     linewidth=2.0, label=entry["site"])
    axes[1].invert_xaxis()
    axes[1].set_xlabel("effective friction angle (degrees)")
    axes[1].set_ylabel("cells that shed material (%)")
    axes[1].set_yscale("symlog", linthresh=0.01)
    axes[1].set_title("Vibration sensitivity under descent load", fontsize=11)
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    detail = "  |  ".join(
        f"{row['site'].split()[0]}: {row['cell']:.1f} m cells, "
        f"{'converged' if row['converged'] else 'censored'}"
        for row in rows
    )
    fig.text(0.005, 0.01,
             f"Hazard estimates are resolution-dependent: a coarser grid averages away the "
             f"steep short-baseline slopes that actually fail.\n{detail}  |  "
             f"generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}",
             fontsize=7.5, family="monospace", color="#333333")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    fig.savefig(OUTPUT_DIR / "ca_summary_three_sites.png", dpi=140)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, vibration = [], []
    for site in SITES:
        print(f"{site.name} ...", flush=True)
        patch = fetch_patch(site.latitude, site.longitude, size_px=site.size_px, verbose=False)
        if patch is None:
            print("  REJECTED by the loader")
            continue
        print(f"  {patch.product.name if patch.product else 'raster'}  "
              f"cell {patch.grid_spacing_y_m:.1f}x{patch.grid_spacing_x_m:.1f} m", flush=True)
        states = run_checkpoints(patch.elevation.astype(np.float64), patch.grid_spacing)
        figure_evolution(site, patch, states)
        figure_redistribution(site, patch, states)
        figure_three_d(site, patch, states)
        vibration.append(figure_vibration(site, patch))
        rows.append({
            "site": site.name,
            "toppled": float(states[-1]["toppled"].mean() * 100),
            "converged": states[-1]["iters"] < MAX_ITER,
            "cell": patch.grid_spacing_y_m,
        })
        print(f"  toppled {rows[-1]['toppled']:.2f}%  "
              f"{'converged' if rows[-1]['converged'] else 'CENSORED'} "
              f"at {states[-1]['iters']} iterations", flush=True)
    if rows:
        figure_summary(rows, vibration)
    print(f"\n{len(rows)} sites -> {OUTPUT_DIR}/")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
