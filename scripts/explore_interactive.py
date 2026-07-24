"""Live, interactive Stage 2 explorer — rotate the terrain, drag the load.

Opens a real matplotlib window (not a saved PNG):

* **left**  — the terrain in 3D, rotatable with the mouse, with failing ground
  painted red.
* **right** — the same failure footprint over a hillshade, from above.
* **slider** — the effective angle of repose. Drag it down to simulate a descent
  engine shaking the regolith, and watch the failure footprint spread.

Every friction angle is simulated up front, so the slider is instant rather than
stalling for a second on each drag.

    python3 -m scripts.explore_interactive
    python3 -m scripts.explore_interactive --site tycho
    python3 -m scripts.explore_interactive --site shackleton --size-px 400

The static PNG suite (scripts.generate_ca_visuals) renders the same physics for
sharing; this is for looking around.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# No matplotlib.use("Agg") here on purpose: this script exists to open a window.
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.widgets import Slider

from src.physics.dem_loader import fetch_patch
from src.physics.relaxation import simulate_mass_wasting

SITES = {
    "apollo15": ("Apollo 15 Landing Site", 26.13, 3.63),
    "shackleton": ("Shackleton Crater", -89.5, 0.0),
    "faustini": ("Faustini Crater (PSR)", -87.1, 77.0),
    "tycho": ("Tycho Crater", -43.31, -11.36),
    "copernicus": ("Copernicus Crater", 9.62, -20.08),
    "serenitatis": ("Mare Serenitatis", 28.0, 17.5),
}
ANGLES_DEG = (34.0, 32.0, 30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="apollo15", choices=sorted(SITES))
    parser.add_argument("--size-px", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--lat", type=float, help="override latitude")
    parser.add_argument("--lon", type=float, help="override longitude")
    args = parser.parse_args()

    name, latitude, longitude = SITES[args.site]
    if args.lat is not None:
        latitude, name = args.lat, f"{latitude:+.2f}, {longitude:+.2f}"
    if args.lon is not None:
        longitude = args.lon

    print(f"Streaming {name} ...")
    patch = fetch_patch(latitude, longitude, size_px=args.size_px, verbose=True)
    if patch is None:
        print("Could not stream this location.")
        return 1

    H = patch.elevation.astype(np.float64)
    dy, dx = patch.grid_spacing
    product = patch.product

    print(f"Simulating {len(ANGLES_DEG)} friction angles (so the slider is instant) ...")
    results = {}
    for angle in ANGLES_DEG:
        crit = float(np.tan(np.radians(angle)))
        _, toppled, iterations = simulate_mass_wasting(
            H, grid_spacing=(dy, dx), crit=crit, max_iter=args.max_iter
        )
        results[angle] = toppled
        print(f"  {angle:4.0f} deg -> {toppled.mean() * 100:6.2f}% shed "
              f"({iterations} iterations)", flush=True)

    step = max(1, H.shape[0] // 160)
    coarse = H[::step, ::step]
    rows, columns = coarse.shape
    X, Y = np.meshgrid(
        np.arange(columns) * dx * step / 1000.0,
        np.arange(rows) * dy * step / 1000.0,
    )
    normalised = (coarse - coarse.min()) / max(float(np.ptp(coarse)), 1e-9)
    base_colours = plt.get_cmap("terrain")(normalised)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(H, vert_exag=1.0, dx=dx, dy=dy)

    figure = plt.figure(figsize=(15, 7.5))
    figure.suptitle(
        f"{name} — drag the slider to shake the regolith  |  "
        f"{product.name if product else 'raster'}, cell {dy:.0f} x {dx:.0f} m",
        fontsize=13, fontweight="bold",
    )
    axis3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis2d = figure.add_subplot(1, 2, 2)
    figure.subplots_adjust(bottom=0.17, top=0.90)

    state: dict = {"surface": None}

    def draw(angle: float) -> None:
        toppled = results[angle]
        percent = float(toppled.mean() * 100)

        colours = base_colours.copy()
        colours[toppled[::step, ::step]] = [1.0, 0.1, 0.0, 1.0]
        if state["surface"] is not None:
            state["surface"].remove()
        state["surface"] = axis3d.plot_surface(
            X, Y, coarse, facecolors=colours, rstride=1, cstride=1,
            linewidth=0, antialiased=True, shade=True,
        )
        axis3d.set_title(
            f"{percent:.2f}% of cells shed material   (drag to rotate)", fontsize=10)
        axis3d.set_xlabel("east (km)")
        axis3d.set_ylabel("north (km)")
        axis3d.set_zlabel("elevation (m)")

        axis2d.clear()
        axis2d.imshow(shade, cmap="gray")
        overlay = np.zeros((*H.shape, 4))
        overlay[toppled] = [1.0, 0.1, 0.0, 0.85]
        axis2d.imshow(overlay)
        axis2d.set_xticks([]); axis2d.set_yticks([])
        axis2d.set_title(
            f"Failure footprint at {angle:.0f}° effective repose\n"
            f"{'quiescent regolith' if angle >= 30 else 'under vibrational load'}",
            fontsize=10,
        )
        figure.canvas.draw_idle()

    slider_axis = figure.add_axes([0.18, 0.06, 0.64, 0.035])
    slider = Slider(
        slider_axis, "effective repose angle (deg)",
        min(ANGLES_DEG), max(ANGLES_DEG), valinit=30.0,
        valstep=sorted(ANGLES_DEG), color="#c1440e",
    )
    slider.on_changed(lambda value: draw(float(value)))
    draw(30.0)
    axis3d.view_init(elev=42, azim=-125)

    print("\nWindow open. Drag the slider; rotate the 3D view with the mouse. "
          "Close the window to exit.")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
