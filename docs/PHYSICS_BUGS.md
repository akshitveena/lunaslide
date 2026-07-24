# Physics Simulation: Bug Logs & Solutions

During the development of the Cellular Automata Physics Engine (`src/physics/relaxation.py`), we encountered two massive numerical instability bugs. This document serves to record those issues and how they were solved to achieve Conservation of Mass.

---

## Stage 1: The "Neon Explosion" (Numerical Instability)

**The Problem:**
Initially, the simulation mathematically blew up. Instead of the steep slopes gently slumping down like sand, the terrain formed intense, diagonal neon waves with elevation changes in the millions of meters (`1e6`). 

**The Cause:**
The original array logic was accidentally subtracting dirt from the source pixel, but then completely overwriting the entire terrain map with a shifted copy. This essentially "created" mass out of thin air and erased previous deductions, causing an infinite positive feedback loop that built spikes up to infinity.

**The Fix:**
Rewrote the cellular automata logic to strictly enforce the **Conservation of Mass**. When a slope is too steep, it mathematically scoops up *only* the excess height, subtracts it from the source, and uses `np.roll(delta, -shift)` to carefully deposit it onto the neighbor cell without overwriting anything else.

![Stage 1: Neon Explosion](../assets/Numerical%20Instability.png)

---

## Stage 2: The "Pac-Man Effect" (Periodic Boundaries)

**The Problem:**
After fixing the explosion, the center of the map became perfectly stable (zero mass moved), but the very top and very bottom edges formed massive, solid horizontal bands of deep blue (-20 meters) and bright red (+20 meters) elevation changes.

**The Cause:**
I used NumPy's `np.roll()` to shift the terrain array to calculate slopes. `np.roll()` wraps around the edges of the map—just like when you walk off the right side of the screen in Pac-Man and appear on the left side. Because the top of our synthetic hill is at elevation 50m and the bottom is at 0m, the math thought there was a giant, sheer 50-meter cliff connecting the top edge directly to the bottom edge. The top edge immediately dumped all of its dirt off the "cliff," which wrapped around and landed on the bottom edge.

**The Fix:**
Added a "Boundary Mask" to the logic. If the cellular automata attempts to shift mass off the edge of the matrix (e.g., `axis=0` and `shift=1`), we explicitly set that `mask = False`. This acts as a solid physical wall, preventing dirt from wrapping around the simulation.

![Stage 2: Pac-Man Bug](../assets/Pac%20Man.png)

---

## Stage 3: Directional Bias (Asynchronous Updates)

**The Problem:**
With mass conserved and the boundaries walled off, the simulation looked correct
by eye — but it was not symmetric. Relaxing a perfectly radially symmetric cone
produced a result that changed depending on which way you rotated the input.

**The Cause:**
The update swept the four neighbour directions in a fixed order —
`[(-1,1), (1,1), (-1,0), (1,0)]` — and mutated the terrain *between* directions.
By the time the last direction was evaluated, it was measuring slopes against a
grid the first three had already modified. Whichever axis happened to be listed
first got to move material before the others saw the terrain, which biased
transport along that axis.

**The Measurement:**
Relaxing a radially symmetric cone and comparing the elevation change against
its own 90-degree rotation:

| Update rule | Peak rot90 asymmetry | Mass error |
|---|---|---|
| Sequential (before) | 2.71e-2 m | 0 |
| Synchronous (after) | 0 (bit-exact) | 0 |

**The Fix:**
Rewrote the rule to be genuinely synchronous. All four directional drops are now
measured against a single frozen snapshot, every flux is derived from that one
snapshot, and the grid is updated once at the end of the iteration. Outflow is
distributed across the downhill directions in proportion to each one's excess
above repose, so no direction gets to act first.

This also exposed a latent stability limit. Because each direction previously
shed `relax_factor x excess` independently, a peak could lose up to
`4 x relax_factor` of its excess per sweep; the scheme was only stable for
`relax_factor <= 1/connectivity` and silently oscillated above it. The rewrite
adds an explicit non-inversion limiter — a cell may never end an iteration below
a neighbour it just fed — which makes it stable for any `relax_factor` in
`(0, 1]`.

Guarded by `TestDirectionalIsotropy` in `tests/test_physics_relaxation.py`.

---

## Stage 4: The Stable Physics Engine

**The Result:**
With mass strictly conserved, boundary walls erected, and the update rule made
synchronous, the simulation is well behaved. Slopes relax to exactly the
30-degree critical angle, and the result is invariant under rotation and
transposition of the input.

Transport is diffusive, so the iteration count scales with the square of the
distance material has to travel — a 25-cell cone settles in ~1.7k iterations, an
81-cell one in ~18k. The `max_iter` ceiling is therefore a real constraint on
large patches, and a run that reaches it has returned partially relaxed terrain
rather than a converged answer. `main.py` records this per sample as `converged`.

![Stage 4: Success](../assets/Cellular%20Automata%20Landslide%20Simulation.png)
