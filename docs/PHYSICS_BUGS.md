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

*(Drop your Stage 1 screenshot here: `assets/stage1_explosion.png`)*
![Stage 1: Neon Explosion](../assets/stage1_explosion.png)

---

## Stage 2: The "Pac-Man Effect" (Periodic Boundaries)

**The Problem:**
After fixing the explosion, the center of the map became perfectly stable (zero mass moved), but the very top and very bottom edges formed massive, solid horizontal bands of deep blue (-20 meters) and bright red (+20 meters) elevation changes.

**The Cause:**
I used NumPy's `np.roll()` to shift the terrain array to calculate slopes. `np.roll()` wraps around the edges of the map—just like when you walk off the right side of the screen in Pac-Man and appear on the left side. Because the top of our synthetic hill is at elevation 50m and the bottom is at 0m, the math thought there was a giant, sheer 50-meter cliff connecting the top edge directly to the bottom edge. The top edge immediately dumped all of its dirt off the "cliff," which wrapped around and landed on the bottom edge.

**The Fix:**
Added a "Boundary Mask" to the logic. If the cellular automata attempts to shift mass off the edge of the matrix (e.g., `axis=0` and `shift=1`), we explicitly set that `mask = False`. This acts as a solid physical wall, preventing dirt from wrapping around the simulation.

*(Drop your Stage 2 screenshot here: `assets/stage2_pacman.png`)*
![Stage 2: Pac-Man Bug](../assets/stage2_pacman.png)

---

## Stage 3: The Stable Physics Engine

**The Result:**
With mass strictly conserved and boundary walls erected, the simulation performs beautifully. The final slope gently smooths out the sharpest peaks below the 30-degree critical angle, and the elevation changes show tiny, realistic amounts (between -1.5m and +1.5m) scattered naturally across the center of the terrain.

*(Drop your Stage 3 screenshot here: `assets/stage3_success.png`)*
![Stage 3: Success](../assets/stage3_success.png)
