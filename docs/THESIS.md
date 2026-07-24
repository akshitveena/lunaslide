# Lunaslide: A Predictive Digital Twin for Lunar Landing-Site Slope Stability

**Author:** Akshit Veena
**Status:** Stage 1 partial, Stage 2 complete and validated, Stage 3 not implemented
**Repository:** `lunaslide`

---

## Abstract

Lunar landing-site selection has historically been a *static* problem: find terrain
that is flat now and free of boulders now. Lunaslide asks a different question —
will the terrain still be stable *after* a spacecraft lands on it? Descent engines
shake regolith, and shaken regolith holds a shallower angle than quiescent
regolith. A slope that is stable at rest can fail under landing load.

This work builds a three-stage system to answer that question. Stage 1 extracts
visual evidence from orbital imagery. Stage 2 — the substantive contribution here —
is a synchronous cellular-automaton mass-wasting simulator that runs over real
NASA/USGS lunar elevation data and predicts gravity-driven slope failure, including
under a reduced effective friction angle representing vibrational load. Stage 3, an
agentic reasoning layer intended to reconcile the two evidence streams into a
GO/NO-GO verdict, is specified but **not implemented**.

The physics engine is validated against three invariants: exact conservation of
mass, bit-exact rotational isotropy, and convergence to the specified angle of
repose. Three substantive defects were found and fixed during development, each
documented here with its measurement. A fourth class of defect — in the data
ingestion path rather than the physics — is reported in Section 5, because it
invalidated every quantitative result the project produced prior to this work.

**The single most important finding of this thesis is negative:** for the entire
prior history of the project, the elevation-data pipeline silently failed and
substituted synthetic terrain generated from the hazard label it was meant to
predict. No previously reported Stage 2 result was measured on real lunar data.
Section 5.1 documents the mechanism, and Section 6 reports what the corrected
pipeline actually shows.

---

## 1. Introduction

### 1.1 Motivation

The Artemis programme targets the lunar south pole, where permanently shadowed
regions (PSRs) may hold water ice. These are among the most operationally hostile
landing environments in the solar system: illumination is grazing or absent,
shadows are deep and hard-edged, and the terrain is ancient, heavily cratered, and
steep.

Existing hazard assessment is largely detection-based — segment the boulders,
measure the slopes, pick the flat spot. That approach has a blind spot. It
characterises the surface as it is, not as it will be once disturbed. Lunar
regolith is a cohesionless-to-weakly-cohesive granular medium; its stability is
governed by an angle of repose that *decreases* when the material is agitated. A
descent engine firing metres above the surface is an agitation source.

### 1.2 Contribution

1. A synchronous, mass-conserving, provably isotropic cellular-automaton model of
   gravity-driven lunar mass wasting (Section 4).
2. A correct, projection-aware ingestion path for three NASA/USGS lunar elevation
   products, automatically selecting the finest available for a given latitude
   (Section 5).
3. A vibration-sensitivity analysis that treats descent-engine loading as a
   reduction in effective friction angle and quantifies the resulting growth in
   failure footprint (Section 6.3).
4. An honest account of four defects — three in the physics, one in the data
   path — with the measurements that exposed each (Sections 4.3, 5.1).

### 1.3 What this thesis does not claim

It does not claim validated landslide *prediction*. No observed lunar mass-wasting
event is used as ground truth anywhere in this work. The simulator's outputs are
physically motivated but empirically unvalidated. Section 7 states this limitation
in full and identifies the dataset that would address it.

---

## 2. System architecture

```
                    ┌─────────────────────────────────┐
   LROC / ShadowCam │ STAGE 1 — Perception            │
   optical imagery  │ enhancement → boulders → debris │──┐
                    └─────────────────────────────────┘  │
                                                          │  VisualEvidence
                                                          │  (georeferenced)
                    ┌─────────────────────────────────┐  │
   LOLA elevation   │ STAGE 2 — Physics               │  │
   (3 products)     │ cellular automaton → hazard     │──┤
                    └─────────────────────────────────┘  │
                                                          ▼
                                        ┌──────────────────────────────┐
                                        │ STAGE 3 — Reasoning          │
                                        │ NOT IMPLEMENTED              │
                                        │ intended: GO / NO-GO verdict │
                                        └──────────────────────────────┘
```

The two upper stages are deliberately decoupled and consume different sensors.
This is not an accident of implementation; it is the architecture's central
insight.

**Optical imaging cannot see into a permanently shadowed region.** There are no
photons to collect. But LOLA is a laser altimeter — it supplies its own
illumination, and its topography inside a PSR is as good as anywhere else. So for
exactly the sites Artemis cares about most, Stage 2 has data where Stage 1 has
none. That asymmetry is why the site registry (`src/perception/sites.py`)
deliberately records **no** visible-light image for Shackleton, and why a
shadow-capable sensor (ShadowCam) is required rather than a brightened optical
raster. Enhancing an image with no signal in it manufactures detail that was never
measured.

---

## 3. Stage 1 — Perception

### 3.1 Implemented and validated

An illumination-aware preprocessing pipeline: robust percentile normalisation →
image-adaptive gamma correction → CLAHE. Gamma is solved analytically so that the
image's mean luminance maps to a target, `γ = log(target) / log(mean)`, clamped to
`[0.45, 1.45]` to avoid amplifying sensor noise in near-black frames.

Measured on the Apollo 15 LROC NAC browse orthophoto (`M111578606`, 0.5 m/px):

| Quantity | Value |
|---|---|
| Adaptive gamma | 0.488 |
| Mean luminance, input | 29.4% |
| Mean luminance, output | 44.2% |
| Shadow fraction (< 0.12) | 18.1% |
| Texture roughness (Laplacian σ) | 46.6 |

Every run emits an auditable `EnhancementReport` and a `VisualEvidence` contract
carrying the image's CRS, affine transform, and ground sample distance.

### 3.2 Specified, architected, but untrained

The following exist as correct, reviewed implementations with **no trained
weights**, and therefore produce no results:

- **YOLOv8** boulder detector. Deliberately refuses generic COCO weights — a
  COCO label set contains no `boulder` class, and using it would generate
  confident nonsense.
- **Mask R-CNN** instance refinement (ResNet-50 FPN, two classes).
- **ResNet-50 U-Net** historical-debris segmenter, trained with BCE + Dice.
- **CurveEnhancer**, a ~40K-parameter self-supervised low-light network
  (spatial attention, dilated multi-scale context, exposure/spatial/smoothness
  losses), independently designed rather than a reproduction of any published
  architecture.

### 3.3 The `not-run` convention

When a checkpoint is absent, the corresponding field in `visual_evidence.json`
reads `"not-run"` — never an empty result. This matters: an empty boulder list
from an untrained detector is indistinguishable in JSON from an empty boulder list
from a working detector that found nothing. One is "no information", the other is
"all clear". Conflating them in a landing-safety system is the kind of error that
loses vehicles.

---

## 4. Stage 2 — The physics engine

### 4.1 Formulation

Terrain is a scalar elevation field `H` on a regular grid with ground spacings
`Δy` (north-south) and `Δx` (east-west), which are **not** in general equal
(Section 5.2). Let `N` be the neighbour offset set, closed under 90° rotation:

- von Neumann (`connectivity=4`): `{(±1,0), (0,±1)}`
- Moore (`connectivity=8`): adds `{(±1,±1)}`

For each cell `i` and offset `d ∈ N` at ground distance `ℓ_d`:

```
drop_d(i)   = H(i) − H(i + d)                        # positive = downhill
excess_d(i) = max( drop_d(i) − crit · ℓ_d , 0 )      # metres above repose
```

where `crit = tan(φ)` and `φ` is the effective angle of repose. Offsets whose
neighbour lies outside the domain are masked to zero excess.

Total outflow is a fixed fraction of total excess, distributed across downhill
directions in proportion to each one's excess:

```
E(i)       = Σ_d excess_d(i)
share_d(i) = excess_d(i) / E(i)
out(i)     = relax_factor · E(i)
```

**Non-inversion limiter.** The cell retains `H(i) − out(i)` while neighbour `d`
receives `out(i)·share_d(i)`. Remaining above that neighbour requires

```
out(i)·(1 + share_d(i)) ≤ drop_d(i)     for every receiving d
```

so `out(i)` is clamped to `min_d drop_d(i)/(1 + share_d(i))`. This bound is exact
and tight. Without it the scheme is stable only for
`relax_factor ≤ 1/|N|` and oscillates silently above that.

The update is applied **once**, synchronously:

```
H'(i) = H(i) − out(i) + Σ_d  out(i−d)·share_d(i−d)
```

Iteration continues until `max_i E(i) ≤ tolerance` or `max_iter` is reached.

### 4.2 Validated invariants

All in `tests/test_physics_relaxation.py` (53 tests pass repo-wide).

| Invariant | How it is verified | Result |
|---|---|---|
| Conservation of mass | `ΣH` before vs after, float64 | exact to 1e-6 m |
| No boundary leakage | Interior-stable ramp with a 500 m edge-to-edge jump must not move | 0 iterations, array unchanged |
| Rotational isotropy | Relax a centred cone; result must commute with `np.rot90` | **bit-exact** |
| Transpose equivariance | `f(Hᵀ) = f(H)ᵀ`, with spacings swapped | ≤ 1e-9 |
| Convergence | Max directional slope after relaxation | ≤ `crit` + 1e-6 |
| Unconditional stability | `relax_factor = 1.0` on a 900 m spike | finite, mass conserved, no inversion |
| Determinism | Repeated runs | bit-identical |
| Monotonicity | Truncated runs never steepen the terrain | holds at 10/100/1000 iterations |

### 4.3 Debugging history

Three substantive defects, in the order found.

#### Defect 1 — Numerical instability ("neon explosion")

Elevation changes reached ~10⁶ m. The update subtracted material from the source
cell but then overwrote the whole grid with a shifted copy, erasing the deduction
and creating mass from nothing — a positive feedback loop.

*Fix:* strict mass bookkeeping. Scoop only the excess, subtract at the source,
deposit via `np.roll(delta, −shift)` without overwriting.

#### Defect 2 — Periodic boundaries ("Pac-Man")

The interior stabilised, but the top and bottom rows developed solid bands of
±20 m change. `np.roll` is periodic: with the crest at 500 m and the toe at 0 m,
the automaton saw a 500 m cliff joining the top edge to the bottom edge and dumped
material across the wrap.

*Fix:* a per-direction boundary mask zeroing flux that would cross the domain
edge. Guarded by `test_a_ramp_stable_in_the_interior_is_left_alone`.

#### Defect 3 — Directional bias (asynchronous updates)

Subtler, and it survived both fixes above. The update swept the four neighbour
directions in the fixed order `[(-1,1), (1,1), (-1,0), (1,0)]` — **both column
directions before either row direction** — mutating the terrain between
directions. The last direction evaluated measured slopes against a grid the first
three had already modified, so transport was biased along the first axis
processed.

Measured by relaxing a radially symmetric cone and comparing the elevation change
against its own 90° rotation:

| Update rule | Peak rot90 asymmetry | Mass error |
|---|---|---|
| Sequential | 2.71 × 10⁻² m | 0 |
| **Synchronous** | **0 (bit-exact)** | 0 |

The artifact was also directly visible in the project's own published figures: in
the pre-fix Shackleton physics panel, erosion forms a correct radial ring at the
crater rim while deposition appears as two **horizontal bands** — a radially
symmetric crater cannot relax anti-symmetrically under gravity.

*Fix:* the synchronous formulation of Section 4.1, plus the non-inversion limiter
that the rewrite exposed as necessary.

### 4.4 Computational characteristics

Transport is diffusive, so the iteration count scales with the square of the
distance material must travel:

| Cone diameter (cells) | Iterations to converge |
|---|---|
| 25 | 1,715 |
| 41 | 4,621 |
| 81 | 18,051 |

This is a material constraint. At `max_iter = 500`, a 500 × 500 patch of
hazardous terrain is *censored* — still relaxing when the run stops. Any feature
derived from the iteration count is therefore right-censored precisely on the
hazardous class, and `main.py` records a per-sample `converged` flag so this is
visible rather than silent.

---

## 5. Data ingestion

### 5.1 The pipeline had never worked

The original loader computed its window with

```python
window = from_bounds(lon-0.5, lat-0.5, lon+0.5, lat+0.5, transform=src.transform)
```

passing **degrees** to a raster whose transform is in **metres**. The LOLA global
mosaic is `PROJCS["SimpleCylindrical Moon"]` spanning ±5,458,203 m.

Measured directly against the live mosaic, for every site tested:

```
Apollo 15   window = 0.008442 x 0.008442 px  ->  read shape (0, 0), size = 0
Shackleton  window = 0.008442 x 0.008442 px  ->  read shape (0, 0), size = 0
random      window = 0.008442 x 0.008442 px  ->  read shape (0, 0), size = 0
```

An eight-thousandth of a pixel. The read returned an empty array, the loader's own
`if data.size == 0: return None` guard fired, and the caller substituted synthetic
terrain — **selected by the hazard label it was meant to predict**. Not
intermittently; on every location, on every run.

Consequences, all now corrected:

1. No prior Stage 2 result was measured on real lunar topography.
2. Two published hazard maps carried real NASA site names over procedurally
   generated terrain.
3. The reported 1.00 macro F1 was measured on data whose dominant feature —
   `noise_roughness = std(slope)` at 0.75 importance — is a direct readout of the
   `noise_scale` generator parameter (2.0 "safe" / 15.0 "moderate" / 25.0
   "extreme") chosen *by the label*. The classifier was reading the answer key.

Two further ingestion defects were found while correcting this:

**Band scaling.** The mosaic is `int16` with `scales = (0.5,)` — it stores
half-metres. `rasterio.read()` returns raw counts and never applies the scale, so
every elevation, and hence every slope, was **2× too large**. Confirmed by Apollo
15 relief falling from 8,777 m to 4,388 m — exactly half — once corrected. 4.4 km
is the correct figure: the Apennine Front rises ~4 km above Mare Imbrium at the
landing site.

**Projection anisotropy.** See Section 5.2.

### 5.2 Cells are not square

In an equirectangular projection the north-south ground spacing is constant but
the east-west spacing contracts as `cos(latitude)`:

| Latitude | N-S spacing | E-W spacing | Anisotropy |
|---|---|---|---|
| 0° | 118 m | 118 m | 1.00 |
| 26° (Apollo 15) | 118 m | 106 m | 1.11 |
| 64° | 118 m | 51 m | 2.31 |
| **89.5° (Shackleton)** | **118 m** | **~1 m** | **~115** |

Treating the grid as square inflates every east-west slope by `1/cos(latitude)`.
`compute_slope` and `simulate_mass_wasting` therefore accept a
`(north_south, east_west)` spacing pair, and per-direction distances are computed
as `√((d_row·Δy)² + (d_col·Δx)²)`.

Spacing is **measured, not assumed**: the loader inverse-projects the pixel's own
corners back to selenographic coordinates and takes great-circle distances. This
is projection-agnostic — it reports the equirectangular contraction and the polar
stereographic scale factor with no special-casing.

### 5.3 The pole is a singularity

At 89.5° S the global mosaic's cells are 118 m × 1 m. Columns 1 m apart carry no
independent information, because the product's true resolution is 118 m: east-west
detail there is interpolation, not measurement. Worse, in a simple-cylindrical
projection the entire bottom raster row *is* the south pole, smeared across 92,160
columns.

**No slope-stability result at Shackleton computed from the global mosaic is
meaningful**, including every earlier figure carrying its name.

The resolution is a different product. `ldem_87s_5mpp_cog.tif` is a LOLA south
polar DEM at **5 m/px** in `PROJCS["Moon2000_spole"]` — polar stereographic,
projected about the pole, where cells stay square. Measured at Shackleton:
**anisotropy 1.000**.

### 5.4 Product registry

Of the three products, only the 5 m south polar DEM is genuinely cloud
optimized (512 × 512 internal tiles, 8 overview levels). The two equirectangular
products are served as COGs but are **strip-organised**: each block is one
full-width scanline — 1 × 184,320 for the 59 m merge. A 128-row window therefore
costs 128 full-width strips, about 47 MB, however few columns are wanted, and
issuing hundreds of those concurrently produces truncated range reads:

```
TIFFReadEncodedStrip: got 14823 bytes, expected 368640   (= 184320 x 2 bytes)
```

That was losing ~36% of an independently-sampled dataset. Because a strip is
full-width regardless, reading a whole band costs the same 47 MB and yields
~1,400 disjoint patches. Retention went from 64% to 100% at roughly 20× less
data per patch.

The trade-off is that patches from one band share a latitude, so sampling is
stratified by band rather than independent — which is why §6.4 splits by band.
In exchange every band spans all longitudes, so both hemispheres draw from
identical latitudes.

### 5.5 The products are not all Cloud Optimized GeoTIFFs

The loader selects the finest-resolution product covering the requested latitude:

| Key | Product | Resolution | Coverage | Projection |
|---|---|---|---|---|
| `lola_spole_5m` | LOLA south polar DEM | **5 m** | 90°S–87°S | polar stereographic |
| `lolakaguya_59m` | LOLA + Kaguya TC merged | **59 m** | ±60° | equirectangular |
| `lola_global_118m` | LOLA global LDEM | 118 m | global | simple cylindrical |

All are Cloud Optimized GeoTIFFs, so only the requested window crosses the
network. Patches are cached on disk. Nodata and boundless padding are masked
together — the only way to distinguish real zero elevation from fill — and a patch
exceeding a nodata threshold is **rejected rather than repaired**.

---

## 6. Results

All figures in `figures/` carry a provenance footer stating source product,
coordinate, window size, both ground spacings, anisotropy, nodata fraction, and
convergence state. This is a direct response to Section 5.1: a figure must be
auditable in isolation.

### 6.1 Hazard is resolution-dependent

Apollo 15, identical coordinate, two products:

| Product | Cell size | Relief | Max slope | Cells shedding material |
|---|---|---|---|---|
| LOLA global | 118 m | 4,388 m | — | **0.23%** |
| LOLA+Kaguya | 59 m | 3,122 m | 59.8° | **1.07%** |

Halving the cell size **quadruples** the detected failure fraction. A coarse grid
averages away the short-baseline slopes that actually fail: at 118 m/px a cell must
drop ~68 m to exceed 30°, which real lunar terrain rarely does over that baseline.
Coarse-resolution hazard estimates are systematic *under*-estimates.

### 6.2 Real terrain versus synthetic proxies

| Site | Product | Cells shedding | Grade (legacy thresholds) |
|---|---|---|---|
| Mare Serenitatis | 118 m | 0.00% | 0 |
| Apollo 15 | 118 m | 0.23% | 1 |
| Copernicus | 118 m | 0.61% | 1 |
| **Tycho** | 118 m | **4.63%** | **2** |
| *synthetic "moderate" proxy* | — | *3.10%* | *2* |
| *synthetic "extreme" proxy* | — | *25.36%* | *2* |

Real terrain does reach the legacy grade-2 threshold — Tycho's terraced walls
clear it — but it takes a deliberately steep target. Five uniformly-random lunar
patches produced no grade-2 samples at all. **Grade 2 is rare, not absent**, which
means uniform random sampling will not find it and the class balance of any future
training set must come from targeted sampling with a geographically separated
hold-out.

The synthetic "extreme" proxy at 25.36% is 5× more active than the most hazardous
real site sampled. The proxies were not merely circular; they were wildly
exaggerated.

### 6.3 Vibration sensitivity

The landing-relevant analysis. Descent-engine loading is modelled as a reduction
in effective friction angle — vibration mobilises regolith, lowering the angle it
will hold. Sweeping `crit` from the quiescent 30° down to 18° maps how far the
failure footprint spreads under load.

Apollo 15 (LOLA+Kaguya, 59 m/px, 26.6 km across):

| Effective friction angle | Cells shedding material |
|---|---|
| 30° (quiescent) | 1.42% |
| 27° | 5.80% |
| 24° | 12.97% |
| 21° | 21.95% |
| 18° | 30.19% |

**A 12° reduction in effective friction angle multiplies the failure footprint
21.3×.** The response is strongly non-linear in the operationally relevant range:
the first 3° of degradation quadruples it.

The spatial pattern is geologically coherent and is itself a check on the model.
At 30° failure is confined to the walls of Hadley Rille and the Apennine front —
the steepest local features, and exactly where one would expect it. As the
friction angle falls, failure propagates outward from those margins onto the
massif and finally onto the plains. Nothing was told to prefer the rille; it
emerges from the topography.

Per-site curves are in `figures/ca_vibration_*.png` and
`figures/ca_summary_three_sites.png`. This is modelled as a friction-angle
reduction, **not** as an explicit seismic forcing term; it is a sensitivity
analysis, not a validated coupled vibro-acoustic model.

### 6.4 The hazard surrogate

The automaton is expensive — a hazardous 128 px patch needs ~30,000 iterations
to settle. The surrogate predicts what it *would* produce from terrain
statistics costing a single pass.

Dataset: 480 patches at 128 px (7.6 km) cut from 24 latitude bands of the 59 m
LOLA+Kaguya merge, sampled uniform-by-area within ±58°, 96.9% converged within a
2,000-iteration budget. Target `toppled_fraction` spans 0 → 0.248 with a median
of 1.5 × 10⁻⁴ — skewed roughly 2,000×.

Patches cut from one band share a latitude and are therefore not independent, so
the band is the unit of splitting:

| Evaluation | XGBoost | Linear on `unstable_fraction` |
|---|---|---|
| **Leave-one-band-out (primary)** | **R² 0.887**, negative in 0/24 | R² 0.695, negative in 3/24 |
| Nearside → farside transfer | R² 0.630, ρ 0.948 | R² 0.793, ρ 0.868 |
| Classification (farside) | **macro F1 0.894** | — |

The two evaluations differ, and the difference is the result. Given 23 bands
spanning all latitudes, the boosted model beats the single-feature baseline by
+0.192 R² and is never negative. Restricted to one hemisphere — half the data,
narrow longitude range — it drops *below* the linear baseline, which is
unmoved. Rebuilding at 352 patches instead of 480 drove that same transfer score
to **−0.379**, confirming the limit is data sufficiency rather than a model
defect.

Throughout, XGBoost ranks better than it calibrates (ρ 0.948 against R² 0.630 on
transfer). For ordering candidate landing sites — the operational task — ranking
is the relevant property; for absolute hazard magnitudes it is not.

This replaces the previously reported macro F1 of 1.00, which was measured on 53
largely synthetic patches, split randomly, against a target defined by a
threshold on a feature the model could see, with no baseline to beat.

A caution on statistical power: 24 bands means 24 independent latitude draws,
not 480 independent samples. Confidence intervals should be computed over bands.

### 6.5 The three target sites

| Site | Product | Cell | Anisotropy | Note |
|---|---|---|---|---|
| Apollo 15 | LOLA+Kaguya 59 m | 59.2 × 53.2 m | 1.114 | Apennine Front |
| Shackleton | LOLA S-polar 5 m | 5.0 × 5.0 m | **1.000** | converged in 42 iterations |
| Faustini (PSR) | LOLA S-polar 5 m | 5.0 × 5.0 m | **1.000** | 1,073 m relief, cold trap |

Faustini stands in for "the dark south" — a permanently shadowed region where
optical imaging cannot operate but laser altimetry can, illustrating the sensor
asymmetry of Section 2.

---

## 7. Limitations

Stated plainly, because several are load-bearing.

1. **No ground truth.** No observed lunar mass-wasting event is used anywhere.
   The simulator is physically motivated and internally validated, but its
   predictions are not empirically confirmed. The Bickel et al. (2020) global
   lunar rockfall catalogue is the dataset that would address this.
2. **The ML component is a surrogate, not a predictor.** XGBoost was trained to
   reproduce the simulator's own output (§6.4). It is a fast emulator of an
   expensive simulation, not a hazard predictor validated against reality, and
   its reported R² of 0.887 measures agreement with the automaton — not with the
   Moon.
3. **Iteration censoring — addressed, not eliminated.** At `max_iter = 500`
   hazardous patches did not converge, right-censoring any iteration-derived
   feature exactly on the class of interest. The budget is now 2,000 (96.9%
   converge) and features are evaluated at a fixed budget rather than as a
   stopping time, but ~3% of patches still stop while relaxing.

9. **Sampling is stratified, not independent.** Patches are cut from shared
   latitude bands because the equirectangular products are strip-organised
   (§5.5). 24 bands is 24 independent latitude draws, not 480 independent
   samples, and statistical power should be judged accordingly.
4. **Stage 1 is untrained.** No boulder or debris result exists. Only the
   classical enhancement path produces output.
5. **Stage 3 does not exist.** There is no implemented reasoning layer, no
   conflict resolution, and no GO/NO-GO output. The two upper stages have never
   exchanged data.
6. **Vibration is a sensitivity sweep**, not a physical vibro-acoustic model. No
   engine plume, no frequency content, no regolith constitutive model.
7. **The CA is a relaxation model, not a runout model.** It finds the stable
   configuration; it does not model avalanche velocity, momentum, or travel time.
8. **Resolution.** Even at 5 m/px, individual boulder-scale failures are
   sub-pixel.

---

## 8. Future work

Ordered by ratio of credibility gained to effort spent.

1. **Implement Stage 3.** A rule-based reconciler that loads `VisualEvidence` plus
   a hazard grid, aligns them via the affine transform, and emits a reasoned
   verdict. This closes the largest structural gap in the system.
2. **Supervise against the rockfall catalogue.** Convert Stage 2 from
   self-referential to genuinely predictive.
3. **Fix the censoring**, either by raising `max_iter` until convergence dominates
   or by replacing the iteration count with an uncensored quantity such as residual
   excess above repose.
4. **Recalibrate hazard thresholds** from the real distribution over thousands of
   patches, rather than the 0.1% / 2.5% cut-points inherited from synthetic craters.
5. **Train Stage 1** — the ~40K-parameter enhancer first, since it needs no
   labels; then the debris segmenter and boulder detector, bootstrapped from
   synthetic renders and fine-tuned on hand-corrected NAC crops. Replacing Mask
   R-CNN with SAM prompted by YOLO boxes eliminates the instance-mask labelling
   burden entirely.
6. **Bring in ShadowCam** (`ShadowCam_SPOLE-90_Mosaic_1m_cog.tif`, 1 m/px) as the
   Stage 1 sensor for PSRs, which is what the architecture has always required for
   the polar sites.

---

## 9. Reproducibility

```bash
pip install -r requirements.txt

python3 -m pytest tests/ -q              # 53 tests
python3 -m scripts.probe_lola_mosaic     # verify the ingestion path
python3 -m scripts.generate_figures      # site panels + reality check
python3 -m scripts.generate_ca_visuals   # CA evolution, vibration, 3D
python3 -m scripts.demo_stage1_apollo15  # Stage 1 enhancement
```

Key modules:

| Path | Role |
|---|---|
| `src/physics/relaxation.py` | Synchronous CA, `compute_slope` |
| `src/physics/dem_loader.py` | Product registry, projection-aware streaming |
| `src/perception/preprocessing.py` | Adaptive gamma + CLAHE |
| `src/perception/contracts.py` | `GeoReference`, `VisualEvidence` |
| `src/perception/geospatial.py` | Affine/CRS recovery for Stage 3 alignment |
| `tests/test_physics_relaxation.py` | The invariants of Section 4.2 |

---

## References

1. Smith, D. E., et al. *The Lunar Orbiter Laser Altimeter Investigation on the
   Lunar Reconnaissance Orbiter Mission.* Space Science Reviews, 2010.
2. Bickel, V. T., et al. *Impacts drive lunar rockfalls over billions of years.*
   Nature Communications, 2020. — the global rockfall catalogue proposed as ground
   truth in Sections 7.1 and 8.2.
3. Robinson, M. S., et al. *Lunar Reconnaissance Orbiter Camera (LROC)
   Instrument Overview.* Space Science Reviews, 2010.
4. Bak, P., Tang, C., Wiesenfeld, K. *Self-organized criticality: An explanation
   of the 1/f noise.* Physical Review Letters, 1987. — the sandpile lineage of the
   relaxation rule.
5. USGS Astrogeology Science Center. *Lunar LRO LOLA global and south polar DEM
   products.* — data sources of Section 5.4.
6. Chen, T., Guestrin, C. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016.
7. Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for
   Biomedical Image Segmentation.* MICCAI, 2015.
8. Guo, C., et al. *Zero-Reference Deep Curve Estimation for Low-Light Image
   Enhancement.* CVPR, 2020. — the family the Stage 1 curve enhancer belongs to.

---

*Supporting bug documentation: [`PHYSICS_BUGS.md`](PHYSICS_BUGS.md).*
