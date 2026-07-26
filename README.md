# Lunaslide — predictive slope-stability screening for lunar landing sites

Lunaslide asks a question static hazard maps don't: a landing site is flat and
boulder-free *today*, but will it stay stable once a spacecraft lands on it? A
descent engine shakes regolith, and shaken regolith holds a shallower angle than
regolith at rest. A slope safe under gravity can fail under landing load.

It answers this with three stages over **real** NASA/USGS lunar data:

1. **Perception** — enhance low-light orbital imagery and detect boulders and craters.
2. **Physics** — simulate gravity-driven mass wasting on real elevation, including under vibrational load.
3. **Reasoning** — reconcile the two into a GO / CAUTION / NO-GO landing verdict.

> **Status (honest).** Stage 2 is complete and validated. Stage 1's enhancers are
> trained; its detector is a single-frame proof of concept. Stage 3 is
> implemented as a rule-based reconciler with a local-LLM agent on top. Full
> per-stage status is in [`docs/THESIS.md`](docs/THESIS.md), which also states
> every limitation plainly.

---

## Stage 1 — Perception

- **Two enhancers, compared.** A classical adaptive-gamma + CLAHE pipeline
  (γ = 0.488, mean illumination 29.4% → 44.2% on Apollo 15) and a self-supervised
  learned curve enhancer (Zero-DCE lineage), evaluated head-to-head with
  no-reference metrics on 600 real patches. Classical is sharper on well-lit
  terrain; **the learned model wins on genuinely dark permanently-shadowed-region
  imagery** (ShadowCam), recovering ~36% more shadow-region contrast — its
  intended domain.
- **Boulder/crater detection.** A two-class YOLOv8 trained on 3,866
  human-reviewed labels. Candidates are proposed by top-hat/black-hat morphology,
  refined by SAM, and split into boulder vs crater by shadow geometry relative to
  the sun (positive vs negative relief). *Honest state:* trained on a single NAC
  frame, so mAP50 = 0.21 — a proof of concept, not an operational detector.
- **`not-run`, never a false all-clear.** An untrained model reports `not-run` in
  the evidence contract; an empty result is never read as "safe."

## Stage 2 — Physics

- **Synchronous cellular-automaton mass wasting**, streamed over real LRO LOLA
  elevation (5 m polar / 59 m / 118 m products, auto-selected by latitude).
  Validated invariants: exact mass conservation, **bit-exact rotational
  isotropy**, convergence to the angle of repose (`tests/test_physics_relaxation.py`).
- **Descent-load sensitivity.** Re-running at a reduced effective friction angle
  shows failure grows sharply under vibration (e.g. 21× at Apollo 15 for a 12°
  reduction) — the signal the whole system is built around.
- **Hazard surrogate.** XGBoost emulates the expensive simulator from cheap
  terrain statistics: leave-one-band-out **R² 0.887**, held-out **macro F1 0.894**,
  beating a single-feature physical baseline. It predicts the *simulator*, not
  the Moon (see validation below).
- **Empirically validated** against the Bickel et al. (2020) catalogue of 136,610
  observed lunar rockfalls. In a case/control test (120 real rockfall sites vs
  120 random controls matched by latitude), every physics metric is significantly
  elevated at observed rockfalls — p95 slope 20.4° vs 14.5°, and simulated
  failure under descent load 1.13% vs 0.07% (a 16× higher median), all at
  **p < 10⁻⁴** (AUC 0.64–0.66). The association is real but modest, and the
  ceiling is honest: rockfalls are metre-scale features while the best available
  DEM here is 59 m/px, and impacts — not slope alone — trigger them
  (`scripts/validate_against_bickel.py`).

## Stage 3 — Reasoning

- **Rule-based reconciler** (`src/reasoning`): weighs slope failure, vibration
  sensitivity, boulder/crater density, and shadow into GO / CAUTION / NO-GO, with
  every signal, conflict, and evidence gap recorded for audit.
- **The epistemic rule.** Missing evidence — an untrained detector, terrain in
  shadow, a non-converged simulation — caps the verdict at CAUTION. A site is
  never certified GO while it cannot be fully seen. A gap never masks a real NO-GO.
- **Local-LLM agent** (`scripts.run_stage3_agent`): a model running locally via
  Ollama (qwen2.5:14b) reads the evidence, **chooses which simulations to run** to
  probe the descent-load margin, and writes a narrative verdict. The rule engine
  is enforced as a safety floor in code — the agent can be more cautious than the
  rules, never less. Fully local: no API key, no network beyond localhost.

---

## The physics debugging history

The engine reached correctness through three documented bugs, each with its
measurement, in [`docs/PHYSICS_BUGS.md`](docs/PHYSICS_BUGS.md):

- **Numerical instability** ("neon explosion") — mass created from nothing; fixed
  by strict conservation.
- **Periodic boundaries** ("Pac-Man") — `np.roll` wrapped material across the map
  edge; fixed by a per-direction boundary mask.
- **Directional bias** — sequential per-direction updates biased transport along
  one axis (2.71 × 10⁻² m asymmetry on a symmetric cone); fixed by a genuinely
  synchronous update, giving bit-exact isotropy.

A fourth defect lived in the data path, not the physics: the elevation stream
silently returned empty windows and the pipeline substituted synthetic terrain
chosen by the hazard label it was meant to predict. Every prior quantitative
result came from that synthetic fallback. It is fixed; see `docs/THESIS.md` §5.

---

## Running it

```bash
pip install -r requirements.txt

python3 -m unittest discover -s tests -t .    # full test suite (no pytest needed)

# Stage 2: build the hazard dataset, train the surrogate, render figures
python3 main.py

# Stage 2 empirical validation against observed rockfalls
python3 -m scripts.validate_against_bickel     # needs data/bickel/ (see script header)

# Stage 1: enhancement evidence, and detector training from reviewed labels
python3 -m scripts.generate_stage1_figures
python3 -m scripts.build_lroc_patches          # pull real NAC patches
python3 -m scripts.label_boulders --limit 60   # SAM-assisted draft labels
python3 -m scripts.review_boulders             # human review (required)
python3 -m scripts.train_boulder_detector

# Stage 3: landing verdicts
python3 -m scripts.run_stage3                   # rule-based
python3 -m scripts.run_stage3_agent             # local-LLM agent (needs Ollama)
```

Generated data (DEM patches, checkpoints, model weights) is gitignored and
regenerated per machine; each script has `--help`.

## Layout

| Path | Role |
|---|---|
| `src/physics/` | CA mass-wasting engine, projection-aware DEM streaming |
| `src/perception/` | Enhancers, SAM labelling, boulder/crater discriminator, contracts |
| `src/reasoning/` | Stage 3 reconciler + local-LLM agent |
| `scripts/` | Data acquisition, training, validation, figures |
| `tests/` | Physics invariants, decision logic, safety guardrails |
| `docs/THESIS.md` | Full write-up, results, and honest limitations |

## Data sources

LRO LOLA DEMs and ShadowCam (USGS/ASU/NASA), LROC NAC imagery (ODE/WUSTL),
SAM (Meta, via Ultralytics), and the Bickel et al. (2020) rockfall catalogue
(Edmond / Max Planck, doi:10.17617/3.OG927P). All real, all public.
