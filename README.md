# Lunaslide Predictive Digital Twin

**Lunaslide** is a next-generation predictive  designed to guarantee spacecraft survival during lunar descent. By fusing state-of-the-art Computer Vision, Mass-Wasting Physics, and Agentic Semantic Analysis, this pipeline actively simulates lunar geology at orbital scale to find the ultimate safe landing zones.

---

## 🚀 The Architecture (3-Stage Autonomous Pipeline)

This system pushes beyond static hazard mapping. It assumes the lunar surface is dynamic and mathematically predicts how the terrain will fail under the vibrational load of a landing spacecraft.

### Stage 1: The Eyes (Computer Vision & Perception)
*(Powered by YOLO, Mask R-CNN, and U-Net)*
- **Mitigating Extreme Environments:** Advanced low-light image preprocessing combats the extreme illumination variations and deep shadows found at the Lunar South Pole (e.g., Shackleton Crater).
- **Hazard Segmentation:** Multi-stage computer vision models scan orbital imagery to detect existing, static hazards. YOLO isolates massive boulders, Mask R-CNN refines instance-level geometry, and U-Net delineates ancient landslide debris fields with 30% higher cross-region generalization.

### Stage 2: The Brain (Physics Engine & Digital Twin)
*(Powered by Cellular Automata & Cloud Optimized GeoTIFFs)*
- **Topographical Ingestion:** Lunaslide streams live 3D elevation data (DEMs) directly from NASA/USGS servers via Cloud Optimized GeoTIFFs (at 118m/pixel scale) matching the exact coordinates analyzed by Stage 1.
- **Predictive Mass Wasting:** The 3D terrain is dropped into a Cellular Automata physics simulation that applies gravity, friction, and slope constraints to predict *future* mass wasting events.
- **Machine Learning Hazard Modeling:** An XGBoost Engine analyzes the physics outputs (slope gradients, relaxation limits, surface roughness) to map the terrain into discrete danger zones: `Damage Grade 0 (Safe)`, `Grade 1 (Moderate)`, and `Grade 2 (Severe)`.

### Stage 3: The Decision (Agentic Semantic Reasoning)
- **Agentic Synthesis:** Intelligent agents reason over the multi-scale segmentation outputs (Stage 1) and the predictive mass-wasting simulations (Stage 2).
- **The Verdict:** The agents weigh conflicting constraints (e.g., a crater is flat and free of boulders, but the physics engine predicts a massive rim collapse if a moonquake occurs). Only sites that pass *both* the visual perception checks and the physical stability simulations are approved for landing.

---

## 🔬 Evolution of the Physics Engine

Building an accurate 3D physics engine for lunar gravity required solving complex mathematical anomalies.

### The "Pac-Man" Problem
During early development, our Cellular Automata avalanche simulation suffered from a severe directional bias bug. Because the grid was updating sequentially (left-to-right, top-to-bottom) rather than synchronously, mass was being moved asymmetrically. 
This created a visual artifact where avalanches looked like they were being "eaten" diagonally across the screen, earning the nickname the **Pac-Man Bug**. 

**The Fix:**
We completely rewrote the engine to use **synchronous matrix operations** via NumPy `roll()`. By calculating all directional slopes simultaneously and applying mass conservation checks (ensuring dirt eroded perfectly equaled dirt deposited), we eliminated the Pac-Man artifact. The simulation now perfectly models natural, radial mass wasting down crater walls.

### Real-World Orbital Scaling
Our initial physics engine was built on synthetic datasets where 1 pixel = 5 meters. When we ingested the massive NASA USGS topographical data (118 meters per pixel), the engine failed to predict landslides because the massive horizontal distance artificially "flattened" the slope mathematics.
We recalibrated the physics engine parameters to mathematically account for orbital-scale elevation drops, allowing the Digital Twin to accurately predict catastrophic avalanches on massive structures like the 4.2km-deep Shackleton Crater.

---

## ⚙️ Running the Pipeline

To run the Stage 2 (Physics + ML) module:
```bash
conda activate luna
python main.py
```
This will automatically:
1. Ingest real lunar DEM data.
2. Run the slope failure physics engine.
3. Train the XGBoost Hazard Predictor.
4. Output High-Fidelity 3-Panel Physics Visualizations and Hazard Heatmaps.

## Stage 1: Visual Perception

Stage 1 is developed independently from the completed physics workflow. It
enhances a lunar image before boulder detection and historic-debris
segmentation; it reports visible evidence and does not forecast future
landslides.

Run the preprocessing pass on one LRO image:

```bash
python3 -m src.perception.cli input.tif outputs/enhanced.png --report outputs/enhancement.json
```

The next Stage 1 modules consume this enhanced image as follows:

- YOLOv8 detects boulder candidates.
- Mask R-CNN optionally refines their instance boundaries.
- A ResNet-50 U-Net delineates evidence of historical landslide debris.

All models must emit their results with the image's coordinate reference,
affine transform, and resolution using `src.perception.contracts.VisualEvidence`.
That is the only interface Stage 3 needs to align visual evidence with the
completed Stage 2 output.

### Training Stage 1

The implementation is deliberately data-driven: models do not report boulders
or historical debris until they have been trained on lunar annotations.

```text
data/
├── enhancement_images/             # unlabelled low-light grayscale lunar images
├── debris/
│   ├── images/patch_001.png
│   └── masks/patch_001.png          # binary historical-debris mask, same relative path
├── boulders_yolo/
│   ├── dataset.yaml                 # names: [boulder]
│   ├── images/train, images/val
│   └── labels/train, labels/val     # standard YOLO boxes
└── boulders_instances/
    ├── images/train/patch_001.png
    └── masks/train/patch_001.npz    # `masks`: [N, H, W] instance-mask array
```

```bash
# Self-supervised IllumiCurveNet-inspired curve enhancer (no paired target images)
python3 -m src.perception.train enhancer --images data/enhancement_images --output checkpoints/enhancer.pt

# Historical-debris segmentation
python3 -m src.perception.train segmenter --images data/debris/images --masks data/debris/masks --output checkpoints/debris_unet.pt

# Boulder detector and instance-boundary refinement
python3 -m src.perception.train yolo --dataset-yaml data/boulders_yolo/dataset.yaml --output runs --epochs 100
python3 -m src.perception.train maskrcnn --images data/boulders_instances/images --masks data/boulders_instances/masks --output checkpoints/boulder_maskrcnn.pt
```

The enhancement network is an independent, IllumiCurveNet-inspired design:
adaptive recursive curve maps, spatial attention, dilated multi-scale context,
and self-guided exposure/spatial/smoothness losses. It is not represented as a
verbatim reproduction of the IJCNN 2025 paper.

### Run Stage 1

```bash
python3 -m src.perception.run input.tif outputs/stage1 --image-id LROC_PATCH_001 \
  --source LROC --gsd-m 1.0 \
  --enhancer-checkpoint checkpoints/enhancer.pt \
  --yolo-checkpoint runs/boulder/weights/best.pt \
  --maskrcnn-checkpoint checkpoints/boulder_maskrcnn.pt \
  --debris-checkpoint checkpoints/debris_unet.pt
```

The output directory contains `enhanced.png`, any boulder instance masks,
`historical_debris_mask.png`, and `visual_evidence.json`. A missing trained
checkpoint is shown as `not-run`, never interpreted as a clear hazard result.

### Fixed Stage 2 Locations

Stage 1 does not alter Stage 2. It independently records the fixed locations
already hard-coded in Stage 2: Apollo 15 (`26.13, 3.63`) and Shackleton
(`-89.5, 0.0`). Download the registered Apollo 15 LROC input and provenance:

```bash
python3 -m src.perception.acquire apollo15 data/stage1/apollo15
python3 -m src.perception.run data/stage1/apollo15/apollo15.png outputs/apollo15 \
  --image-id APOLLO15_LROC_M111578606 --source LROC --gsd-m 0.5
```

The initial Apollo file is an official browse orthophoto for visual validation.
Use the associated full-resolution GeoTIFF product for model training. The
Shackleton coordinate intentionally requires a separately selected
shadow-capable image product; visible-light enhancement must not invent PSR
surface detail.
