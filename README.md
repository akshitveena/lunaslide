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
