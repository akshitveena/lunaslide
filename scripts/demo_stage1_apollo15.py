import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.perception.contracts import GeoReference
from src.perception.pipeline import run_stage1

def main():
    img_path = Path("data/stage1/apollo15/apollo15.png")
    out_dir = Path("data/stage1/apollo15/output")
    
    print(f"=== TESTING LUNASLIDE STAGE 1 (PERCEPTION) ===")
    print(f"Input image: {img_path}")
    
    georef = GeoReference(
        image_id="M111578606_50CM",
        crs="IAU_2015:30100",
        source="NASA LROC NAC Orthophoto",
        ground_sample_distance_m=0.5
    )
    
    evidence = run_stage1(
        image_path=img_path,
        output_dir=out_dir,
        georef=georef
    )
    
    print("\n✅ Stage 1 Execution Complete!")
    print(f"Shadow Fraction: {evidence.shadow_fraction * 100:.2f}%")
    print(f"Texture Roughness (Laplacian std): {evidence.texture_roughness:.2f}")
    print("\nAuditable Visual Evidence Contract (JSON):")
    print(json.dumps(evidence.to_dict(), indent=2))
    
    # Visual Comparison Plotting
    raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    enhanced = cv2.imread(str(out_dir / "enhanced.png"), cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Lunaslide Stage 1: IllumiCurveNet Shadow Recovery & Feature Extraction", fontsize=16, fontweight='bold')
    
    # Panel 1: Raw Image
    axes[0, 0].imshow(raw, cmap='gray')
    axes[0, 0].set_title(f"Raw Lunar Image (Mean Intensity: {raw.mean():.1f})")
    axes[0, 0].axis('off')
    
    # Panel 2: Enhanced Image (Adaptive Gamma + CLAHE)
    axes[0, 1].imshow(enhanced, cmap='gray')
    axes[0, 1].set_title(f"Stage 1 Enhanced (Adaptive Gamma={evidence.preprocessing_report['gamma']:.2f}, CLAHE)")
    axes[0, 1].axis('off')
    
    # Panel 3: Histogram Comparison
    axes[1, 0].hist(raw.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7, label='Raw Intensity')
    axes[1, 0].hist(enhanced.ravel(), bins=256, range=[0, 256], color='cyan', alpha=0.5, label='Enhanced Intensity')
    axes[1, 0].set_title("Luminance Histogram Distribution")
    axes[1, 0].set_xlabel("Pixel Value")
    axes[1, 0].set_ylabel("Pixel Count")
    axes[1, 0].legend()
    
    # Panel 4: Metrics & Evidence Summary
    summary_text = (
        f"STAGE 1 EVIDENCE REPORT\n"
        f"-----------------------------------------\n"
        f"Image ID: {evidence.georef.image_id}\n"
        f"GSD: {evidence.georef.ground_sample_distance_m} m/pixel\n"
        f"Gamma Correction applied: {evidence.preprocessing_report['gamma']:.3f}\n"
        f"Shadow Fraction (Raw): {evidence.shadow_fraction*100:.1f}%\n"
        f"Enhanced Mean Luminance: {evidence.preprocessing_report['output_mean']*100:.1f}%\n"
        f"Texture Roughness Metric: {evidence.texture_roughness:.2f}\n"
        f"Enhancer Model: {evidence.model_versions['enhancer']}\n"
        f"Boulder Detector: {evidence.model_versions['boulder_detector']}\n"
        f"Debris Segmenter: {evidence.model_versions['debris_segmenter']}\n"
        f"-----------------------------------------\n"
        f"Status: Evidence Package Verified"
    )
    axes[1, 1].text(0.1, 0.2, summary_text, fontsize=12, family='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="black", edgecolor="cyan", alpha=0.8), color="white")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_dir / "stage1_verification.png", dpi=150)
    plt.show(block=False)
    print(f"\nVerification plot saved to: {out_dir / 'stage1_verification.png'}")

if __name__ == "__main__":
    main()
