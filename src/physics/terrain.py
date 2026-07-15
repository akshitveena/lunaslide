import numpy as np
from scipy.ndimage import gaussian_filter

def generate_base_terrain(ny=100, nx=100, slope_start=50, slope_end=0, noise_sigma=1.2, noise_scale=2.0, seed=42):
    """Generates synthetic lunar terrain with a base slope and gaussian noise."""
    np.random.seed(seed)
    base_slope = np.linspace(slope_start, slope_end, ny).reshape(-1, 1)
    noise = gaussian_filter(np.random.randn(ny, nx), sigma=noise_sigma)
    H = base_slope + noise_scale * noise
    return H
