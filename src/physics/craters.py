import numpy as np
from scipy.ndimage import gaussian_filter

def generate_crater_terrain(size=500, crater_depth=4200.0, crater_radius=85.0, noise_scale=10.0, seed=42):
    np.random.seed(seed)
    
    # 1. Base flat terrain
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    # Start with a base elevation
    H = np.ones((size, size)) * (crater_depth + 500.0)
    
    # 2. Add a massive central crater (e.g. Shackleton)
    center_x, center_y = size // 2, size // 2
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # Crater profile: Parabolic interior, raised rim, flat exterior
    in_crater = dist_from_center < crater_radius
    depth_profile = (dist_from_center[in_crater] / crater_radius)**2 * crater_depth - crater_depth
    H[in_crater] += depth_profile
    
    # 3. Add Fractal Multi-Octave Noise
    # Macro noise (large rolling hills)
    noise_macro = gaussian_filter(np.random.randn(size, size), sigma=20) * noise_scale * 5.0
    # Medium noise (rough terrain)
    noise_med = gaussian_filter(np.random.randn(size, size), sigma=5) * noise_scale * 2.0
    # Micro noise (jagged rocks)
    noise_micro = gaussian_filter(np.random.randn(size, size), sigma=1) * noise_scale * 0.5
    # Nano noise (raw jagged lunar regolith to trigger dense mass wasting)
    noise_nano = np.random.randn(size, size) * noise_scale * 0.5
    
    H += (noise_macro + noise_med + noise_micro + noise_nano)
    
    return H.astype(np.float32)

def generate_mare_terrain(size=500, noise_scale=2.0, seed=15):
    np.random.seed(seed)
    H = np.ones((size, size)) * 1000.0
    
    noise_macro = gaussian_filter(np.random.randn(size, size), sigma=30) * noise_scale * 3.0
    noise_micro = gaussian_filter(np.random.randn(size, size), sigma=2) * noise_scale
    noise_nano = np.random.randn(size, size) * noise_scale * 0.2
    H += (noise_macro + noise_micro + noise_nano)
    
    # Add a few tiny craters
    for _ in range(5):
        cx, cy = np.random.randint(0, size, 2)
        r = np.random.uniform(5, 15)
        d = np.random.uniform(20, 100)
        X, Y = np.meshgrid(np.arange(size), np.arange(size))
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        in_crater = dist < r
        H[in_crater] += ((dist[in_crater] / r)**2 * d - d)
        
    return H.astype(np.float32)
