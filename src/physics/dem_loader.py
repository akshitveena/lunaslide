import cv2
import numpy as np
import os
import rasterio
from rasterio.windows import from_bounds
from rasterio.env import Env

def fetch_lunar_patch(lat, lon, size_deg=2.0):
    """
    Streams a patch of the real Lunar LRO LOLA DEM from USGS servers.
    Uses Cloud Optimized GeoTIFF (COG) technology to only download the requested bounding box.
    """
    # USGS Global Lunar DEM (118m/pixel resolution)
    url = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
    print(f"📡 Initiating COG Stream from USGS for Lat: {lat:.2f}, Lon: {lon:.2f}...")
    
    # Configure GDAL to use HTTP range requests and timeout after 15 seconds
    # This prevents the script from hanging forever if the network blocks the connection.
    with Env(GDAL_HTTP_TIMEOUT=15, VSI_CACHE=True):
        try:
            with rasterio.open(url) as src:
                left = lon - (size_deg / 2)
                right = lon + (size_deg / 2)
                bottom = lat - (size_deg / 2)
                top = lat + (size_deg / 2)
                
                window = from_bounds(left, bottom, right, top, transform=src.transform)
                
                # Fetch only the pixels inside the window over the internet
                data = src.read(1, window=window)
                if data.size == 0:
                    print(f"❌ Network Error: Stream returned empty data (Likely a coordinate projection mismatch for USGS lunar datums).")
                    return None
                    
                print(f"✅ Successfully streamed {data.shape} pixel DEM patch from USGS.")
                
                # The raw values are typically planetary radius in meters, or elevation relative to a datum.
                # We cast to float32 for the physics engine.
                return data.astype(np.float32)
                
        except Exception as e:
            print(f"❌ Network Error: Could not stream from USGS (Timeout or Blocked).")
            print(f"Details: {e}")
            return None

def save_mock_dem(H, filepath, max_elevation=100.0):
    H_clipped = np.clip(H, 0, max_elevation)
    H_normalized = (H_clipped / max_elevation) * 65535.0
    cv2.imwrite(filepath, H_normalized.astype(np.uint16))

def load_dem(filepath, max_elevation=100.0):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find DEM image at {filepath}")
        
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load image from {filepath}")
        
    if img.dtype == np.uint8:
        H_meters = (img.astype(np.float32) / 255.0) * max_elevation
    elif img.dtype == np.uint16:
        H_meters = (img.astype(np.float32) / 65535.0) * max_elevation
    else:
        H_meters = img.astype(np.float32)
        
    return H_meters
