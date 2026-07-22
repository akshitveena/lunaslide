import rasterio
from rasterio.windows import Window

url = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
print("Attempting to open Cloud Optimized GeoTIFF remotely...")
try:
    with rasterio.open(url) as src:
        print(f"Opened successfully! Profile: {src.profile}")
        # Read a 500x500 window
        window = Window(col_off=10000, row_off=10000, width=500, height=500)
        data = src.read(1, window=window)
        print(f"Read data shape: {data.shape}, min: {data.min()}, max: {data.max()}")
except Exception as e:
    print(f"Failed: {e}")
