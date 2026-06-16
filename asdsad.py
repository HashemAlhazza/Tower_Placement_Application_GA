import rasterio, rasterio.warp

with rasterio.open("data/egy_pop_2025_CN_1km_R2025A_UA_v1.tif") as src:
    print(rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds))
# prints: (west, south, east, north)
