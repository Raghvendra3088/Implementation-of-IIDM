import os
from sentinelhub import (
    SHConfig, BBox, CRS, DataCollection,
    SentinelHubRequest, MimeType
)

config = SHConfig("iidm")

OUTPUT_DIR = os.path.expanduser("~/iidm_project/sentinel2/raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Huize ko 4 tiles mein divide karo
TILES = {
    "tile_NW": [103.055, 26.430, 103.483, 27.052],
    "tile_NE": [103.483, 26.430, 103.911, 27.052],
    "tile_SW": [103.055, 25.809, 103.483, 26.430],
    "tile_SE": [103.483, 25.809, 103.911, 26.430],
}

for band in ["B02", "B03", "B04", "B08"]:
    print(f"\n{'='*40}\nDownloading {band}...")
    for tile_name, bbox_coords in TILES.items():
        print(f"  {tile_name}...", end=" ")
        save_dir = os.path.join(OUTPUT_DIR, band, tile_name)
        os.makedirs(save_dir, exist_ok=True)
        request = SentinelHubRequest(
            data_folder=save_dir,
            evalscript=f"""
                //VERSION=3
                function setup() {{
                    return {{ input: ["{band}"], output: {{ bands: 1, sampleType: "FLOAT32" }} }};
                }}
                function evaluatePixel(sample) {{ return [sample.{band}]; }}
            """,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=("2020-09-01", "2020-09-30"),
                mosaicking_order="leastCC",
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=BBox(bbox_coords, crs=CRS.WGS84),
            size=(2048, 2048),
            config=config,
        )
        request.get_data(save_data=True)
        print(f"✅")

print(f"\n🎉 Done! → {OUTPUT_DIR}/")
