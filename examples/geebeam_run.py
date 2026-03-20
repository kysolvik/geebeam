"""Execute GEE tile extraction in Beam + Dataflow"""

import ee
import geebeam
import google.auth

PROJECT_ID = google.auth.default()[1]

RANDOM_SEED = 54

# Define your configuration
config = {
  "project_id": PROJECT_ID,
  "patch_size": 128,
  "scale": 500,
  "crs": "EPSG:4326",
  "n_sample": 10,
  "validation_ratio": 0.2,
}

ee.Initialize(project=config['project_id'])

# MB Land-use/land-cover forest fraction
mb_amz_lulc = (
    ee.Image('projects/mapbiomas-public/assets/amazon/lulc/collection6/mapbiomas_collection60_integration_v1')
    .lt(10)
   .reduceResolution('mean', maxPixels=500)
)

# MODIS Burned Area
mcd64 = (ee.ImageCollection('MODIS/061/MCD64A1')
            .select('BurnDate')
            .filter(ee.Filter.calendarRange(2024, 2024, 'year'))
            .min()
            .gt(0)
            .rename(['Burn'])
            )
im_list = [mb_amz_lulc, mcd64]

    # Execute
if __name__ == '__main__':
    # Execute
    geebeam.run_pipeline(
        config=config,
        image_list=im_list,
        random_seed=RANDOM_SEED,
        output_path='./test/',
        sampling_region=ee.Geometry.Rectangle(-63.0, -9.0, -56.0, -4.0)
    )
