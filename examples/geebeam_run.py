"""Execute GEE tile extraction in Beam + Dataflow"""

import ee
import geebeam
import google

# Get default project id from environment (or specify PROJECT_ID manually)
PROJECT_ID = google.auth.default()[1]

# Initialize ee client, replace with your GCP project ID
ee.Initialize(project=PROJECT_ID)

# Build image for download
burned_2024 = (ee.ImageCollection('MODIS/061/MCD64A1')
            .select('BurnDate')
            .filter(ee.Filter.calendarRange(2024, 2024, 'year'))
            .min()
            .gt(0)
            .rename(['Burn'])
            )

# Building and triggering the pipeline is done with a single command:
geebeam.run_pipeline(
    image_list = [burned_2024],
    project=PROJECT_ID,
    patch_size=128, # Pixel dimensions in each direction
    scale=500, # Final export resolution in meters
    n_sample=10, # Number of tiles to sample
    validation_ratio=0.2, # Fraction to select as validation data
    output_path='./test/',
    sampling_region=ee.Geometry.Rectangle(-63.0, -9.0, -56.0, -4.0)
)
