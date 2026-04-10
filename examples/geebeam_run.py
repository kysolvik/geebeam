"""Execute GEE tile extraction in Beam + Dataflow"""

import ee
import geebeam
import google

# Get default project id from environment (or specify PROJECT_ID manually)
PROJECT_ID = google.auth.default()[1]

# Initialize ee client, replace with your GCP project ID
ee.Initialize(project=PROJECT_ID)

# Load a raw Landsat 5 ImageCollection for a single year.
ls5_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1').filterDate(
    '2010-01-01', '2010-12-31'
)

# Create a (mostly) cloud-free Landsat composite
ls5_composite = ee.Algorithms.Landsat.simpleComposite(
    ls5_collection,
    asFloat=True,
    cloudScoreRange=5)

# Get some locations to sample from:
sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0) # In central-west Brazil
sample_points = geebeam.sampler.sample_region_random(
    roi=sampling_region,
    crs='EPSG:4326',
    n_sample=10
)

# Split into train and validation
sample_points_split = geebeam.sampler.split_sets(
    sample_points, split_names=['train','validation'], split_ratios=[0.8, 0.2]
)

# Building and triggering the pipeline is done with a single command:
geebeam.pipeline.run_pipeline(
    image_list = [ls5_composite], # Important: has to be a list of images
    crs='EPSG:4326', # CRS for final output
    sampling_points=sample_points_split, # Points we already generated
    output_type='tiff', # Output type: tiff with parquets for tabular data
    project=PROJECT_ID, # GCP Project ID
    patch_size=128, # Patch dimensions in each direction (# pixels)
    scale=30, # Final export resolution in meters
    output_path='./test_data/', # Output path, local or on GCP
)
