# GeeBeam

[![Testing + Linting](https://github.com/kysolvik/geebeam/actions/workflows/pytest-lint.yml/badge.svg)](https://github.com/kysolvik/geebeam/actions/workflows/pytest-lint.yml)

Google Earth Engine + Apache Beam for building geospatial training datasets

## Purpose:

GeeBeam is a lightweight library for building and executing Apache Beam pipelines that download data "chips" from Google Earth Engine and write them to TensorFlow records for model training.

The user defines the Earth Engine images they want to download chips from using the Python earthengine-api. geebeam then serialized the graph-definition of the images so they can be passed to the Beam workers.

The pipelines can be run locally or on Google Cloud Dataflow. (Note: currently local jobs are limited to short-running tasks due to grpc "Deadline Exceeded" error).


## Install:

```bash
pip install geebeam
```

## Examples:

### Running locally:

Here we'll create a burned area mask for 2024 using the MCD64A1 product.
For example, this could be the target variable for a burn risk model.

```python
import ee
import geebeam
import google

# Get default project id from environment (or specify PROJECT_ID manually)
PROJECT_ID = google.auth.default()[1]

# Initialize ee client, replace with your GCP project ID
ee.Initialize(project=PROJECT_ID)

### Build image for download
# Load a raw Landsat 5 ImageCollection for a single year.
ls5_collection = ee.ImageCollection('LANDSAT/LT05/C02/T1').filterDate(
    '2010-01-01', '2010-12-31'
)
# Create a (mostly) cloud-free Landsat composite
ls5_composite = ee.Algorithms.Landsat.simpleComposite(
    ls5_collection,
    asFloat=True,
    cloudScoreRange=5)

# Building and triggering the pipeline is done with a single command:
geebeam.sample_and_run_pipeline(
    image_list = [ls5_composite], # Important: has to be a list of images
    sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0), # In central-west Brazil
    n_sample=10, # Number of tiles to sample
    patch_size=128, # Number of pixels in each direction
    scale=30, # Final export resolution in meters
    crs='EPSG:4326', # CRS for final output
    project=PROJECT_ID, # GCP Project ID
    output_path='./test_data/', # Output path, local or on GCP
    validation_ratio=0.2, # Fraction to select as validation data
)
```

Now let's add another dataset: MapBiomas land-cover from same year. 
For more info, and legend, see: [MapBiomas Brasil](https://brasil.mapbiomas.org/en/codigos-de-legenda/)
```python
# MB Land-use/land-cover
mb_lulc = (
    ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_coverage_v1')
    .select('classification_2010')
)

# Exporting both together is as simple as this:
geebeam.sample_and_run_pipeline(
    image_list = [ls5_composite, mb_lulc],
    project=PROJECT_ID,
    crs='EPSG:4326',
    patch_size=128,
    scale=30,
    n_sample=10,
    validation_ratio=0.2,
    output_path='./test_data_w_mb/',
    sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0)
)
```

### Scaling up with DataFlow:

The export process can be scaled to many workers via Google Cloud DataFlow. First write a script containing your `geebeam.run_pipeline()` command. Then execute using the Beam DataFlow runner:

```bash
python examples/geebeam_run.py \
    --region=us-east1 \
    --worker=zone us-east1-b \
    --runner=DataflowRunner \
    --max_num_workers=8 \
    --experiments=use_runner_v2 \
    --temp_location=gs://[your-bucket]/[path_to_temp_dir]
    --machine_type=n2-highmem-2 \
    --sdk_container_image=kysolvik/geebeam:latest
```

Note in this case your `output_path` in run_pipeline() should be a Google Cloud Storage path. If you're running an older version of geebeam, replace "latest" in the sdk_container_image URI with the version number (e.g. v0.1.2). You can also build your own Docker image to run on. More info in the [DataFlow docs](https://docs.cloud.google.com/dataflow/docs/guides/using-custom-containers).

See the Apache Beam and Google Cloud DataFlow docs for full documentation, e.g. pipeline command-line options

#### Common DataFlow gotchas

1. Before running, you must [enable the DataFlow API on Google Cloud Console](https://console.developers.google.com/apis/api/dataflow.googleapis.com/overview).

2. If you get an error stating "Subnetwork ''... does not have Private Google Access...", you may need to activate it for your subnetwork (replace us-east1 with your region):

```bash

gcloud compute networks subnets update default \
    --region=us-east1 \
    --enable-private-ip-google-access
```

3. You can test your pipeline script (e.g. geebeam_run.py) and Beam options using the DirectRunner before submitting to DataFlow:

```bash
python examples/geebeam_run.py \
    --runner=DirectRunner
```

See [DataFlow documentation on specifying network and subnetwork](https://docs.cloud.google.com/dataflow/docs/guides/specifying-networks) for DataFlow jobs.

4. For more common errors, see the [Google Cloud DataFlow troubleshooting guide](https://docs.cloud.google.com/dataflow/docs/guides/common-errors).

## Alternatives:

- [GeeFlow](https://github.com/google-deepmind/geeflow): Google DeepMind's GeeFlow fulfills a similar purpose. It is more flexible, allowing for more user control of data processing, reprojection, and writing, but slower and no longer actively maintained. With the goal of meeting *most* users' needs, GeeBeam is designed to be easier and quicker to use, but allows from more limited data transformations.
- Export training data to Google Cloud Storage then download chips from there: This works, but if you need to get data from many different datasets it's slow to export all that data to Cloud Storage and can be expensive to store it there if you don't delete it quickly. This also uses a lot of Earth Engine compute hours, which are now subject to stricter monthly limits.
- [Xee](https://github.com/google/Xee): Xee allows for accessing Earth Engine objects as xarray.Datasets. You could use this to define a xarray.Dataset and download "chips" from it, but geebeam interfaces with Beam to automatically parallelize this task and export to Tensorflow records.
