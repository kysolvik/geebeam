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
```

Now let's add another dataset: MapBiomas Amazonia forest fraction
```python
# MB Land-use/land-cover forest fraction
# Note that LULC codes less than 10 area forest in MapBiomas Amazon Collection 6
mb_amz_lulc = (
    ee.Image('projects/mapbiomas-public/assets/amazon/lulc/collection6/mapbiomas_collection60_integration_v1')
    .lt(10)
   .reduceResolution('mean', maxPixels=500)
)

# Exporting both together is as simple as this:
geebeam.run(
    image_list = [burned_2024, mb_amz_lulc],
    project=PROJECT_ID,
    patch_size=128,
    scale=500,
    n_sample=10,
    validation_ratio=0.2,
    output_path='./test/',
    sampling_region=ee.Geometry.Rectangle(-63.0, -9.0, -56.0, -4.0),
    num_workers=2
)

```

### Dataflow: 

The export process can be scaled to many workers via Google Cloud DataFlow: 

```bash
python examples/geebeam_run.py \
    --region us-east1 \
    --worker_zone us-east1-b \
    --runner DataflowRunner \
    --max_num_workers=8 \
    --num_workers=1 \
    --experiments=use_runner_v2 \
    --machine_type=n2-highmem-2 
```

Note that in this case the output_path defined in geebeam_run.py should be a Google Cloud Storage path.

#### Artifact registry setup:

To speed up the start-up and running of jobs on DataFlow, you can build a Docker image containing `geebeam` and it's dependencies.

See instructions on Google Cloud

First, you'll need to have Docker installed on your computer, either just [Docker Engine](https://docs.docker.com/engine/) or full [Docker Desktop](https://docs.docker.com/desktop/) will do.

Next, you'll need to create  a Google Artifact Registry respository and configure your Docker to authenticate requests for the Artifact Registry. [Google Cloud DataFlow documentation has step-by-step instructions](https://docs.cloud.google.com/dataflow/docs/guides/build-container-image#before_you_begin).

Now you can pre-build the container at the start of your DataFlow job:

```bash
python examples/geebeam_run.py \
    --region us-east1 \
    --worker_zone us-east1-b \
    --runner DataflowRunner \
    --max_num_workers=8 \
    --num_workers=1 \
    --experiments=use_runner_v2 \
    --machine_type=n2-highmem-2 
    --prebuild_sdk_container_engine=local_docker \
    --docker_registry_push_url=us-east1-docker.pkg.dev/[PROJECT_ID]/[REPO_NAME]/[IMAGE_NAME] \
    --setup_file=./setup.py
```

Next time you can use the existing image with:

```bash
--sdk_container_image=us-east1-docker.pkg.dev/[PROJECT_ID]/[REPO_NAME]/[IMAGE_NAME]:[IMAGE_TAG]
```


## Alternatives:

- [GeeFlow](https://github.com/google-deepmind/geeflow): Google DeepMind's GeeFlow fulfills a similar purpose. It is more flexible, allowing for more user control of data processing, reprojection, and writing, but slower and no longer actively maintained. With the goal of meeting *most* users' needs, GeeBeam is designed to be easier and quicker to use, but allows from more limited data transformations. 
- Export training data to Google Cloud Storage then download chips from there: This works, but if you need to get data from many different datasets it's slow to export all that data to Cloud Storage and can be expensive to store it there if you don't delete it quickly. This also uses a lot of Earth Engine compute hours, which are now subject to stricter monthly limits.
- [Xee](https://github.com/google/Xee): Xee allows for accessing Earth Engine objects as xarray.Datasets. You could use this to define a xarray.Dataset and download "chips" from it, but geebeam interfaces with Beam to automatically parallelize this task.
