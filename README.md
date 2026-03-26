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
geebeam.run_pipeline(
    image_list = [burned_2024, mb_amz_lulc],
    project=PROJECT_ID,
    patch_size=128,
    scale=500,
    n_sample=10,
    validation_ratio=0.2,
    output_path='./test/',
    sampling_region=ee.Geometry.Rectangle(-63.0, -9.0, -56.0, -4.0),
    num_workers=1
)
```

### DataFlow:

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
    --sdk_container_image=us-docker.pkg.dev/mmacedo-reservoirid/geebeam-public/geebeam:latest
```

Note in this case your `output_path` should be a Google Cloud Storage path. If you're running an older version of geebeam, replace "latest" in the sdk_container_image URI with the version number (e.g. v0.1.2). You can also build your own Docker image to run on. More info in the [DataFlow docs](https://docs.cloud.google.com/dataflow/docs/guides/using-custom-containers).

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
