# GeeBeam
Google Earth Engine + Apache Beam for building geospatial training datasets

## Purpose:

GeeBeam is a lightweight library for building and executing Apache Beam pipelines that download data "chips" from Google Earth Engine and write them to TensorFlow records for model training.

The user defines the Earth Engine images they want to download chips from using the Python earthengine-api. geebeam then serialized the graph-definition of the images so they can be passed to the Beam workers. 

The pipelines can be run locally or on Google Cloud Dataflow. (Note: currently local jobs are limited to short-running tasks due to grpc "Deadline Exceeded" error)


## Install:

```
pip install geebeam
```

## Quick-Start:




### Dataflow: 

#### Artifact registry setup:

To speed up the start-up and running of jobs on DataFlow, you can build a Docker image containing `geebeam` and it's dependencies.

First, you'll need to have Docker installed on your computer, either just [Docker Engine](https://docs.docker.com/engine/) or full [Docker Desktop](https://docs.docker.com/desktop/) will do.

Next, you'll configure your Docker to [authenticate to Artificat Registry Docker repositories](https://docs.cloud.google.com/artifact-registry/docs/docker/authentication). For example, you can do this via the gcloud CLI tool (replacing us-east1 with your desired region):

```
gcloud auth configure-docker us-east1-docker.pkg.dev
```




## Alternatives:

- [GeeFlow](https://github.com/google-deepmind/geeflow): Google DeepMind's GeeFlow fulfills a similar purpose. It is more flexible, allowing for more user control of data processing, reprojection, and writing, but slower and no longer actively maintained. With the goal of meeting *most* users' needs, GeeBeam is designed to be easier and quicker to use, but allows from more limited data transformations. 
- Export training data to Google Cloud Storage then download chips from there: This works, but if you need to get data from many different datasets it's slow to export all that data to Cloud Storage and can be expensive to store it there if you don't delete it quickly. This also uses a lot of Earth Engine compute hours, which are now subject to stricter monthly limits.
- [Xee](https://github.com/google/Xee): Xee allows for accessing Earth Engine objects as xarray.Datasets. You could use this to define a xarray.Dataset and download "chips" from it, but geebeam interfaces with Beam to automatically parallelize this task.
