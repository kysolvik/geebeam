# GeeBeam

[![Testing + Linting](https://github.com/kysolvik/geebeam/actions/workflows/pytest-lint.yml/badge.svg)](https://github.com/kysolvik/geebeam/actions/workflows/pytest-lint.yml)

Google Earth Engine + Apache Beam for building geospatial training datasets

## Purpose:

GeeBeam is a lightweight library for building and executing Apache Beam pipelines that download data "chips" from Google Earth Engine and write them to TensorFlow records for model training. 

The pipelines can be run locally or on Google Cloud Dataflow.


## Install:

Work in progress

## Quick-Start:

Work in progress

### Artifact registry setup:
gcloud auth configure-docker us-east1-docker.pkg.dev

## Adding other datasets:

Work in progress

## Alternatives:

- [GeeFlow](https://github.com/google-deepmind/geeflow): Google DeepMind's GeeFlow fulfills a similar purpose. It is more flexible, allowing for more user control of data processing, reprojection, and writing, but slower and no longer actively maintained. With the goal of meeting *most* users' needs, GeeBeam is designed to be easier and quicker to use, but allows from more limited data transformations. 
- Export training data to Google Cloud Storage then download chips from there: This works, but if you need to get data from many different datasets it's slow to export all that data to Cloud Storage and can be expensive to store it there if you don't delete it quickly.
