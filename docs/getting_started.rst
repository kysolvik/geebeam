Getting Started
===============

In this tutorial we will download a set of Landsat 8 image patches from a
region of central-west Brazil and save them as GeoTIFFs on your local machine.
By the end you will have real files you can open in QGIS, rasterio, or any GIS
tool — and a working mental model of how ``geabeam`` pipelines are structured.

.. note::

   This tutorial runs locally. We'll keep the sample count small (10 patches)
   so everything finishes in under a minute or so. For info on running on
   DataFlow, see :ref:`scaling-up`.

What is ``geebeam``?
--------------------

``geebeam`` builds and executes Apache Beam pipelines to download patches of 
Earth Engine data. It automatically runs in parallel, can be run locally or
on Google Cloud Dataflow, and supports writing to various popular geospatial 
machine learning/AI dataset formats, including GeoTIFFs, Tensorflow Datasets,
and WebDataset.

Prerequisites
-------------

Before starting you will need:

- **Python 3.10 or later**
- **A Google account with Earth Engine access.** Sign up at
  `earthengine.google.com <https://earthengine.google.com>`_ if you haven't
  already. Approval is usually instant for existing Google accounts.
- **A Google Cloud project.** Only needed if running on DataFlow and/or
  if you do not meet `Earth Engine's Noncommercial/Research Use criteria 
  <https://earthengine.google.com/noncommercial/>`_. The Google Cloud 
  free tier is sufficient for this tutorial. Create one at 
  `console.cloud.google.com <https://console.cloud.google.com>`_.


Installation
------------

.. code-block:: bash

   pip install geebeam

To verify the install worked:

.. code-block:: python

   import geebeam
   print(geebeam.__version__)


Authentication
--------------

geebeam uses the Earth Engine Python API, which needs a one-time authentication
step:

.. code-block:: bash

   earthengine authenticate

This opens a browser window and saves credentials locally. You only need to do
this once per machine.

You also need to tell Earth Engine which Google Cloud project to bill API calls
against. The easiest way is to let the SDK detect it from your environment:

.. code-block:: python

   import google.auth
   PROJECT_ID = google.auth.default()[1]   # reads from gcloud config
   print(PROJECT_ID)                       # confirm it's the right project

Or just set it directly::

   PROJECT_ID = "my-gcp-project"


Step 1: Define the image you want
----------------------------------

``geabeam`` downloads image "chips" — fixed-size pixel patches — from any
``ee.Image`` you can build with the Earth Engine Python API. You tell ``geabeam``
*what* to download; it handles the parallelism and file writing.

Here we'll build a cloud-free Landsat 8 composite for 2023:

.. code-block:: python

   import ee
   import geebeam

   ee.Initialize(project=PROJECT_ID)

   ls8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
       .filterDate("2023-01-01", "2023-12-31")

   ls8_composite = ls8_collection.median().select(
       ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
   )

A few things to notice:

- ``image_list`` (used in the next step) must be a **Python list** of
  ``ee.Image`` objects, even when you only have one image. This is how ``geabeam``
  knows how to split bands across workers when needed.
- The image is not downloaded yet — this is just an EE graph definition. Nothing
  leaves Google's servers until the pipeline runs. When that happens, ``geebeam``
  will automatically serialize the  graph definition and send it to the workers
  to start downloading patches.


Step 2: Run the pipeline
-------------------------

:func:`~geebeam.sample_and_run_pipeline` is the quickest way to get started. It
randomly samples ``n_sample`` points within a region, then downloads a patch
centered on each point.

.. code-block:: python

   geebeam.sample_and_run_pipeline(
       image_list=[ls8_composite],
       sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0),
       n_sample=10,
       patch_size=128,   # 128 × 128 pixels per chip
       scale=30,         # 30 m/pixel (native Landsat resolution)
       crs="EPSG:4326",
       project=PROJECT_ID,
       output_path="./tutorial_output/",
       validation_ratio=0.2,   # 2 of the 10 patches go to the validation split
   )

You will see Apache Beam log lines scroll by. When it finishes the
``tutorial_output/`` directory will exist.

**What does ``patch_size`` and ``scale`` mean together?**
Each chip covers ``128 × 30 m = 3.84 km`` on a side. Increasing ``patch_size``
gives more spatial context. Increasing ``scale`` also gives more spatial_context
but at the cost of lower resolution/detail.

**What does ``position`` do?**
By default each sampling point is the *center* of its patch
(``position="center"``). You can change this to ``"top-left"``,
``"top-right"``, ``"bottom-left"``, or ``"bottom-right"`` if your workflow
requires the coordinate to refer to a specific corner.


Step 3: Inspect the output
---------------------------

.. code-block:: text

   tutorial_output/
   ├── train/
   │   ├── 00000.tif
   │   ├── 00001.tif
   │   └── ...          (8 patches)
   ├── validation/
   │   ├── 00008.tif
   │   └── 00009.tif    (2 patches)
   └── metadata-00000-of-00001.parquet

Each ``.tif`` is a multi-band GeoTIFF containing all the bands from your image
list. The ``metadata`` Parquet file records the sampling location (``x``, ``y``),
the patch origin (``x_topleft``, ``y_topleft``), the split assignment, and the
file path for each chip.

Open a chip with rasterio to confirm it looks right:

.. code-block:: python

   import rasterio
   import matplotlib.pyplot as plt
   import numpy as np

   with rasterio.open("tutorial_output/train/00000.tif") as ds:
       print("CRS:", ds.crs)
       print("Bands:", ds.count)
       print("Shape:", ds.height, "×", ds.width)

       # Read RGB bands (B4=red, B3=green, B2=blue in Landsat 8 L2)
       r, g, b = ds.read(2), ds.read(1), ds.read(0)

   rgb = np.stack([r, g, b], axis=-1).astype(float)
   rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)   # stretch to [0, 1]
   plt.imshow(rgb)
   plt.title("Patch 00000 — RGB")
   plt.axis("off")
   plt.show()

You can also read the metadata table:

.. code-block:: python

   import pandas as pd

   df = pd.read_parquet("tutorial_output/metadata-00000-of-00001.parquet")
   print(df[["id", "x", "y", "x_topleft", "y_topleft", "split", "image_path"]])


Adding a second image
----------------------

One of the main reasons to use ``geabeam`` is that downloading many datasets in a
single pipeline is just as easy as downloading one. The bands from every image in
``image_list`` are stacked into the same chip/patch files:

.. code-block:: python

   # MapBiomas land-use / land-cover, same year
   mb_lulc = (
       ee.Image(
           "projects/mapbiomas-public/assets/brazil/lulc/"
           "collection10_1/mapbiomas_brazil_collection10_1_coverage_v1"
       )
       .select("classification_2023")
   )

   geebeam.sample_and_run_pipeline(
       image_list=[ls8_composite, mb_lulc],    # both images here
       sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0),
       n_sample=10,
       patch_size=128,
       scale=30,
       crs="EPSG:4326",
       project=PROJECT_ID,
       output_path="./tutorial_output_with_lulc/",
       validation_ratio=0.2,
   )

The output TIFFs will now contain 7 bands (6 Landsat + 1 LULC). Band names in
the file metadata match the EE band names, so you can always tell which is which.


Splitting processing to avoid the 50 MB limit
----------------------------------------------

Each call to Earth Engine's ``computePixels`` API is capped at **50 MB** per
response. By default ``geabeam`` combines all bands from all images in
``image_list`` into a single request per patch, so you can hit this limit with
large patches or many bands.

A rough estimate of response size:

.. code-block:: text

   response_bytes ≈ patch_size × patch_size × n_bands × bytes_per_pixel
   e.g. 512 × 512 × 48 bands × 4 bytes (float32) ≈ 50 MB

When you expect to be near or over the limit, pass ``split_processing=True``.
This makes ``geabeam`` issue **one** ``computePixels`` request per image in
``image_list`` instead of one combined request, so each individual call stays
well under the cap:

.. code-block:: python

   geebeam.sample_and_run_pipeline(
       image_list=[ls8_composite, mb_lulc],
       split_processing=True,    # one EE request per image
       sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0),
       n_sample=10,
       patch_size=512,
       scale=30,
       crs="EPSG:4326",
       project=PROJECT_ID,
       output_path="./tutorial_output_split/",
   )

The output is identical — bands from all images are still merged into the same
chip files. The only difference is the number of round-trips to Earth Engine
per patch: one per image rather than one total. For small patches or few bands
this adds unnecessary overhead, so leave it at the default (``False``) unless
you actually need it.


.. _scaling-up:

Scaling up with Dataflow
-------------------------

The local runner is useful for development and small jobs. For production
workloads — thousands of patches or large images — run on Google Cloud
Dataflow. Write your pipeline call to a script and invoke it with Dataflow
runner options:

.. code-block:: bash

   python my_pipeline.py \
       --runner=DataflowRunner \
       --region=us-east1 \
       --worker_zone=us-east1-b \
       --max_num_workers=8 \
       --temp_location=gs://my-bucket/tmp/ \
       --sdk_container_image=kysolvik/geebeam:|version| \
       --machine_type=n2-highmem-2 \
       --experiments=use_runner_v2

Set ``output_path`` to a ``gs://`` path and ``geabeam`` will write directly to
Google Cloud Storage. See the
`Dataflow documentation <https://cloud.google.com/dataflow/docs>`_ for full
option reference.

.. tip::

   Test your script locally first with ``--runner=DirectRunner`` before
   submitting to Dataflow. This catches serialisation errors and EE API issues
   without spending cloud credits.


What's next
------------

- **Grid sampling** — use :func:`~geebeam.grid_and_run_pipeline` to sample on a
  regular grid rather than randomly. Useful when you need complete spatial
  coverage without gaps. You can also overlap by controlling the ``stride`` arg.
- **Custom sampling points** — use :func:`~geebeam.run_pipeline` directly
  and pass your own ``sampling_points`` as a Geopandas GeoDataFrame,
  Pandas DataFrame, or Earth Engine Feature Collection (but they *must* have
  point geometries). The ':mod:`~geebeam.sampler` submodule also has more options
  for sampling, including custom dataset splits (e.g. train/val/test).
- **Other output formats** — pass ``output_type="webdataset"`` to produce a 
  `WebDataset tar <https://github.com/webdataset/webdataset>`_
  or output_type="tfds"` (requires ``pip install geebeam[tensorflow]``) to 
  write as a Tensorflow dataset. See :doc:`output_types` for more info.
- **API reference** — full parameter documentation for all three pipeline
  functions is in the :doc:`autoapi/geebeam/index` page.

