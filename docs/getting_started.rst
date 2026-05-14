Getting Started
===============

In this tutorial we will download a set of Landsat 8 image patches from a
region of central-west Brazil and save them as GeoTIFFs on your local machine.
By the end you will have real files you can open in QGIS, rasterio, or any GIS
tool — and everything you need to use ``geebeam`` for your work!

.. note::

   This tutorial runs locally. We'll keep the sample count small (10 patches)
   so everything finishes in under a minute or so. For info on running on
   DataFlow, see :doc:`scaling_up`.

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
- **A Google Cloud project.** Although this tutorial does not cost
  anything (as long as you meet `Earth Engine's Noncommercial/Research Use criteria 
  <https://earthengine.google.com/noncommercial/>`_), the Python Earth Engine
  API requires an active Google Cloud Project. You can create one at 
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

This opens a browser window to login and save credentials locally. You only
need to do this once.

You also need to tell Earth Engine which Google Cloud project to use. 
The easiest way is to detect it from your environment:

.. code-block:: python

   import google.auth
   PROJECT_ID = google.auth.default()[1]   # reads from gcloud config
   print(PROJECT_ID)                       # confirm it's the right project

Or just set it directly::

   PROJECT_ID = "my-gcp-project"


Step 1: Define the image you want
----------------------------------

``geebeam`` downloads image "chips" — fixed-size pixel patches — from any
``ee.Image`` you can build with the Earth Engine Python API. You tell ``geebeam``
*what* to download; it handles the parallelism and file writing.

Here we'll build a cloud-free Landsat 8 composite for 2023:

.. code-block:: python

   import ee
   import geebeam

   ee.Initialize(project=PROJECT_ID) # Uses PROJECT_ID from the last step

   ls8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
       .filterDate("2023-01-01", "2023-12-31")

   ls8_composite = ls8_collection.median().select(
       ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
   )

The image is not downloaded yet — this is just an EE graph definition (the "recipe" for 
  creating the image). Nothing leaves Google's servers until the pipeline runs. 
  When that happens, ``geebeam`` will automatically serialize the  graph definition and 
  send it to the workers to start downloading patches, all with one command!


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
       position='center',
       output_path="./tutorial_output/",
       validation_ratio=0.2,   # 2 of the 10 patches go to the validation split
   )

You will see some Apache Beam log lines scroll by. When it finishes the
``tutorial_output/`` directory will exist.

**What do ``patch_size`` and ``scale`` mean?**
Each chip covers ``128 × 30 m = 3.84 km`` on a side. Increasing ``patch_size``
gives more spatial context. Increasing ``scale`` also gives more spatial context
but at the cost of lower resolution/detail.

**What does ``position`` do?**
By default each sampling point is the *center* of its patch
(``position="center"``). You can change this to ``"top-left"``,
``"top-right"``, ``"bottom-left"``, or ``"bottom-right"`` if your workflow
requires the coordinate to refer to a specific corner.

**Why is ``ls8_composite`` wrapped in a list ``[]``?**
``image_list`` must be a Python list of ``ee.Image`` objects, even
when you only have one image. This is how ``geebeam`` knows how to split
bands across workers when needed (see :doc:`split_processing`). If you pass 
a single ``ee.Image``, ``geebeam`` will wrap it in a list and keep going
(with a warning). Currently, ``ee.ImageCollection`` objects are NOT supported.


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

Each ``.tif`` is a multi-band GeoTIFF "chip" containing all the bands from your image
list. The ``metadata`` Parquet file records the sampling location (``x``, ``y``),
the patch origin (``x_topleft``, ``y_topleft``), the split assignment, and the
file path for each chip.

Open a chip with rasterio to confirm it looks right (you may need to install matplotlib):

.. code-block:: python

   import rasterio
   import matplotlib.pyplot as plt
   import numpy as np

   with rasterio.open("tutorial_output/train/00000.tif") as ds:
       print("CRS:", ds.crs)
       print("Bands:", ds.count)
       print("Shape:", ds.height, "×", ds.width)

       # Read RGB bands (B4=red, B3=green, B2=blue in Landsat 8 L2)
       # Rasterio bands are 1-indexed (start at 1 instead of 0)
       r, g, b = ds.read(3), ds.read(2), ds.read(1)

   rgb = np.stack([r, g, b], axis=-1).astype(float)
   rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)   # stretch to [0, 1]
   plt.imshow(rgb)
   plt.title("Patch 00000 — RGB")
   plt.axis("off")
   plt.show()

You can also read the metadata table:

.. code-block:: python

   import pandas as pd
   import glob

   first_parquet = glob.glob('tutorial_output/metadata-00000*.parquet')[0]
   df = pd.read_parquet(first_parquet)
   print(df[["id", "x", "y", "x_topleft", "y_topleft", "split", "image_path"]])


Step 4. Adding a second image
----------------------

With ``geebeam``, downloading many datasets in a single pipeline is just 
as easy as downloading one. The bands from every image in ``image_list`` 
are stacked into the same chip/patch files:

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

The output TIFFs will now contain 7 bands (6 Landsat + 1 LULC).

What's next
------------

- **Other output formats** — pass ``output_type="webdataset"`` to produce a 
  `WebDataset tar <https://github.com/webdataset/webdataset>`_
  or ``output_type="tfds"`` (requires ``pip install geebeam[tensorflow]``) to
  write as a Tensorflow dataset. See :doc:`output_types` for more info.
- **Scaling up with DataFlow** — geebeam makes it super easy to run large 
  pipelines on Google Cloud Dataflow by providing pre-built Docker images.
  See :doc:`scaling_up` for more.
- **Custom and gridded sampling** — use :func:`~geebeam.run_pipeline` directly
  and pass your own ``sampling_points`` as a Geopandas GeoDataFrame,
  Pandas DataFrame, or Earth Engine Feature Collection (but they *must* have
  point geometries). Or use :func:`~geebeam.grid_and_run_pipeline` to sample on a
  regular grid rather than randomly for when you need complete spatial
  coverage without gaps. See :mod:`~geebeam.sampler`
  for more info, including custom dataset splits (e.g. train/val/test).
- **Split processing of large patches** — Earth Engine limits single request sizes 
  to 50 MB, but geebeam lets you split processing: :doc:`split_processing`
- **API reference** — full parameter documentation for all three pipeline
  functions and the sampler module is on the :mod:`~geebeam` page.


Get Help / Contribute
---------------------

If you run into a problem or have a question, please
`open an issue <https://github.com/kysolvik/geebeam/issues>`_ on GitHub.
Check the existing issues first — your question may already be asked/answered.

Contributions are welcome. Some ways to help:

- **Report a bug or request a feature** — open an issue describing what you
  encountered or what you'd like to see.
- **Add a new feature** — browse the
  `issue tracker <https://github.com/kysolvik/geebeam/issues>`_ for ideas,
  then submit a pull request against the
  `main repository <https://github.com/kysolvik/geebeam>`_.
- **Write examples** — new notebooks or scripts showing real-world usage are
  always appreciated.