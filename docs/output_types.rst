Output Types
============

geebeam supports four output formats, selected with the ``output_type``
argument. The right choice depends on your downstream tooling:

.. list-table::
   :header-rows: 1
   :widths: 15 15 35 35

   * - Format
     - ``output_type``
     - Best uses
     - Extra dependencies required?
   * - GeoTIFF
     - ``"tiff"`` *(default)*
     - GIS tools, rasterio, any language
     - No
   * - WebDataset
     - ``"webdataset"``
     - PyTorch / streaming data pipelines
     - No
   * - TensorFlow Datasets
     - ``"tfds"``
     - TensorFlow ``tf.data`` pipelines
     - Yes (``geebeam[tensorflow]``)
   * - TFRecord
     - ``"tfrecord"``
     - Automatic dataset statistics with TFDV
     - Yes (``geebeam[tensorflow]``)


GeoTIFF (default)
-----------------

Each patch is written as a multi-band GeoTIFF, with a
Parquet file containing metadata for all patches.

.. code-block:: python

   geebeam.sample_and_run_pipeline(
       ...,
       output_type="tiff",   # this is the default
       output_path="./output/",
   )

**Output structure:**

.. code-block:: text

   output/
   ├── train/
   │   ├── 00000.tif
   │   ├── 00001.tif
   │   └── ...
   ├── validation/
   │   └── ...
   └── metadata-00000-of-00001.parquet

Each ``.tif`` contains all bands from all images in ``image_list``, in the
order they were passed. The Parquet file has one row per patch with columns
``id``, ``x``, ``y``, ``x_topleft``, ``y_topleft``, ``split``,
``image_path``, and any columns from ``extra_metadata``.

**Reading the output:**

.. code-block:: python

   import rasterio
   import pandas as pd

   with rasterio.open("output/train/00000.tif") as ds:
       data = ds.read()          # shape: (n_bands, height, width)
       print(ds.descriptions)    # band names

   df = pd.read_parquet("output/metadata-00000-of-00001.parquet")

GeoTIFF is the most portable format and the best starting point if you are
not sure which format to use.


WebDataset
----------

Patches are written as sharded ``.tar`` archives in
`WebDataset <https://github.com/webdataset/webdataset>`_ format. Each sample
inside a shard is a pair of files: a GeoTIFF (``{id}.tif``) and a JSON
metadata file (``{id}.json``).

.. code-block:: python

   geebeam.sample_and_run_pipeline(
       ...,
       output_type="webdataset",
       output_path="./output/",
   )

**Output structure:**

.. code-block:: text

   output/
   ├── train-<worker-id>-000000.tar
   ├── train-<worker-id>-000001.tar
   └── validation-<worker-id>-000000.tar

Each ``.tar`` contains alternating ``.tif`` and ``.json`` entries. The
worker ID in the filename is a short UUID that prevents shard collisions
when multiple Beam workers write in parallel.

**Reading the output:**

.. code-block:: python

   import webdataset as wds
   import rasterio
   import io

   dataset = (
       wds.WebDataset("output/train-*.tar")
       .decode()
   )

   for sample in dataset:
       tif_bytes = sample["tif"]
       metadata  = sample["json"]   # already decoded to a dict
       with rasterio.open(io.BytesIO(tif_bytes)) as ds:
           data = ds.read()

WebDataset works well with PyTorch ``DataLoader`` and is a good choice for
large-scale training pipelines that stream data directly from Google Cloud
Storage without materialising it locally.


TensorFlow Datasets
-------------------

Patches are written as a
`TensorFlow Datasets <https://www.tensorflow.org/datasets>`_ (TFDS) custom
dataset. This format integrates directly with ``tf.data`` and the TFDS
catalogue.

**Install:**

.. code-block:: bash

   pip install geebeam[tensorflow]

**Usage:**

.. code-block:: python

   geebeam.sample_and_run_pipeline(
       ...,
       output_type="tfds",
       output_path="./output/",
       dataset_name="my_dataset",     # used as the TFDS dataset name
       dataset_version="1.0.0",       # must be a valid semver string
   )

**Reading the output:**

.. code-block:: python

   import tensorflow_datasets as tfds

   ds = tfds.load("my_dataset", data_dir="./output/", split="train")
   for example in ds.take(1):
       print(example.keys())

The TFDS format is the best choice for TensorFlow-native training pipelines.
It handles split management, shuffling, and prefetching automatically through
the standard ``tf.data`` API.


TFRecord
--------

.. warning::

   TFRecord output is **not recommended** for most use cases. Use
   ``"tiff"`` or ``"tfds"`` instead. Choose TFRecord only if you
   specifically need to compute dataset statistics with
   `TensorFlow Data Validation <https://www.tensorflow.org/tfx/data_validation/get_started>`_
   (TFDV) — that is the only thing this format provides over ``"tfds"`` that
   does not also come with significant drawbacks (no standard loading API,
   harder to inspect, schema coupling).

**Install:**

.. code-block:: bash

   pip install geebeam[tensorflow]

**Usage:**

.. code-block:: python

   geebeam.sample_and_run_pipeline(
       ...,
       output_type="tfrecord",
       output_path="./output/",
   )

**Output structure:**

.. code-block:: text

   output/
   ├── train/
   │   └── *.tfrecord
   ├── validation/
   │   └── *.tfrecord
   ├── schema.json      ← feature names and types
   └── stats.tfrecord   ← TFDV statistics (training split only)

The pipeline automatically computes TFDV statistics over the training split
and writes them alongside the records. This is the primary reason to choose
this format — if you want to validate feature distributions, detect anomalies,
or generate a data schema for a TFX pipeline. If you do not need TFDV stats,
``"tfds"`` gives you a better reading API with no extra cost.
