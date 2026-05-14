Avoiding the 50 MB per-request limit
==============================================

Each call to Earth Engine's ``computePixels`` API is capped at **50 MB** per
response. By default ``geebeam`` combines all bands from all images in
``image_list`` into a single request per patch, so you can hit this limit with
large patches or many bands.

A rough estimate of response size:

.. code-block:: text

   response_bytes ≈ patch_size × patch_size × n_bands × bytes_per_pixel
   e.g. 512 × 512 × 48 bands × 4 bytes (float32) ≈ 50 MB

If you hit an error saying that you've exceeded the max size, pass
``split_processing=True``. This makes ``geebeam`` run **one** ``computePixels``
request **per** image in ``image_list`` instead of one combined request,
so each individual call stays under the limit. If you're already only targeting 
one image, you can split the bands into multiple images. For example, to get an extra large
patch of the ls8 composite:

.. code-block:: python

  import ee
  import geebeam

  ee.Initialize(project=PROJECT_ID)

  ls8_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
      .filterDate("2023-01-01", "2023-12-31")

  ls8_composite = ls8_collection.median()

  ls8_rgb = ls8_composite.select(["SR_B2", "SR_B3", "SR_B4"])
  ls8_nir_swir = ls8_composite.select(["SR_B5", "SR_B6", "SR_B7"])

  geebeam.sample_and_run_pipeline(
      image_list=[ls8_rgb, ls8_nir_swir],
      split_processing=True,  # one EE request per image
      sampling_region=ee.Geometry.Rectangle(-55.0, -12.0, -50.0, -16.0),
      n_sample=10,
      patch_size=1024,
      scale=30,
      crs="EPSG:4326",
      project=PROJECT_ID,
      output_path="./tutorial_output_split/",
  )

The output is identical — bands from all images are still merged into the same
chip files. The only difference is the number of round-trips to Earth Engine
per patch: one per image rather than one total. This is slower, so unless you 
run into errors it's better to leave it at the default (``False``).