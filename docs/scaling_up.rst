Scaling up with Dataflow
========================

The local runner is useful for development and small jobs. For bigger
workloads (e.g. thousands of patches or large images), you can run on Google Cloud
Dataflow (note: Dataflow is a billed resource). Write your ``run_pipeline()`` inside a script and run it with Dataflow
runner options:

.. code-block:: bash

   python my_pipeline.py \
       --runner=DataflowRunner \
       --region=us-east1 \
       --worker_zone=us-east1-b \
       --max_num_workers=8 \
       --temp_location=gs://[my-bucket]/tmp/ \
       --sdk_container_image=kysolvik/geebeam:[version] \
       --machine_type=n2-highmem-2 \
       --experiments=use_runner_v2

Set ``output_path`` to a ``gs://`` path and ``geebeam`` will write directly to
Google Cloud Storage. See the
`Dataflow documentation <https://cloud.google.com/dataflow/docs>`_ for full details.

Make sure to replace [version] in the sdk_container_image URI with the geebeam version 
number installed on your system:

.. code-block:: bash

   python -c "import geebeam;print(geebeam.__version__)"

.. tip::

   Test a small sample of your script locally first with ``--runner=DirectRunner``
   before submitting to Dataflow. This catches errors without using Dataflow 
   resources and costing $$$.


Common DataFlow gotchas
-----------------------

1. Before running, you must `enable the DataFlow API on Google Cloud Console <https://console.developers.google.com/apis/api/dataflow.googleapis.com/overview>`_.

2. If you get an error stating "Subnetwork ''... does not have Private Google Access...", you may need to activate it for your subnetwork (replace us-east1 with your region):


.. code-block:: bash

   gcloud compute networks subnets update default \
      --region=us-east1 \
      --enable-private-ip-google-access
