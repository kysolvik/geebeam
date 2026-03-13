"""Prepare and run Beam pipeline to download image 'chips' from Earth Engine

Example execution:
    python geebeam_main.py \
        --output_path gs://aic-fire-amazon/results/ \
        --sampling_region ./data/Limites_RAISG_2025/Lim_Raisg.shp \
        --runner DataflowRunner \
        --experiments=use_runner_v2 \
        --max_num_workers=16 \
        --num_workers=8 \
        --requirements_file ./pipeline_requirements.txt

"""
import ee
import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, DirectOptions
import tensorflow_data_validation as tfdv
from tfx_bsl.coders import example_coder
import argparse
import os
from google.protobuf.json_format import MessageToJson
import time

from geebeam import ee_utils, sampler, transforms
import os
os.environ['BEAM_SDK_WORKER_STALL_TIMEOUT'] = '0'
import logging

class TestSleepDoFn(beam.DoFn):
    def process(self, element):
        logging.info(f"Starting work on: {element}")
        # Simulate the 10-second GEE request
        time.sleep(10)
        logging.info(f"Finished work on: {element}")
        yield f"processed_{element}"

def prepare_run_metadata(config):
    ee.Initialize(project=config['project_id'])

    proj = ee.Projection(config['proj']).atScale(config['scale'])
    proj_dict = proj.getInfo()

    scale_x = proj_dict['transform'][0]
    scale_y = -proj_dict['transform'][4]

    return scale_x, scale_y

def run(config, image_list, random_seed=None, split_processing=False, extra_metadata={},
        output_path=None, sampling_region=None):
    import logging

    logging.getLogger().setLevel(logging.INFO)

    rng = np.random.default_rng(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        required=False,
        help="Directory to save TFRecord files (local or GCS).",
    )
    parser.add_argument(
        "--sampling_region",
        required=False,
        help="Local geopandas-readable file of region to sample from randomly."
    )

    # Beam args are leftover after parsing known args
    args, beam_args = parser.parse_known_args()

    if args.sampling_region is None:
        args.sampling_region = sampling_region
    if args.output_path is None:
        args.output_path = output_path

    # Randomly sample points
    roi = sampler.get_roi(args.sampling_region)
    sampled_points  = sampler.sample_random_points(roi, config['n_sample'], rng)

    # Split into training and validation
    input_records = sampler.split_train_validation(
        sampled_points, config['validation_ratio'], rng=rng
        ).to_dict('records')

    # Pre-run info:
    scale_x, scale_y = prepare_run_metadata(config)

#     # Set up pipeline
#     extra_args = [
#     # Increase the keepalive time and timeout to avoid the 5-min wall
#     "--experiments=grpc_keepalive_time_ms=600000",    # 10 minutes
#     "--experiments=grpc_keepalive_timeout_ms=600000", # 10 minutes
#     # Force the data buffer to flush more frequently
#     "--experiments=data_buffer_time_limit_ms=1000",   # 1 second
# ]
    beam_options = PipelineOptions([
    "--runner=DirectRunner",
    "--environment_type=LOOPBACK",
    "--direct_num_workers=1",
    "--experiments=max_bundle_size=1",
    "--experiments=max_bundle_time_sec=5",
    # This ensures the worker doesn't time out the gRPC link
    "--experiments=grpc_keepalive_timeout_ms=600000"
])
#     print(beam_args + extra_args)
#     beam_options = DirectOptions(
#             beam_args + extra_args,
#             project=config['project_id'],
#             temp_location='gs://aic-fire-amazon/tmp/',
#             save_main_session=True,
#             use_public_ips=False,
#             max_bundle_time_sec=60,
#             max_bundle_size=1,
#
#             )

    # Prepare and serialize inputs
    # band_groups is a list of lists containing bands to export
    # if split_processing = False, will contain one list with all bands
    # if split_processing = True,  contains separate band_lists for each image in image_list
    prepped_image, band_groups = ee_utils.build_prepped_image(image_list, split_processing=split_processing)
    serialized_image = ee_utils.serialize(prepped_image)

    # Execute pipeline
    with beam.Pipeline(options=beam_options) as pipeline:

        # Gather data and split
      #  training_data, validation_data = (
      (
            pipeline
            | 'Create points' >> beam.Create(input_records)
            | 'Reshuffle' >> beam.Reshuffle()
            | 'Add Dummy Key' >> beam.Map(lambda x: (None, x))
            | 'Force Single Batches' >> beam.GroupIntoBatches(batch_size=1)
            | 'Extract and Process' >> beam.Map(lambda x: x[1][0]) # Get the single element out of the list
            | 'Test' >> beam.ParDo(TestSleepDoFn())
      )
      #      | 'Get patch' >> beam.ParDo(transforms.EEComputePatch(config, serialized_image, scale_x, scale_y, band_groups))
#             | 'Add metadata' >> beam.ParDo(transforms.AddMetadata(extra_metadata))
#             | 'Split dataset' >> beam.Partition(transforms.split_dataset, 2)
#         )
#
#         # Convert to TF examples
#         training_examples = (
#             training_data
#             | 'Train to tf.Example' >> beam.Map(transforms.dict_to_example)
#         )
#         # Calculate stats on training data
#         decoder = example_coder.ExamplesToRecordBatchDecoder()
#         stats = (
#             training_examples
#             | 'Batch' >> beam.BatchElements(
#                 min_batch_size=10,
#                 max_batch_size=100)
#             | 'Decode to arrow' >> beam.Map(lambda b: decoder.DecodeBatch(b))
#             | 'Generate Statistics' >> tfdv.GenerateStatistics()
#         )
#         stats | 'Write stats' >> tfdv.WriteStatisticsToTFRecord(
#             os.path.join(args.output_path, 'stats.tfrecord'))
#
#         # Write out examples
#         (training_examples
#          | 'Write training' >> transforms.WriteTFExample(
#              os.path.join(args.output_path, 'training'))
#         )
#         if config['validation_ratio'] > 0:
#             validation_examples = (
#                 validation_data
#                 | 'Val to tf.Example' >> beam.Map(transforms.dict_to_example)
#             )
#
#             (validation_examples
#             | 'Write validation' >> transforms.WriteTFExample(
#                 os.path.join(args.output_path, 'validation'))
#             )
#
#
#     # NOTE: testing a few different formats.
#     # Need to simplify once we decide on one
#     # Infer schema
#     stats = tfdv.load_statistics(
#         os.path.join(args.output_path, 'stats.tfrecord')
#     )
#
#     schema = tfdv.infer_schema(stats)
#
#     tfdv.write_schema_text(
#         schema,
#         os.path.join(args.output_path, 'schema.pbtxt')
#     )
#
#     # Also write as pbtxt, easier to read
#     tfdv.write_stats_text(
#         stats,
#         os.path.join(args.output_path, 'stats.pbtxt')
#     )
#
#     # Also write stats and schema as jsons
#     out_schema_json = os.path.join(args.output_path, 'schema.json')
#     out_stats_json = os.path.join(args.output_path, 'stats.json')
#     if args.output_path.startswith('gs://'):
#         transforms.write_json_to_gcs(MessageToJson(schema), out_schema_json)
#         transforms.write_json_to_gcs(MessageToJson(stats), out_stats_json)
#     else:
#         transforms.write_json_to_local(MessageToJson(schema), out_schema_json)
#         transforms.write_json_to_local(MessageToJson(stats), out_stats_json)


