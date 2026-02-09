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
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow_data_validation as tfdv
from tfx_bsl.coders import example_coder
import argparse
import os
from google.protobuf.json_format import MessageToJson

from geebeam import ee_utils, sampler, transforms


def prepare_run_metadata(config):
    ee.Initialize(project=config['project_id'])

    proj = ee.Projection(config['proj']).atScale(config['scale'])
    proj_dict = proj.getInfo()

    scale_x = proj_dict['transform'][0]
    scale_y = -proj_dict['transform'][4]

    return scale_x, scale_y

def run(config, image_list, random_seed=None):
    import logging

    logging.getLogger().setLevel(logging.INFO)

    rng = np.random.default_rng(random_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory to save TFRecord files (local or GCS).",
    )
    parser.add_argument(
        "--sampling_region",
        required=True,
        help="Local geopandas-readable file of region to sample from randomly."
    )

    # Beam args are leftover after parsing known args
    args, beam_args = parser.parse_known_args()

    # Randomly sample points
    roi = sampler.get_roi(args.sampling_region)
    sampled_points  = sampler.sample_random_points(roi, config['n_sample'], rng)

    # Split into training and validation
    input_records = sampler.split_train_validation(
        sampled_points, config['validation_ratio'], rng=rng
        ).to_dict('records')

    # Pre-run info:
    scale_x, scale_y = prepare_run_metadata(config)

    # Set up pipeline
    beam_options = PipelineOptions(beam_args,
                                   project=config['project_id'],
                                   temp_location='gs://aic-fire-amazon/tmp/',
                                   save_main_session=True,
                                   use_public_ips=False
                                   )

    # Prepare and serialize inputs
    prepped_image = ee_utils.build_prepped_image(image_list)
    serialized_image = ee_utils.serialize(prepped_image)

    # Execute pipeline
    with beam.Pipeline(options=beam_options) as pipeline:

        # Gather data and split
        training_data, validation_data = (
            pipeline
            | 'Create points' >> beam.Create(input_records)
            | 'Get patch' >> beam.ParDo(transforms.EEComputePatch(config, serialized_image, scale_x, scale_y))
            | 'Split dataset' >> beam.Partition(transforms.split_dataset, 2)
        )

        # Convert to TF examples
        training_examples = (
            training_data
            | 'Train to tf.Example' >> beam.Map(transforms.dict_to_example)
        )
        validation_examples = (
            validation_data
            | 'Val to tf.Example' >> beam.Map(transforms.dict_to_example)
        )

        # Calculate stats on training data
        decoder = example_coder.ExamplesToRecordBatchDecoder()
        stats = (
            training_examples
            | 'Batch' >> beam.BatchElements(
                min_batch_size=10,
                max_batch_size=100)
            | 'Decode to arrow' >> beam.Map(lambda b: decoder.DecodeBatch(b))
            | 'Generate Statistics' >> tfdv.GenerateStatistics()
        )
        stats | 'Write stats' >> tfdv.WriteStatisticsToTFRecord(
            os.path.join(args.output_path, 'stats.tfrecord'))

        # Write out examples
        (training_examples
         | 'Write training' >> transforms.WriteTFExample(
             os.path.join(args.output_path, 'training'))
        )
        (validation_examples
         | 'Write validation' >> transforms.WriteTFExample(
             os.path.join(args.output_path, 'validation'))
        )


    # NOTE: testing a few different formats.
    # Need to simplify once we decide on one
    # Infer schema
    stats = tfdv.load_statistics(
        os.path.join(args.output_path, 'stats.tfrecord')
    )

    schema = tfdv.infer_schema(stats)

    tfdv.write_schema_text(
        schema,
        os.path.join(args.output_path, 'schema.pbtxt')
    )

    # Also write as pbtxt, easier to read
    tfdv.write_stats_text(
        stats,
        os.path.join(args.output_path, 'stats.pbtxt')
    )

    # Also write stats and schema as jsons
    out_schema_json = os.path.join(args.output_path, 'schema.json')
    out_stats_json = os.path.join(args.output_path, 'stats.json')
    if args.output_path.startswith('gs://'):
        transforms.write_json_to_gcs(MessageToJson(schema), out_schema_json)
        transforms.write_json_to_gcs(MessageToJson(stats), out_stats_json)
    else:
        transforms.write_json_to_local(MessageToJson(schema), out_schema_json)
        transforms.write_json_to_local(MessageToJson(stats), out_stats_json)


