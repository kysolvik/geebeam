"""Prepare and run Beam pipeline to download image 'chips' from Earth Engine"""

import ee
import geopandas as gpd
import pandas as pd
import numpy as np
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow_data_validation as tfdv
from tfx_bsl.coders import example_coder
import os
from google.protobuf.json_format import MessageToJson

from geebeam import ee_utils, sampler, transforms


def _check_if_localrunner(beam_options):
    """Fixes gRPC timeout issue for local runners."""
    runner = beam_options.get_all_options()['runner']
    if runner is None or runner in ['DirectRunner', 'PrismRunner']:
        return True
    else:
        return False


def _prepare_run_metadata(config):
    ee.Initialize(project=config['project_id'])

    proj = ee.Projection(config['crs']).atScale(config['scale'])
    proj_dict = proj.getInfo()

    scale_x = proj_dict['transform'][0]
    scale_y = -proj_dict['transform'][4]

    return scale_x, scale_y

def run_pipeline(
        image_list: list[ee.Image],
        output_path: str,
        project: str,
        patch_size: int,
        scale: float,
        n_sample: int,
        sampling_region: str | gpd.GeoDataFrame | ee.Geometry | None = None,
        sampling_points: pd.DataFrame | gpd.GeoDataFrame | ee.Geometry | None = None,
        validation_ratio: float = 0.2,
        crs: str = 'EPSG:4326',
        random_seed: int = None,
        split_processing: bool = False,
        extra_metadata: dict = {},
        beam_options_dict: dict[str] = {}
        ) -> None:
    """Run a Beam pipeline to download image chips from Earth Engine.

    Args:
        image_list: A list of image identifiers to process.
        sampling_region: The region for sampling images.
        sampling_points: Center-points of locations to sample from.
        output_path: The path where output will be saved.
        project: The Google Cloud project ID.
        patch_size: The size of the patches to be processed.
        scale: The scale factor for image processing.
        n_sample: The number of samples to take.
        validation_ratio: The ratio of data to use for validation. Defaults to 0.2.
        crs: The coordinate reference system. Defaults to 'EPSG:4326'.
        random_seed: Seed for random number generation. Defaults to None.
        split_processing: Flag to indicate if processing should be split. Defaults to False.
        extra_metadata: Additional metadata to include. Defaults to an empty dictionary.
        beam_options_dict: Options for the Beam pipeline. Defaults to an empty dictionary.
    """
    import logging

    logging.getLogger().setLevel(logging.INFO)

    rng = np.random.default_rng(random_seed)

    # Parses from command line and/or retrieves from dict. Note that dict takes precedent.
    beam_options = PipelineOptions(beam_options_dict)

    # Set up configuration dict to pass along
    config = {
        'project_id': project,
        'patch_size': patch_size,
        'scale': scale,
        'n_sample': n_sample,
        'validation_ratio':validation_ratio,
        'crs': crs
    }

    # Get sample points
    if sampling_points is not None:
        sampled_points = sampler.process_sampling_points(sampling_points, target_crs=config['crs'])
    else:
        roi = sampler.get_roi(sampling_region, target_crs=config['crs'])
        sampled_points  = sampler.sample_random_points(roi, config['n_sample'], rng)

    # Split into training and validation
    input_records = sampler.split_train_validation(
        sampled_points, config['validation_ratio'], rng=rng
        ).to_dict('records')

    # Pre-run info:
    scale_x, scale_y = _prepare_run_metadata(config)

    # Set up pipeline
    beam_options = PipelineOptions(beam_options,
                                   project=config['project_id'],
                                   save_main_session=True,
                                   use_public_ips=False
                                   )

    # Check if a local runner
    is_local = _check_if_localrunner(beam_options)

    # Prepare and serialize inputs
    # band_groups is a list of lists containing bands to export
    # if split_processing = False, will contain one list with all bands
    # if split_processing = True,  contains separate band_lists for each image in image_list
    prepped_image, band_groups = ee_utils.build_prepped_image(image_list, split_processing=split_processing)
    serialized_image = ee_utils.serialize(prepped_image)

    # Execute pipeline
    with beam.Pipeline(options=beam_options) as pipeline:

        points = pipeline | 'Create points' >> beam.Create(input_records)

        if is_local:
            batches = (
                points
                | 'Add Dummy Key' >> beam.Map(lambda x: (None, x))
                | 'Reshuffle' >> beam.Reshuffle()
                | 'Force Single Batches' >> beam.GroupIntoBatches(batch_size=1)
                | 'Extract' >> beam.FlatMap(lambda x: x[1])
            )
        else:
            batches = points

        training_data, validation_data = (
            batches
            | 'Get patch' >> beam.ParDo(transforms.EEComputePatch(
                config,
                serialized_image,
                scale_x,
                scale_y,
                band_groups
                ))
            | 'Add metadata' >> beam.ParDo(transforms.AddMetadata(extra_metadata))
            | 'Split dataset' >> beam.Partition(transforms.split_dataset, 2)
        )

        # Convert to TF examples
        training_examples = (
            training_data
            | 'Train to tf.Example' >> beam.Map(transforms.dict_to_example)
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
            os.path.join(output_path, 'stats.tfrecord'))

        # Write out examples
        (training_examples
         | 'Write training' >> transforms.WriteTFExample(
             os.path.join(output_path, 'training'))
        )
        if config['validation_ratio'] > 0:
            validation_examples = (
                validation_data
                | 'Val to tf.Example' >> beam.Map(transforms.dict_to_example)
            )

            (validation_examples
            | 'Write validation' >> transforms.WriteTFExample(
                os.path.join(output_path, 'validation'))
            )


    # NOTE: testing a few different formats.
    # Need to simplify once we decide on one
    # Infer schema
    stats = tfdv.load_statistics(
        os.path.join(output_path, 'stats.tfrecord')
    )

    schema = tfdv.infer_schema(stats)

    tfdv.write_schema_text(
        schema,
        os.path.join(output_path, 'schema.pbtxt')
    )

    # Also write as pbtxt, easier to read
    tfdv.write_stats_text(
        stats,
        os.path.join(output_path, 'stats.pbtxt')
    )

    # Also write stats and schema as jsons
    out_schema_json = os.path.join(output_path, 'schema.json')
    out_stats_json = os.path.join(output_path, 'stats.json')
    if output_path.startswith('gs://'):
        transforms.write_json_to_gcs(MessageToJson(schema), out_schema_json)
        transforms.write_json_to_gcs(MessageToJson(stats), out_stats_json)
    else:
        transforms.write_json_to_local(MessageToJson(schema), out_schema_json)
        transforms.write_json_to_local(MessageToJson(stats), out_stats_json)


