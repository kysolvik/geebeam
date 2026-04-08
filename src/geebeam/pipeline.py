"""Prepare and run Beam pipeline to download image 'chips' from Earth Engine"""

import warnings
import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from apache_beam.options.pipeline_options import PipelineOptions

from geebeam import ee_utils, sampler, transforms


def _check_if_localrunner(pipeline_options):
    """Fixes gRPC timeout issue for local runners."""
    runner = pipeline_options.get_all_options()['runner']
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
        output_type: str = 'tfds',
        sampling_region: str | gpd.GeoDataFrame | ee.Geometry | None = None,
        sampling_points: pd.DataFrame | gpd.GeoDataFrame | ee.Geometry | None = None,
        validation_ratio: float = 0.2,
        crs: str = 'EPSG:4326',
        random_seed: int = None,
        split_processing: bool = False,
        extra_metadata: dict = {},
        beam_options: dict[str] | list[str] | None = None,
        dataset_version: str = '1.0.0',
        dataset_name: str = 'geebeam_dataset'
        ) -> None:
    """Run a Beam pipeline to download image chips from Earth Engine.

    Args:
        image_list: A list of image identifiers to process.
        sampling_region: The region for sampling images.
        sampling_points: Locations to sample from, specifying upper-left of box.
        output_path: The path where output will be saved.
        output_type: 'tfrecord' (raw tfrecords) or 'tfds' (tensorflow-dataset).
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

    # Set up configuration dict to pass along
    config = {
        'project_id': project,
        'patch_size': patch_size,
        'scale': scale,
        'n_sample': n_sample,
        'validation_ratio':validation_ratio,
        'crs': crs
    }

    # Parses from command line and/or retrieves from dict. Note that dict takes precedent.
    if isinstance(beam_options, dict):
        pipeline_options = PipelineOptions(
            **beam_options,
            project=config['project_id'],
            save_main_session=True,
            )
    elif isinstance(beam_options, list):
        warnings.warn('Creating PipelineOptions from beam_options list.'
                      ' Ignores command-line beam options.')
        pipeline_options = PipelineOptions(
            beam_options,
            project=config['project_id'],
            save_main_session=True,
            )
    else:
        pipeline_options = PipelineOptions(
            project=config['project_id'],
            save_main_session=True,
            )

    # Get sample points
    if sampling_points is not None:
        sampled_points = sampler.process_sampling_points(sampling_points, target_crs=config['crs'])
    else:
        roi = sampler.get_roi(sampling_region, image_list, target_crs=config['crs'])
        sampled_points  = sampler.sample_random_points(roi, config['n_sample'], rng)

    # Split into training and validation
    input_records = sampler.split_train_validation(
        sampled_points, config['validation_ratio'], rng=rng
        ).to_dict('records')

    # Pre-run info:
    scale_x, scale_y = _prepare_run_metadata(config)

    # Check if a local runner
    is_local = _check_if_localrunner(pipeline_options)

    # Prepare and serialize inputs
    # band_groups is a list of lists containing bands to export
    # if split_processing = False, will contain one list with all bands
    # if split_processing = True,  contains separate band_lists for each image in image_list
    prepped_image, band_groups, all_bands = ee_utils.build_prepped_image(image_list, split_processing=split_processing)
    serialized_image = ee_utils.serialize(prepped_image)


    # Execute pipeline based on output type:
    if output_type == 'tfrecord':
        # Write sidecar schema before pipeline execution
        extra_keys = list(extra_metadata.keys())
        transforms.write_sidecar_schema(output_path, all_bands, extra_keys,
                                        is_gcs=output_path.startswith('gs://'))
        from geebeam import tfrecord_writer
        tfrecord_writer.run_tfrecord_export(
            input_records=input_records,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            is_local=is_local,
            pipeline_options=pipeline_options
        )
    elif output_type == 'tfds':
        from geebeam import tfds_writer
        tfds_writer.run_tfds_export(
            input_records=input_records,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            all_bands=all_bands,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            pipeline_options=pipeline_options,
            dataset_name=dataset_name,
            dataset_version=dataset_version
        )
    elif output_type == 'tif' or output_type == 'tiff':
        from geebeam import tiff_writer
        tiff_writer.run_tiff_export(
            input_records=input_records,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            is_local=is_local,
            pipeline_options=pipeline_options
        )
    else:
        raise ValueError(f"output_type {output_type} not implemented. Options are ['tfds', 'tfrecord', 'tiff']")