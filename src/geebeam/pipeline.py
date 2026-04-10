"""Prepare and run Beam pipeline to download image 'chips' from Earth Engine"""

import warnings
import ee
import geopandas as gpd
import pandas as pd
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
        sampling_points: pd.DataFrame | gpd.GeoDataFrame | ee.FeatureCollection,
        output_type: str = 'tiff',
        crs: str = 'EPSG:4326',
        split_processing: bool = False,
        extra_metadata: dict = {},
        beam_options: dict[str] | list[str] | None = None,
        dataset_version: str = '1.0.0',
        dataset_name: str = 'geebeam_dataset'
        ) -> None:
    """Run a Beam pipeline to download image chips from Earth Engine.

    Args:
        image_list: A list of image identifiers to process.
        sampling_points: Locations to sample from, specifying upper-left of box.
        output_path: The path where output will be saved.
        output_type: 'tfrecord' (raw tfrecords) or 'tfds' (tensorflow-dataset).
        project: The Google Cloud project ID.
        patch_size: The size of the patches to be processed.
        scale: The scale factor for image processing.
        crs: The coordinate reference system. Defaults to 'EPSG:4326'.
        split_processing: Flag to indicate if processing should be split. Defaults to False.
        extra_metadata: Additional metadata to include. Defaults to an empty dictionary.
        beam_options_dict: Options for the Beam pipeline. Defaults to an empty dictionary.
    """
    import logging

    logging.getLogger().setLevel(logging.INFO)

    # Set up configuration dict to pass along
    config = {
        'project_id': project,
        'patch_size': patch_size,
        'scale': scale,
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
    input_records, splits = sampler._process_sampling_points(sampling_points, target_crs=config['crs'])

    # Pre-run info:
    scale_x, scale_y = _prepare_run_metadata(config)

    # Check if a local runner. If so, add longer job timeout to fix grpcio timeout issue
    is_local = _check_if_localrunner(pipeline_options)
    if is_local:
        logging.warning('Running on local runner. Setting beam job_server_timeout'
                        ' to 9999999 seconds to avoid grpcio timeout errors.')
        pipeline_options_dict = pipeline_options.get_all_options(drop_default=True)
        pipeline_options_dict['job_server_timeout'] = 9999999
        pipeline_options = PipelineOptions.from_dictionary(pipeline_options_dict)

    # Prepare and serialize inputs
    # band_groups is a list of lists containing bands to export
    # if split_processing = False, will contain one list with all bands
    # if split_processing = True,  contains separate band_lists for each image in image_list
    prepped_image, band_groups, all_bands = ee_utils.build_prepped_image(image_list, split_processing=split_processing)
    serialized_image = ee_utils.serialize(prepped_image)

    # Execute pipeline based on output type:
    if output_type == 'tfrecord':
        try:
            from geebeam import tfrecord_writer
        except ImportError:
            raise ImportError(
                "Missing dependencies for tfrecord writer. "
                "Install them with `pip install geebeam[tensorflow]`"
            )
        # Write sidecar schema before pipeline execution
        extra_keys = list(extra_metadata.keys())
        transforms.write_sidecar_schema(output_path, all_bands, extra_keys,
                                        is_gcs=output_path.startswith('gs://'))
        tfrecord_writer.run_tfrecord_export(
            input_records=input_records,
            splits=splits,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            pipeline_options=pipeline_options
        )
    elif output_type == 'tfds':
        try:
            from geebeam import tfds_writer
        except ImportError:
            raise ImportError(
                "Missing dependencies for TFDS writer. "
                "Install them with `pip install geebeam[tensorflow]`"
            )
        tfds_writer.run_tfds_export(
            input_records=input_records,
            splits=splits,
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
            splits=splits,
            output_path=output_path,
            config=config,
            serialized_image=serialized_image,
            band_groups=band_groups,
            scale_x=scale_x,
            scale_y=scale_y,
            extra_metadata=extra_metadata,
            pipeline_options=pipeline_options
        )
    else:
        raise ValueError(f"output_type {output_type} not implemented. Options are ['tfds', 'tfrecord', 'tiff']")

def sample_and_run_pipeline(
        sampling_region: str | gpd.GeoDataFrame | ee.Geometry,
        n_sample: int,
        validation_ratio: float = 0,
        random_seed: int = 0,
        crs: str = 'EPSG:4326',
        *args,
        **kwargs
        ) -> None:

    sample_points = sampler.sample_region_random(
        roi=sampling_region,
        n_sample=n_sample,
        random_seed=random_seed,
        crs=crs,
    )

    if validation_ratio > 0:
        sample_points = sampler.split_sets(
            sample_points, split_names=['train','validation'], split_ratios=[1-validation_ratio,validation_ratio],
            random_seed=random_seed
        )

    return run_pipeline(sampling_points=sample_points, crs=crs, *args, **kwargs)

def grid_and_run_pipeline(
        sampling_region: str | gpd.GeoDataFrame | ee.Geometry,
        validation_ratio: float,
        scale: float,
        stride: int,
        crs: str = 'EPSG:4326',
        random_seed: int = 0,
        *args,
        **kwargs
        ) -> None:

    sample_points = sampler.sample_region_grid(
        roi=sampling_region,
        crs='EPSG:4326',
        stride=stride,
        scale=scale,
    )

    if validation_ratio > 0:
        sample_points = sampler.split_sets(
            sample_points, split_names=['train','validation'], split_ratios=[1-validation_ratio,validation_ratio],
            random_seed=random_seed
        )

    return run_pipeline(sampling_points=sample_points, crs=crs, scale=scale, *args, **kwargs)
