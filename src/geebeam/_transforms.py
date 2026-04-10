"""Beam _transforms and related utilities"""

import logging
import time
import json
import os

import numpy as np
from apache_beam.io.gcp.gcsio import GcsIO
import apache_beam as beam
import ee

from geebeam import _ee_utils

def _write_json_to_local(json_string, local_path):
    data = json_string.encode('utf-8')
    with open(local_path, mode="wb") as f:
        f.write(data)

def _write_json_to_gcs(json_string, gcs_path):
    data = json_string.encode('utf-8')
    with GcsIO().open(gcs_path, mode="wb") as f:
        f.write(data)

def _convert_to_iterable(val):
    # Convert to iterable
    try:
        _iter_check = iter(val)
    except TypeError:
        return [val]
    else:
        return val

def _write_sidecar_schema(output_path, band_names, extra_metadata_keys, is_gcs):
    """Write a sidecar schema JSON describing the expected features."""
    schema_dict = {"features": {}}
    schema_dict["features"]["md_id"] = "int64"
    schema_dict["features"]["md_y"] = "float"
    schema_dict["features"]["md_x"] = "float"
    schema_dict["features"]["md_split"] = "string"
    for key in extra_metadata_keys:
        schema_dict["features"]["md_" + key] = "float"
    for band in band_names:
        schema_dict["features"]["im_" + band] = "float"

    json_string = json.dumps(schema_dict, indent=2)
    schema_path = os.path.join(output_path, 'schema.json')
    if is_gcs:
        _write_json_to_gcs(json_string, schema_path)
    else:
        # Check if diretory exists
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        _write_json_to_local(json_string, schema_path)

def _split_dataset(element, n_partitions) -> int:
    split_mappings = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    return split_mappings[element['metadata']['split']]

def join_structured_arrays(array_list):
    """Join structured array along features axis"""
    template = array_list[0]
    new_dtype = sum([a.dtype.descr for a in array_list], [])
    merged = np.empty(template.shape, dtype=new_dtype)
    for a in array_list:
        for feat in a.dtype.names:
            merged[feat] = a[feat]
    return merged

def _join_struct_arrays_to_dict(array_list):
    """Join structured array along features axis, return as dict"""
    merged_dict = {}
    for a in array_list:
        for feat in a.dtype.names:
            merged_dict[feat] = a[feat]
    return merged_dict


class EEComputePatch(beam.DoFn):
    """DoFn() for computing EE patch

    config (dict): Dictionary containing configuration settings
        in the following key:value pairs:
            project_id (str): Google Cloud project id
            patch_size (int): Patch size, in pixels, of output chips
            scale (float): Final scale, in m
            target_year (int): Year of prediction
            target_key (str): Name of target data, corresponds to key in self.prep_dict
            inputs_keys (list): Names of input data, correspond to keys in self.prep_dict
            proj (str): Projection, e.g. "EPSG:4326"
    """
    def __init__(self, config, serialized_image, scale_x, scale_y, band_groups):
        self.config = config
        self.serialized_image = serialized_image
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.band_groups = band_groups
        self.initialized = False

    def _initialize_ee(self):
        logging.info(f"Initializing Earth Engine for project: {self.config['project_id']}")
        ee.Initialize(project=self.config['project_id'],
                      opt_url='https://earthengine-highvolume.googleapis.com')
        self.initialized = True

    def process(self, point):
        """Compute a patch of pixel, with upper-left corner defined by the coords."""
        if not self.initialized:
            self._initialize_ee()

        t0 = time.time()
        out_ars = _ee_utils.get_pixels_allbands(
            im=_ee_utils._deserialize(self.serialized_image),
            band_groups=self.band_groups,
            point=point,
            patch_size=self.config['patch_size'],
            scale_x=self.scale_x,
            scale_y=self.scale_y,
            crs_code = self.config['crs']
            )

        out_dict = {'metadata': dict(point)}
        out_dict['array'] = _join_struct_arrays_to_dict(out_ars)
        logging.info(
            f"EE loaded, {point['id']}, took {time.time() - t0:.1f}s"
        )
        yield out_dict

class AddMetadata(beam.DoFn):
    def __init__(self, metadata):
        self.metadata = metadata

    def process(self, example):
        merged_metadata = {
            **example.get("metadata", {}),
            **self.metadata
        }

        yield {
            "array": example["array"],
            "metadata": merged_metadata
        }