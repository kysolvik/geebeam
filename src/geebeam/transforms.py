"""Beam transforms and related utilities"""

import logging
import io
import time

import numpy as np
import tensorflow as tf
import ee
import apache_beam as beam

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def dict_to_example(element):
    """"Convert structured numpy array to tf.Example proto."""
    # First add metadata
    feature = {
        'id': _int64_feature(element['id']),
        'lat': _float_feature(element['lat']),
        'lon': _float_feature(element['lon']),
    }

    # Build image feature with named bands
    for im_feat in element['array'].dtype.names:
        feature[im_feat] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = element['array'][im_feat].flatten()))

    # Build example and serialize
    return tf.train.Example(
        features = tf.train.Features(feature = feature)).SerializeToString()

def array_to_example(structured_array):
    """"Convert structured numpy array to tf.Example proto."""
    feature = {}
    for f in structured_array.dtype.names:
        feature[f] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = structured_array[f].flatten()))
    return tf.train.Example(
        features = tf.train.Features(feature = feature))

def split_dataset(element, n_partitions) -> int:
    split_mappings = {
        'train': 0,
        'val': 1,
        'test': 2
    }
    return split_mappings[element['split']]


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
    def __init__(self, config, serialized_image, scale_x, scale_y):
        self.config = config
        self.serialized_image = serialized_image
        self.scale_x = scale_x
        self.scale_y = scale_y

    def deserialize(self, obj_json):
        return ee.deserializer.fromJSON(obj_json)

    def setup(self):
        print(f"Initializing Earth Engine for project: {self.config['project_id']}")
        logging.warning("EE setup: starting")
        credentials = ee.ServiceAccountCredentials(
            'earth-engine-calls@ksolvik-misc.iam.gserviceaccount.com',
            '/home/ksolvik/.ssh/ksolvik-misc-70411b88d818.json'
        )
        ee.Initialize(credentials=credentials,
                      project=self.config['project_id'],
                      opt_url='https://earthengine-highvolume.googleapis.com')
        logging.warning("EE setup: finished")
        ee.data.setDeadline(900000)
        self.prepped_image = self.deserialize(self.serialized_image)
        self.last_init_time = 0


    def process(self, point):
        """Compute a patch of pixel, with upper-left corner defined by the coords."""
        t0 = time.time()
        if (t0 - self.last_init_time > 180):
            self.setup()
        logging.warning(f"EE start {point['id']}")


        # Make a request object.
        request = {
            'expression': self.prepped_image,
            'fileFormat': 'NPY',
            'grid': {
                'dimensions': {
                    'width': self.config['patch_size'],
                    'height':self.config['patch_size']
                },
                'affineTransform': {
                    'scaleX': self.scale_x,
                    'shearX': 0,
                    'translateX': point['lon'],
                    'shearY': 0,
                    'scaleY': self.scale_y,
                    'translateY': point['lat']
                },
                'crsCode': 'EPSG:4326',
            },
        }

        try:
            raw = ee.data.computePixels(request)
        except Exception as e:
            if "DEADLINE_EXCEEDED" in str(e) or "RST_STREAM" in str(e):
                print(str(e))
                # Rerun setup
                self.setup()
                time.sleep(2)  # Give the gRPC loop a chance to catch up
                raw = ee.data.computePixels(request)
            raise e
        logging.warning(f"EE bytes: {len(raw)}")

        if not raw:
            raise RuntimeError("Empty EE response")

        try:
            arr = np.load(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(f"Failed to load NPY for coords {point['id']}") from e

        logging.warning(
            f"EE end {point['id']}, took {time.time() - t0:.1f}s, bytes={len(raw)}"
        )

        out_dict = dict(point)
        out_dict['array'] = arr
        yield out_dict

class WriteTFExample(beam.PTransform):
    """Write example"""
    def __init__(self, output_dir, file_name_suffix='.tfrecord.gz'):
        self.output_dir = output_dir
        self.file_name_suffix = file_name_suffix

    def expand(self, pcoll):
        return (
            pcoll
            | 'Write to TFRecord' >> beam.io.WriteToTFRecord(
                self.output_dir,
                file_name_suffix=self.file_name_suffix
            )
        )
