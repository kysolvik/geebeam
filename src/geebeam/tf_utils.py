"""Helpers for tensorflow"""

import tensorflow as tf
import apache_beam as beam
import numpy as np

from geebeam import transforms

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def dict_to_example(record):
    """"Convert structured numpy array to tf.Example proto."""
    # First add metadata
    # Add metadata features
    md_dict = {}
    for md_key, md_val in record['metadata'].items():
        if isinstance(md_val, str):
            md_dict[f'md_{md_key}'] = _bytes_feature(md_val.encode('utf-8'))
        elif np.isscalar(md_val):
            if isinstance(md_val, int):
                md_dict[f'md_{md_key}'] = _int64_feature(record['metadata'][md_key])
            else:
                md_dict[f'md_{md_key}'] = _float_feature(record['metadata'][md_key])
        else:
            md_dict[f'md_{md_key}'] = tf.train.Feature(float_list=
                tf.train.FloatList(
                    value = transforms.convert_to_iterable(record['metadata'][md_key])
                )
            )

    # Build image feature with named bands
    array_dict = {}
    for band_name, band_data in record['array'].items():
        array_dict[f'{band_name}'] = band_data.flatten().astype('float32')

    # Build image feature with named bands
    array_dict = {}
    for im_feat, im_data in record['array'].items():
        array_dict['im_'+im_feat] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = im_data.flatten()
                )
        )

    # Combine
    feature = {**md_dict, **array_dict}

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