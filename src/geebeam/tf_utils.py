"""Helpers for tensorflow"""

import tensorflow as tf

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

def dict_to_example(element):
    """"Convert structured numpy array to tf.Example proto."""
    # First add metadata
    md_dict = {
        'md_id': _int64_feature(element['metadata']['id']),
        'md_y': _float_feature(element['metadata']['y']),
        'md_x': _float_feature(element['metadata']['x']),
        }
    for md_key in element['metadata'].keys():
        if md_key not in ['id','y','x', 'split']:
            md_dict['md_' + md_key] = tf.train.Feature(float_list=
                tf.train.FloatList(
                    value = transforms.convert_to_iterable(element['metadata'][md_key])
                )
            )

    # Build image feature with named bands
    array_dict = {}
    for im_feat in element['array'].keys():#.dtype.names:
        array_dict['im_'+im_feat] = tf.train.Feature(
            float_list = tf.train.FloatList(
                value = element['array'][im_feat].flatten()))

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