import tensorflow as tf
import numpy as np
from geebeam.tf_utils import _bytes_feature, _float_feature, _int64_feature, dict_to_example

def test_features():
    assert isinstance(_bytes_feature(b'test'), tf.train.Feature)
    assert isinstance(_float_feature(0.5), tf.train.Feature)
    assert isinstance(_int64_feature(1), tf.train.Feature)

def test_dict_to_example():
    element = {
        'metadata': {
            'id': 1,
            'y': 10.0,
            'x': 20.0,
            'extra': 0.5
        },
        'array': {
            'band1': np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        }
    }

    example_str = dict_to_example(element)
    assert isinstance(example_str, bytes)

    example = tf.train.Example()
    example.ParseFromString(example_str)

    features = example.features.feature
    assert features['md_id'].int64_list.value[0] == 1
    assert features['md_y'].float_list.value[0] == 10.0
    assert features['md_x'].float_list.value[0] == 20.0
    assert features['md_extra'].float_list.value[0] == 0.5
    assert list(features['im_band1'].float_list.value) == [1.0, 2.0, 3.0, 4.0]