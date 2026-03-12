import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import MagicMock, patch
from geebeam.transforms import _bytes_feature, _float_feature, _int64_feature, dict_to_example, split_dataset

def test_features():
    assert isinstance(_bytes_feature(b'test'), tf.train.Feature)
    assert isinstance(_float_feature(0.5), tf.train.Feature)
    assert isinstance(_int64_feature(1), tf.train.Feature)

def test_dict_to_example():
    element = {
        'metadata': {
            'id': 1,
            'lat': 10.0,
            'lon': 20.0,
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
    assert features['md_lat'].float_list.value[0] == 10.0
    assert features['md_lon'].float_list.value[0] == 20.0
    assert features['md_extra'].float_list.value[0] == 0.5
    assert list(features['im_band1'].float_list.value) == [1.0, 2.0, 3.0, 4.0]

def test_split_dataset():
    element = {'metadata': {'split': 'train'}}
    assert split_dataset(element, 2) == 0
    element = {'metadata': {'split': 'val'}}
    assert split_dataset(element, 2) == 1
    element = {'metadata': {'split': 'test'}}
    assert split_dataset(element, 2) == 2

# Testing EEComputePatch might be hard without a real EE session, 
# but we can mock the internal methods.
@patch('ee.Initialize')
@patch('ee.data.computePixels')
@patch('ee.deserializer.fromJSON')
def test_ee_compute_patch(mock_from_json, mock_compute_pixels, mock_ee_init):
    from geebeam.transforms import EEComputePatch
    
    config = {
        'project_id': 'test-project',
        'patch_size': 2,
    }
    serialized_image = '{"json": "image"}'
    scale_x = 30.0
    scale_y = -30.0
    band_groups = [['b1']]
    
    # Mocking computePixels response
    # It returns raw bytes of NPY
    import io
    arr = np.array([np.ones((2,2))], dtype=[('b1', 'f4')])
    buf = io.BytesIO()
    np.save(buf, arr)
    mock_compute_pixels.return_value = buf.getvalue()
    
    mock_image = MagicMock()
    mock_from_json.return_value = mock_image
    mock_image.select.return_value = mock_image
    
    patch_fn = EEComputePatch(config, serialized_image, scale_x, scale_y, band_groups)
    patch_fn.setup() # Initialize EE
    
    point = {'id': 1, 'lat': 10.0, 'lon': 20.0}
    results = list(patch_fn.process(point))
    print(results)
    
    assert len(results) == 1
    result = results[0]
    assert 'metadata' in result
    assert result['metadata']['id'] == 1
    assert 'array' in result
    assert 'b1' in result['array']
    assert result['array']['b1'].shape == (1, 2, 2)
