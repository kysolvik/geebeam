import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from geebeam._transforms import _split_dataset

def test__split_dataset():
    element = {'metadata': {'split': 'train'}}
    assert _split_dataset(element, 2) == 0
    element = {'metadata': {'split': 'val'}}
    assert _split_dataset(element, 2) == 1
    element = {'metadata': {'split': 'test'}}
    assert _split_dataset(element, 2) == 2

# Testing EEComputePatch might be hard without a real EE session,
# but we can mock the internal methods.
@patch('ee.Initialize')
@patch('ee.data.computePixels')
@patch('ee.deserializer.fromJSON')
def test_ee_compute_patch(mock_from_json, mock_compute_pixels, mock_ee_init):
    from geebeam._transforms import EEComputePatch

    config = {
        'project_id': 'test-project',
        'patch_size': 2,
        'crs': 'EPSG:4326'
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

    point = {'id': 1, 'y': 10.0, 'x': 20.0}
    results = list(patch_fn.process(point))
    print(results)

    assert len(results) == 1
    result = results[0]
    assert 'metadata' in result
    assert result['metadata']['id'] == 1
    assert 'array' in result
    assert 'b1' in result['array']
    assert result['array']['b1'].shape == (1, 2, 2)
