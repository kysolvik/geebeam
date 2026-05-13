import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from geebeam._transforms import (
    _split_dataset,
    _convert_to_iterable,
    _write_sidecar_schema,
    join_structured_arrays,
    _join_struct_arrays_to_dict,
    AddMetadata,
)

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

def test_convert_to_iterable_list():
    val = [1, 2, 3]
    result = _convert_to_iterable(val)
    assert result is val

def test_convert_to_iterable_non_iterable():
    result = _convert_to_iterable(42)
    assert result == [42]

def test_write_sidecar_schema_local(tmp_path):
    output_path = str(tmp_path)
    band_names = ['band1', 'band2']
    extra_metadata_keys = ['year']
    _write_sidecar_schema(output_path, band_names, extra_metadata_keys, is_gcs=False)
    schema_file = tmp_path / 'schema.json'
    assert schema_file.exists()
    with open(schema_file) as f:
        schema = json.load(f)
    assert 'features' in schema
    assert 'im_band1' in schema['features']
    assert 'im_band2' in schema['features']
    assert 'md_year' in schema['features']

def test_join_structured_arrays():
    a = np.zeros((2, 2), dtype=[('x', 'f4')])
    b = np.ones((2, 2), dtype=[('y', 'f4')])
    result = join_structured_arrays([a, b])
    assert 'x' in result.dtype.names
    assert 'y' in result.dtype.names
    assert result.shape == (2, 2)

def test_join_struct_arrays_to_dict():
    a = np.zeros((2, 2), dtype=[('band1', 'f4')])
    b = np.ones((2, 2), dtype=[('band2', 'f4')])
    result = _join_struct_arrays_to_dict([a, b])
    assert isinstance(result, dict)
    assert 'band1' in result
    assert 'band2' in result
    assert np.all(result['band1'] == 0)
    assert np.all(result['band2'] == 1)

def test_add_metadata_process():
    extra = {'source': 'test_source'}
    dofn = AddMetadata(extra)
    element = {
        'array': {'band1': np.ones((4, 4))},
        'metadata': {'id': 1, 'x': 10.0, 'y': 20.0}
    }
    results = list(dofn.process(element))
    assert len(results) == 1
    result = results[0]
    assert result['metadata']['source'] == 'test_source'
    assert result['metadata']['id'] == 1
    assert 'array' in result