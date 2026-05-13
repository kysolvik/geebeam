import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from geebeam._pipeline import (
    _prepare_run_metadata,
    _check_if_localrunner,
    _type_inference,
    _build_md_feature_dict,
    run_pipeline,
)
from apache_beam.options.pipeline_options import PipelineOptions

@patch('ee.Initialize')
@patch('ee.Projection')
def test_prepare_run_metadata(mock_projection, mock_ee_init):
    config = {
        'project_id': 'test-project',
        'crs': 'EPSG:4326',
        'scale': 30
    }

    mock_proj_obj = MagicMock()
    mock_proj_obj.getInfo.return_value = {
        'transform': [30.0, 0, 100, 0, -30.0, 200]
    }
    mock_projection.return_value.atScale.return_value = mock_proj_obj

    scale_x, scale_y = _prepare_run_metadata(config)

    assert scale_x == 30.0
    assert scale_y == 30.0 # -(-30.0) in code: scale_y = -proj_dict['transform'][4]

def test_local_runner_check():
    assert _check_if_localrunner(PipelineOptions())
    assert _check_if_localrunner(PipelineOptions(runner='PrismRunner'))
    assert _check_if_localrunner(PipelineOptions(runner='DirectRunner'))
    assert not _check_if_localrunner(PipelineOptions(runner='DataflowRunner'))

def test_type_inference_int():
    assert _type_inference(42) == 'int'

def test_type_inference_float():
    assert _type_inference(3.14) == 'float'

def test_type_inference_list():
    result = _type_inference([1, 2, 3])
    assert isinstance(result, dict)
    assert 'arraylike' in result
    assert result['arraylike'] == (3,)

def test_type_inference_ndarray():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _type_inference(arr)
    assert isinstance(result, dict)
    assert 'arraylike' in result
    assert result['arraylike'] == (2, 2)

def test_type_inference_str():
    assert _type_inference('hello') == 'str'

def test_type_inference_invalid():
    with pytest.raises(ValueError):
        _type_inference({'key': 'value'})

def test_build_md_feature_dict_basic():
    record = {'id': 1, 'x': 10.0, 'y': 20.0, 'split': 'train'}
    result = _build_md_feature_dict(record, None)
    assert result == {'id': 'int', 'x': 'float', 'y': 'float', 'split': 'str'}

def test_build_md_feature_dict_with_extra_metadata():
    record = {'id': 1, 'x': 10.0, 'y': 20.0, 'split': 'train'}
    extra_metadata = {'year': 2020, 'weight': 0.5}
    result = _build_md_feature_dict(record, extra_metadata)
    assert result['year'] == 'int'
    assert result['weight'] == 'float'
    assert result['id'] == 'int'

def test_build_md_feature_dict_invalid_type():
    record = {'id': 1, 'x': 10.0, 'y': 20.0, 'split': 'train'}
    extra_metadata = {'bad': {'nested': 'dict'}}
    with pytest.raises(ValueError):
        _build_md_feature_dict(record, extra_metadata)

def test_run_pipeline_invalid_output_type():
    with pytest.raises(ValueError):
        run_pipeline(
            image_list=[MagicMock()],
            output_path='/tmp/test',
            project='test-project',
            patch_size=4,
            scale=30.0,
            sampling_points=MagicMock(),
            output_type='invalid_type',
        )
