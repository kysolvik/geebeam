import pytest
import numpy as np
import ee
from unittest.mock import patch, MagicMock
from geebeam.pipeline import (
    _prepare_run_metadata,
    _apply_position_offset,
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

PATCH_SIZE = 10
SCALE_X = 0.001
SCALE_Y = 0.001
# Top-left corner of a known patch
X0, Y0 = 10.0, 20.0

def _records_at_position(position):
    """Build a single record whose x/y is at the given position of the patch anchored at (X0, Y0)."""
    offsets = {
        'top-left':     (0,                       0),
        'top-right':    (PATCH_SIZE * SCALE_X,    0),
        'bottom-left':  (0,                       PATCH_SIZE * SCALE_Y),
        'bottom-right': (PATCH_SIZE * SCALE_X,    PATCH_SIZE * SCALE_Y),
        'center':       (PATCH_SIZE / 2 * SCALE_X, PATCH_SIZE / 2 * SCALE_Y),
    }
    dx, dy = offsets[position]
    return [{'id': 0, 'x': X0 + dx, 'y': Y0 + dy, 'split': 'full'}]

@pytest.mark.parametrize('position', ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'])
def test_apply_position_offset_topleft_roundtrip(position):
    """All five positions on the same patch should yield the same top-left corner."""
    records = _records_at_position(position)
    result = _apply_position_offset(records, position, PATCH_SIZE, SCALE_X, SCALE_Y)
    assert pytest.approx(result[0]['x_topleft']) == X0
    assert pytest.approx(result[0]['y_topleft']) == Y0

@pytest.mark.parametrize('position', ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'])
def test_apply_position_offset_preserves_original_xy(position):
    """Original x/y (the sampling location) must not be modified."""
    records = _records_at_position(position)
    original_x, original_y = records[0]['x'], records[0]['y']
    result = _apply_position_offset(records, position, PATCH_SIZE, SCALE_X, SCALE_Y)
    assert result[0]['x'] == original_x
    assert result[0]['y'] == original_y

def test_apply_position_offset_invalid_position():
    records = [{'id': 0, 'x': 1.0, 'y': 2.0}]
    with pytest.raises(ValueError, match='Invalid position'):
        _apply_position_offset(records, 'middle', PATCH_SIZE, SCALE_X, SCALE_Y)

@patch('ee.Initialize')
@patch('ee.Projection')
def test_run_pipeline_wraps_single_image(mock_projection, mock_ee_init):
    """A bare ee.Image passed as image_list should warn and be wrapped in a list."""
    mock_proj_obj = MagicMock()
    mock_proj_obj.getInfo.return_value = {'transform': [0.001, 0, 0, 0, -0.001, 0]}
    mock_projection.return_value.atScale.return_value = mock_proj_obj

    single_image = MagicMock(spec=ee.Image)
    with pytest.warns(UserWarning, match='Wrapping single image'):
        try:
            run_pipeline(
                image_list=single_image,
                output_path='/tmp/test',
                project='test-project',
                patch_size=4,
                scale=30.0,
                sampling_points=MagicMock(),
            )
        except Exception:
            pass  # pipeline will fail further on; we only care the warning fired

def test_run_pipeline_rejects_image_collection():
    """An ee.ImageCollection passed as image_list should raise ValueError."""
    with pytest.raises(ValueError, match='ee.ImageCollection'):
        run_pipeline(
            image_list=MagicMock(spec=ee.ImageCollection),
            output_path='/tmp/test',
            project='test-project',
            patch_size=4,
            scale=30.0,
            sampling_points=MagicMock(),
        )

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
