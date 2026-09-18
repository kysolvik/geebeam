from unittest.mock import MagicMock, patch

import ee
import numpy as np
import pytest
from apache_beam.options.pipeline_options import PipelineOptions
from rasterio.transform import Affine

from geebeam.pipeline import (
    _apply_position_offset,
    _build_md_feature_dict,
    _check_if_localrunner,
    _prepare_run_metadata,
    _type_inference,
    run_pipeline,
)


@patch('ee.Initialize')
@patch('ee.Projection')
def test_prepare_run_metadata(mock_projection, mock_ee_init):
    config = {
        'project_id': 'test-project',
        'crs': 'EPSG:4326',
        'scale': 30
    }

    mock_proj_obj = MagicMock()
    # EE reports a positive e -- verified against the live API:
    # ee.Projection('EPSG:4326').atScale(30).getInfo()['transform']
    #   -> [0.00026949458523585647, 0, 0, 0, 0.00026949458523585647, 0]
    mock_proj_obj.getInfo.return_value = {
        'transform': [30.0, 0, 100, 0, 30.0, 200]
    }
    mock_projection.return_value.atScale.return_value = mock_proj_obj

    scale_x, scale_y = _prepare_run_metadata(config)

    assert scale_x == 30.0
    # scale_y = -proj_dict['transform'][4] -> negative, i.e. north-up
    assert scale_y == -30.0

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
    with pytest.raises(TypeError):
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
    with pytest.raises(TypeError):
        _build_md_feature_dict(record, extra_metadata)

PATCH_SIZE = 10
SCALE_X = 0.001
SCALE_Y = -0.001  # north-up: y decreases down the patch
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

@pytest.mark.parametrize('position', ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'])
def test_apply_position_offset_topleft_is_north_up(position):
    """The top-left corner is never right of, nor below, the sampling point.

    This is the invariant the positive-yres bug violated: with a positive scale_y the
    patch grew upward from y_topleft, producing south-up output.
    """
    records = _records_at_position(position)
    x, y = records[0]['x'], records[0]['y']
    result = _apply_position_offset(records, position, PATCH_SIZE, SCALE_X, SCALE_Y)
    assert result[0]['x_topleft'] <= x
    assert result[0]['y_topleft'] >= y


def test_apply_position_offset_invalid_position():
    records = [{'id': 0, 'x': 1.0, 'y': 2.0}]
    with pytest.raises(ValueError, match='Invalid position'):
        _apply_position_offset(records, 'middle', PATCH_SIZE, SCALE_X, SCALE_Y)

@pytest.mark.parametrize('position', ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'])
@pytest.mark.parametrize('patch_size', [4, 5])  # even and odd
def test_apply_position_offset_align_snaps_topleft(position, patch_size):
    """With align_transform, the top-left corner must land exactly on the reference grid,
    for any position and any (even/odd) patch_size."""
    # origin (0, 0), pixel size 1.0 -> grid nodes are integers
    align = Affine(1.0, 0, 0.0, 0, -1.0, 0.0)
    records = [{'id': 0, 'x': 3.37, 'y': 8.62, 'split': 'full'}]
    result = _apply_position_offset(records, position, patch_size, 1.0, -1.0,
                                    align_transform=align)
    x_tl, y_tl = result[0]['x_topleft'], result[0]['y_topleft']
    assert x_tl == round(x_tl)
    assert y_tl == round(y_tl)
    # original x/y preserved
    assert result[0]['x'] == 3.37
    assert result[0]['y'] == 8.62

@patch('ee.Initialize')
@patch('ee.Projection')
def test_prepare_run_metadata_align_overrides_scale(mock_projection, mock_ee_init):
    """align_transform pixel size overrides scale; ee.Projection should not be consulted."""
    config = {'project_id': 'test-project', 'crs': 'EPSG:4326', 'scale': 30}
    align = Affine(0.25, 0, 100.0, 0, -0.5, 200.0)

    scale_x, scale_y = _prepare_run_metadata(config, align_transform=align)

    assert scale_x == 0.25
    assert scale_y == -0.5  # returned signed, not as a magnitude
    mock_projection.assert_not_called()

@patch('ee.Initialize')
@patch('ee.Projection')
def test_prepare_run_metadata_paths_agree_in_sign(mock_projection, mock_ee_init):
    """The `scale=` and `align_transform=` branches must return the same sign convention.

    They disagreed before the fix (align returned a positive scale_y), which is what made
    align_transform runs come out south-up.
    """
    config = {'project_id': 'test-project', 'crs': 'EPSG:4326', 'scale': 30}
    mock_proj_obj = MagicMock()
    mock_proj_obj.getInfo.return_value = {'transform': [30.0, 0, 0, 0, 30.0, 0]}
    mock_projection.return_value.atScale.return_value = mock_proj_obj

    scale_sx, scale_sy = _prepare_run_metadata(config)
    align_sx, align_sy = _prepare_run_metadata(
        config, align_transform=Affine(30.0, 0, 100.0, 0, -30.0, 200.0))

    assert scale_sx > 0 and align_sx > 0
    assert scale_sy < 0 and align_sy < 0
    assert (scale_sx, scale_sy) == (align_sx, align_sy)


@patch('ee.Initialize')
@patch('ee.Projection')
def test_run_pipeline_wraps_single_image(mock_projection, mock_ee_init):
    """A bare ee.Image passed as image_list should warn and be wrapped in a list."""
    mock_proj_obj = MagicMock()
    mock_proj_obj.getInfo.return_value = {'transform': [0.001, 0, 0, 0, -0.001, 0]}
    mock_projection.return_value.atScale.return_value = mock_proj_obj

    single_image = MagicMock(spec=ee.Image)
    with (pytest.warns(UserWarning, match='Wrapping provided single ee.Image'),
          pytest.raises(TypeError)):
            run_pipeline(
                image_list=single_image,
                output_path='/tmp/test',
                project='test-project',
                patch_size=4,
                scale=30.0,
                sampling_points=MagicMock(),
            )

def test_run_pipeline_transform_warns_scale_ignored():
    """Passing align_transform should warn that `scale` is ignored."""
    with (pytest.warns(UserWarning, match='`scale` argument is ignored'),
          pytest.raises(TypeError)):
                run_pipeline(
                    image_list=[MagicMock()],
                    output_path='/tmp/test',
                    project='test-project',
                    patch_size=4,
                    scale=30.0,
                    sampling_points=MagicMock(),
                    align_transform=Affine(0.001, 0, 0, 0, -0.001, 0),
                )

def test_run_pipeline_requires_scale_or_align():
    """Neither scale nor align_transform provided should raise."""
    with pytest.raises(ValueError, match='scale.*align_transform'):
        run_pipeline(
            image_list=[MagicMock()],
            output_path='/tmp/test',
            project='test-project',
            patch_size=4,
            sampling_points=MagicMock(),
        )

def test_run_pipeline_invalid_output_dtype():
    """An unrecognized output_dtype should fail fast via np.dtype."""
    with pytest.raises(TypeError):
        run_pipeline(
            image_list=[MagicMock()],
            output_path='/tmp/test',
            project='test-project',
            patch_size=4,
            scale=30.0,
            sampling_points=MagicMock(),
            output_dtype='not-a-real-dtype',
        )

def test_run_pipeline_rejects_image_collection():
    """An ee.ImageCollection passed as image_list should raise TypeError."""
    with pytest.raises(TypeError, match='ee.ImageCollection'):
        run_pipeline(
            image_list=MagicMock(spec=ee.ImageCollection),
            output_path='/tmp/test',
            project='test-project',
            patch_size=4,
            scale=30.0,
            sampling_points=MagicMock(),
        )

def test_run_pipeline_invalid_output_type():
    with pytest.raises(TypeError):
        run_pipeline(
            image_list=[MagicMock()],
            output_path='/tmp/test',
            project='test-project',
            patch_size=4,
            scale=30.0,
            sampling_points=MagicMock(),
            output_type='invalid_type',
        )
