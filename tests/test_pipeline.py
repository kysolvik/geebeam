import pytest
from unittest.mock import patch, MagicMock
from geebeam.pipeline import _prepare_run_metadata, _check_if_localrunner
from apache_beam.options.pipeline_options import PipelineOptions

@patch('ee.Initialize')
@patch('ee.Projection')
def test_prepare_run_metadata(mock_projection, mock_ee_init):
    config = {
        'project_id': 'test-project',
        'proj': 'EPSG:4326',
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
