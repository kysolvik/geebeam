import pytest
from unittest.mock import patch, MagicMock
from geebeam.runner import prepare_run_metadata

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
    
    scale_x, scale_y = prepare_run_metadata(config)
    
    assert scale_x == 30.0
    assert scale_y == 30.0 # -(-30.0) in code: scale_y = -proj_dict['transform'][4]
