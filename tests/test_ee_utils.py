import pytest
import ee
from unittest.mock import patch, MagicMock
from geebeam._ee_utils import get_band_names, build_prepped_image, list_to_im

def test_get_band_names():
    mock_image1 = MagicMock()
    mock_image1.bandNames.return_value = ['b1', 'b2']
    mock_image2 = MagicMock()
    mock_image2.bandNames.return_value = ['b3']
    
    result = get_band_names([mock_image1, mock_image2])
    assert result == [['b1', 'b2'], ['b3']]

@patch('ee.ImageCollection')
@patch('ee.List')
def test_build_prepped_image(mock_ee_list, mock_ee_ic):
    mock_image1 = MagicMock()
    mock_image1.bandNames.return_value = ee.List(['b1'])
    mock_image2 = MagicMock()
    mock_image2.bandNames.return_value = ee.List(['b2'])
    
    mock_ee_list.return_value.flatten.return_value = ee.List(['b1', 'b2'])
    
    # Mocking ee.ImageCollection(input_list).toBands().rename(band_names_flat)
    mock_prepped_im = MagicMock()
    mock_ee_ic.return_value.toBands.return_value.rename.return_value = mock_prepped_im
    
    result_im, band_groups, all_bands = build_prepped_image([mock_image1, mock_image2], split_processing=False)
    
    print(all_bands)
    assert result_im == mock_prepped_im
    assert band_groups == [ee.List(['b1', 'b2'])]
    assert all_bands == ee.List(['b1', 'b2']).getInfo()
    
    result_im, band_groups, all_bands = build_prepped_image([mock_image1, mock_image2], split_processing=True)
    assert band_groups == [ee.List(['b1']), ee.List(['b2'])]
    assert all_bands == ee.List(['b1', 'b2']).getInfo()

@patch('geebeam._ee_utils.build_prepped_image')
def test_list_to_im(mock_build_prepped_image):
    mock_prepped_im = MagicMock()
    mock_build_prepped_image.return_value = (mock_prepped_im, None)
    
    result = list_to_im([MagicMock()])
    assert result == mock_prepped_im
