import io
import pytest
import numpy as np
import ee
from unittest.mock import patch, MagicMock
from geebeam._ee_utils import (
    get_band_names,
    build_prepped_image,
    _dedupe_band_names,
    list_to_im,
    get_pixels,
    get_pixels_allbands,
)

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
    mock_image2 = MagicMock()

    mock_ee_list.return_value.getInfo.return_value = [['b1'], ['b2']]

    mock_prepped_im = MagicMock()
    mock_ee_ic.return_value.toBands.return_value.rename.return_value = mock_prepped_im

    result_im, band_groups, all_bands = build_prepped_image([mock_image1, mock_image2], split_processing=False)
    assert result_im == mock_prepped_im
    assert band_groups == [['b1', 'b2']]
    assert all_bands == ['b1', 'b2']

    result_im, band_groups, all_bands = build_prepped_image([mock_image1, mock_image2], split_processing=True)
    assert band_groups == [['b1'], ['b2']]
    assert all_bands == ['b1', 'b2']

@patch('ee.ImageCollection')
@patch('ee.List')
def test_build_prepped_image_deduplicates(mock_ee_list, mock_ee_ic):
    mock_image1 = MagicMock()
    mock_image2 = MagicMock()

    mock_ee_list.return_value.getInfo.return_value = [['b1', 'b2'], ['b1', 'b3']]

    mock_prepped_im = MagicMock()
    mock_ee_ic.return_value.toBands.return_value.rename.return_value = mock_prepped_im

    with pytest.warns(UserWarning, match='Duplicate band names'):
        result_im, band_groups, all_bands = build_prepped_image(
            [mock_image1, mock_image2], split_processing=False
        )
    assert all_bands == ['b1_0', 'b2', 'b1_1', 'b3']
    assert band_groups == [['b1_0', 'b2', 'b1_1', 'b3']]

    with pytest.warns(UserWarning, match='Duplicate band names'):
        result_im, band_groups, all_bands = build_prepped_image(
            [mock_image1, mock_image2], split_processing=True
        )
    assert all_bands == ['b1_0', 'b2', 'b1_1', 'b3']
    assert band_groups == [['b1_0', 'b2'], ['b1_1', 'b3']]

def test_dedupe_band_names_no_duplicates():
    result, changed = _dedupe_band_names(['a', 'b', 'c'])
    assert result == ['a', 'b', 'c']
    assert changed is False

def test_dedupe_band_names_with_duplicates():
    result, changed = _dedupe_band_names(['a', 'b', 'a'])
    assert result == ['a_0', 'b', 'a_1']
    assert changed is True

def test_dedupe_band_names_multiple_duplicates():
    result, changed = _dedupe_band_names(['x', 'x', 'x'])
    assert result == ['x_0', 'x_1', 'x_2']
    assert changed is True

@patch('geebeam._ee_utils.build_prepped_image')
def test_list_to_im(mock_build_prepped_image):
    mock_prepped_im = MagicMock()
    mock_build_prepped_image.return_value = (mock_prepped_im, None, None)

    result = list_to_im([MagicMock()])
    assert result == mock_prepped_im

def _make_npy_bytes(patch_size=4, band_name='band1'):
    """Helper to create valid NPY bytes for a structured array."""
    dtype = [(band_name, 'f4')]
    arr = np.zeros((patch_size, patch_size), dtype=dtype)
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()

@patch('ee.data.computePixels')
def test_get_pixels_uses_topleft_coords(mock_compute_pixels):
    """get_pixels should use x_topleft/y_topleft for the GEE affine transform when present."""
    patch_size = 4
    mock_compute_pixels.return_value = _make_npy_bytes(patch_size=patch_size, band_name='band1')

    point = {'id': 0, 'x': 10.0, 'y': 20.0, 'x_topleft': 5.0, 'y_topleft': 15.0}
    get_pixels(MagicMock(), point, patch_size, scale_x=30.0, scale_y=-30.0, crs_code='EPSG:4326')

    request = mock_compute_pixels.call_args[0][0]
    assert request['grid']['affineTransform']['translateX'] == 5.0
    assert request['grid']['affineTransform']['translateY'] == 15.0

@patch('ee.data.computePixels')
def test_get_pixels_falls_back_to_xy(mock_compute_pixels):
    """get_pixels falls back to x/y when x_topleft/y_topleft are absent."""
    patch_size = 4
    mock_compute_pixels.return_value = _make_npy_bytes(patch_size=patch_size, band_name='band1')

    point = {'id': 0, 'x': 10.0, 'y': 20.0}
    get_pixels(MagicMock(), point, patch_size, scale_x=30.0, scale_y=-30.0, crs_code='EPSG:4326')

    request = mock_compute_pixels.call_args[0][0]
    assert request['grid']['affineTransform']['translateX'] == 10.0
    assert request['grid']['affineTransform']['translateY'] == 20.0

@patch('ee.data.computePixels')
def test_get_pixels_empty_response(mock_compute_pixels):
    mock_compute_pixels.return_value = b''

    point = {'id': 0, 'x': 10.0, 'y': 20.0}
    with pytest.raises(RuntimeError, match='Empty EE response'):
        get_pixels(MagicMock(), point, 4, scale_x=30.0, scale_y=-30.0, crs_code='EPSG:4326')

@patch('ee.data.computePixels')
def test_get_pixels_allbands_two_groups(mock_compute_pixels):
    patch_size = 4
    raw_bytes = _make_npy_bytes(patch_size=patch_size, band_name='band1')
    mock_compute_pixels.return_value = raw_bytes

    mock_im = MagicMock()
    mock_im.select.return_value = mock_im

    point = {'id': 0, 'x': 10.0, 'y': 20.0, 'x_topleft': 10.0, 'y_topleft': 20.0}
    band_groups = [['band1'], ['band1']]
    results = get_pixels_allbands(mock_im, band_groups, point, patch_size,
                                  scale_x=30.0, scale_y=-30.0, crs_code='EPSG:4326')

    assert len(results) == 2
    for r in results:
        assert r.shape == (patch_size, patch_size)
