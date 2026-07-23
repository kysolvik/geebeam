import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from rasterio.transform import Affine
from unittest.mock import MagicMock, patch
from geebeam.sampler import (
    sample_region_random,
    split_sets,
    _process_sampling_points,
    _get_roi,
    sample_region_grid,
    _parse_transform,
    _snap_to_grid,
)

def test_split_sets():
    df = pd.DataFrame({
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    result = split_sets(df, ['train','validation'], [0.8, 0.2], 42, shuffle=False)
    assert (result.loc[0:7,'split'] == 'train').all()
    assert (result.loc[8:,'split'] == 'validation').all()
    
    assert result['split'].value_counts()['train'] == 8
    assert result['split'].value_counts()['validation'] == 2
    
    result_shuffled = split_sets(df, ['train','validation'], [0.8, 0.2], 42, shuffle=True)
    assert 'id' in result_shuffled.columns
    assert result_shuffled['split'].value_counts()['train'] == 8
    assert result_shuffled['split'].value_counts()['validation'] == 2

def test_process_sampling_points_geodataframe():
    gdf = gpd.GeoDataFrame(
        {'id': [0, 1], 'split': ['train', 'validation']},
        geometry=gpd.points_from_xy([10.0, 20.0], [50.0, 60.0]),
        crs='EPSG:4326'
    )
    records, splits = _process_sampling_points(gdf, 'EPSG:4326')
    assert isinstance(records, list)
    assert len(records) == 2
    assert 'x' in records[0]
    assert 'y' in records[0]
    assert set(splits) == {'train', 'validation'}

def test_process_sampling_points_dataframe():
    df = pd.DataFrame({'x': [10.0, 20.0], 'y': [50.0, 60.0]})
    records, splits = _process_sampling_points(df, 'EPSG:4326')
    assert len(records) == 2
    assert records[0]['x'] == 10.0
    assert records[0]['y'] == 50.0
    assert 'id' in records[0]
    assert 'split' in records[0]

def test_process_sampling_points_crs_mismatch():
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([10.0], [50.0]),
        crs='EPSG:4326'
    )
    with pytest.raises(ValueError, match='projection does not match'):
        _process_sampling_points(gdf, 'EPSG:3857')

def test_process_sampling_points_dataframe_missing_xy():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    with pytest.raises(ValueError, match='x.*y'):
        _process_sampling_points(df, 'EPSG:4326')

def test_process_sampling_points_invalid_type():
    with pytest.raises(ValueError):
        _process_sampling_points({'x': 10, 'y': 20}, 'EPSG:4326')

def test_process_sampling_points_adds_id_and_split():
    df = pd.DataFrame({'x': [10.0], 'y': [50.0]})
    records, splits = _process_sampling_points(df, 'EPSG:4326')
    assert 'id' in records[0]
    assert records[0]['split'] == 'full'
    assert list(splits) == ['full']

def test_get_roi_geodataframe():
    gdf = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)],
        crs='EPSG:4326'
    )
    result = _get_roi(gdf, 'EPSG:4326')
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs.to_epsg() == 4326

def test_get_roi_shapely():
    geom = box(0, 0, 1, 1)
    result = _get_roi(geom, 'EPSG:4326')
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.crs.to_epsg() == 4326

def test_get_roi_string():
    with patch('geopandas.read_file') as mock_read_file:
        mock_gdf = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)],
            crs='EPSG:4326'
        )
        mock_read_file.return_value = mock_gdf
        result = _get_roi('/fake/path/file.gpkg', 'EPSG:4326')
        assert isinstance(result, gpd.GeoDataFrame)
        mock_read_file.assert_called_once_with('/fake/path/file.gpkg')

def test_get_roi_invalid():
    with pytest.raises(ValueError):
        _get_roi(12345, 'EPSG:4326')

def test_sample_region_grid():
    roi_gdf = gpd.GeoDataFrame(
        geometry=[box(0.0, 0.0, 1.0, 1.0)],
        crs='EPSG:4326'
    )
    with patch('geebeam.sampler._get_crs_scale', return_value=0.1):
        result = sample_region_grid(roi=roi_gdf, crs='EPSG:4326', stride=1, scale=1000.0)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) > 0

def test_sample_region_random():
    mock_roi = MagicMock(spec=gpd.GeoDataFrame)
    mock_points = gpd.points_from_xy([10.0, 20.0], [50.0, 60.0], crs='EPSG:4326')
    mock_roi.sample_points.return_value.geometry.explode.return_value = mock_points
    mock_roi.crs.to_string.return_value = 'EPSG:4326'
    
    result = sample_region_random(mock_roi, 'EPSG:4326', 2)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert 'geometry' in result.columns
    assert result.iloc[0].geometry.x == 10.0
    assert result.iloc[0].geometry.y == 50.0
    assert result.crs == 'EPSG:4326'

def test_split_sets_three_splits():
    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    result = split_sets(df, ['train', 'val', 'test'], split_ratios=[0.6, 0.2, 0.2], shuffle=False)
    counts = result['split'].value_counts()
    assert counts['train'] == 6
    assert counts['val'] == 2
    assert counts['test'] == 2

def test_split_sets_ratios_not_summing_to_one():
    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    with pytest.raises(ValueError, match='ratios do not equal 1'):
        split_sets(df, ['train', 'val'], split_ratios=[0.5, 0.3])

def test_split_sets_no_ratios_or_counts():
    df = pd.DataFrame({'x': range(10), 'y': range(10)})
    with pytest.raises(ValueError, match='split_ratios.*split_counts'):
        split_sets(df, ['train', 'val'])

def test_split_sets_empty_split_names():
    df = pd.DataFrame({'x': range(5), 'y': range(5)})
    result = split_sets(df, split_names=[])
    assert (result['split'] == 'full').all()

def test_parse_transform_affine_and_tuple_match():
    affine = Affine(30.0, 0, 500000.0, 0, -30.0, 4500000.0)
    tup = (30.0, 0, 500000.0, 0, -30.0, 4500000.0)
    assert _parse_transform(affine) == _parse_transform(tup)
    x0, y0, sx, sy = _parse_transform(affine)
    assert (x0, y0) == (500000.0, 4500000.0)
    # scale_x/scale_y are positive magnitudes (matches scale_y = -transform[4] convention)
    assert (sx, sy) == (30.0, 30.0)

def test_parse_transform_rejects_shear():
    with pytest.raises(ValueError, match='shear'):
        _parse_transform((30.0, 0.1, 0.0, 0.0, -30.0, 0.0))
    with pytest.raises(ValueError, match='shear'):
        _parse_transform((30.0, 0.0, 0.0, 0.2, -30.0, 0.0))

def test_snap_to_grid_scalar():
    # origin (0, 0), pixel size 5 -> nearest multiple of 5
    x, y = _snap_to_grid(12.3, 63.9, 0.0, 0.0, 5.0, 5.0)
    assert x == 10.0
    assert y == 65.0
    assert isinstance(x, float) and isinstance(y, float)

def test_snap_to_grid_array_and_offset_origin():
    xs = np.array([12.3, 27.6])
    ys = np.array([51.1, 63.9])
    xs_snap, ys_snap = _snap_to_grid(xs, ys, 1.0, 1.0, 5.0, 5.0)
    # nodes must satisfy (v - origin) / pixel == integer
    assert np.allclose((xs_snap - 1.0) / 5.0, np.round((xs_snap - 1.0) / 5.0))
    assert np.allclose((ys_snap - 1.0) / 5.0, np.round((ys_snap - 1.0) / 5.0))

def test_sample_region_grid_transform():
    roi_gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)], crs='EPSG:4326')
    # origin (0.05, 0.05), pixel size 0.1; align mode should NOT call _get_crs_scale
    align = Affine(0.1, 0, 0.05, 0, -0.1, 0.05)
    result = sample_region_grid(roi=roi_gdf, crs='EPSG:4326', stride=1, scale=1000.0,
                                transform=align)
    assert len(result) > 0
    kx = (result.geometry.x.values - 0.05) / 0.1
    ky = (result.geometry.y.values - 0.05) / 0.1
    assert np.allclose(kx, np.round(kx))
    assert np.allclose(ky, np.round(ky))

def test_sample_region_grid_align_no_scale():
    """transform supplies pixel size, so scale can be omitted."""
    roi_gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)], crs='EPSG:4326')
    align = Affine(0.1, 0, 0.05, 0, -0.1, 0.05)
    result = sample_region_grid(roi=roi_gdf, crs='EPSG:4326', stride=1, transform=align)
    assert len(result) > 0
    kx = (result.geometry.x.values - 0.05) / 0.1
    assert np.allclose(kx, np.round(kx))

def test_sample_region_grid_requires_scale():
    """Without transform, scale is required."""
    roi_gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, 1.0, 1.0)], crs='EPSG:4326')
    with pytest.raises(ValueError, match='scale'):
        sample_region_grid(roi=roi_gdf, crs='EPSG:4326', stride=1)

def test_sample_region_random_transform_snaps_points():
    mock_roi = MagicMock(spec=gpd.GeoDataFrame)
    mock_points = gpd.points_from_xy([12.3, 27.6], [51.1, 63.9], crs='EPSG:4326')
    mock_roi.sample_points.return_value.geometry.explode.return_value = mock_points
    mock_roi.crs.to_string.return_value = 'EPSG:4326'

    align = Affine(5.0, 0, 0.0, 0, -5.0, 0.0)
    result = sample_region_random(mock_roi, 'EPSG:4326', 2, transform=align)

    assert list(result.geometry.x) == [10.0, 30.0]
    assert list(result.geometry.y) == [50.0, 65.0]

def test_sample_region_random_transform_warns_on_collision():
    mock_roi = MagicMock(spec=gpd.GeoDataFrame)
    # Two nearby points snap to the same coarse grid node -> collision
    mock_points = gpd.points_from_xy([12.1, 12.4], [50.2, 49.8], crs='EPSG:4326')
    mock_roi.sample_points.return_value.geometry.explode.return_value = mock_points
    mock_roi.crs.to_string.return_value = 'EPSG:4326'

    align = Affine(5.0, 0, 0.0, 0, -5.0, 0.0)
    with pytest.warns(UserWarning, match='duplicate locations'):
        result = sample_region_random(mock_roi, 'EPSG:4326', 2, transform=align)
    # both collapsed onto (10, 50)
    assert list(result.geometry.x) == [10.0, 10.0]
    assert list(result.geometry.y) == [50.0, 50.0]
