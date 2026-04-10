import pytest
import pandas as pd
import geopandas as gpd
import numpy as np
from unittest.mock import MagicMock
from geebeam.sampler import sample_region_random, split_sets

def test_sample_region_random():
    mock_roi = MagicMock(spec=gpd.GeoDataFrame)
    mock_points = gpd.points_from_xy([10.0, 20.0], [50.0, 60.0])
    mock_roi.to_crs.return_value.sample_points.return_value.geometry.explode.return_value = mock_points
    
    result = sample_region_random(mock_roi, 'EPSG:4326', 2)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert 'geometry' in result.columns
    assert result.iloc[0].geometry.x == 10.0
    assert result.iloc[0].geometry.y == 50.0
    assert result.crs == 'EPSG:4326'

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
