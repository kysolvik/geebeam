import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from geebeam.sampler import sample_random_points, split_train_validation

def test_sample_random_points():
    mock_roi = MagicMock()
    # mock_roi.sample_points(n_sample, rng=rng).geometry.explode().get_coordinates()
    mock_points = MagicMock()
    mock_coords = pd.DataFrame({
        'x': [10.0, 20.0],
        'y': [50.0, 60.0]
    })
    mock_roi.sample_points.return_value.geometry.explode.return_value.get_coordinates.return_value = mock_coords
    
    rng = np.random.default_rng(42)
    result = sample_random_points(mock_roi, 2, rng)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 2)
    assert 'y' in result.columns
    assert 'x' in result.columns
    assert result.iloc[0]['x'] == 10.0
    assert result.iloc[0]['y'] == 50.0

def test_split_train_validation():
    df = pd.DataFrame({
        'y': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    rng = np.random.default_rng(42)
    
    # 20% validation -> 2 points
    result = split_train_validation(df, 0.2, rng, shuffle=False)
    
    assert result['split'].value_counts()['train'] == 8
    assert result['split'].value_counts()['val'] == 2
    assert 'id' not in result.columns # shuffle=False doesn't set id in original code but wait...
    # split_train_validation code:
    # if shuffle:
    #   points_df = points_df.sample(frac=1, random_state=rng).reset_index(drop=True)
    #   points_df['id'] = points_df.index
    
    result_shuffled = split_train_validation(df, 0.2, rng, shuffle=True)
    assert 'id' in result_shuffled.columns
    assert result_shuffled['split'].value_counts()['train'] == 8
    assert result_shuffled['split'].value_counts()['val'] == 2
