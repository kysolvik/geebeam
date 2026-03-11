import pytest
import pandas as pd
from unittest.mock import patch
from geebeam.climate_indices import download_clim_indices

def test_download_clim_indice_amo():
    # Mock data for AMO
    mock_data = {
        'Year': [2020, 2020],
        'month': [1, 2],
        'SSTA': [0.1, 0.2]
    }
    mock_df = pd.DataFrame(mock_data)
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = mock_df
        
        result = download_clim_indices('amo', 2020, 2020)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 1)
        assert result.columns == ['metric']
        assert result.index[0] == pd.Timestamp('2020-01-01')
        assert result.iloc[0]['metric'] == 0.1

def test_download_clim_indices_other():
    # Mock data for other indices (e.g., SOI)
    mock_data = {
        'Date': ['2020-01-01', '2020-02-01'],
        'Value': [0.5, -0.5]
    }
    mock_df = pd.DataFrame(mock_data)
    
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = mock_df
        
        result = download_clim_indices('soi', 2020, 2020)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 1)
        assert result.columns == ['metric']
        assert result.index[0] == pd.Timestamp('2020-01-01')
        assert result.iloc[0]['metric'] == 0.5

def test_download_clim_indices_invalid():
    with pytest.raises(ValueError):
        download_clim_indices('invalid_index', 2020, 2020)
