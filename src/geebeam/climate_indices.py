"""Helpers to download non-spatial climate indices"""

import pandas as pd

def download_clim_indices(
        index_name: str,
        year_start: int,
        year_end: int
    ) -> pd.DataFrame:
    clim_registry = {
        'amo':'https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.amo.dat',
        'soi':'https://psl.noaa.gov/data/timeseries/month/data/soi.long.csv',
        'oni':'https://psl.noaa.gov/data/correlation/oni.csv',
        'mei': 'https://psl.noaa.gov/data/correlation/meiv2.csv',
        'tna': 'https://psl.noaa.gov/data/correlation/tna.csv'
    }

    try:
        download_url = clim_registry[index_name]
    except ValueError:
        raise ValueError(f'{index_name} not found. Current options are {clim_registry.keys()}')

    if index_name == 'amo':
        df = pd.read_csv(download_url, skiprows=1, sep='\s+')
        df['Date'] = df['Year'].astype(str) + '-' + df['month'].astype(str) + '-01'
        df = df.drop(columns=['Year','month'])[['Date','SSTA']]
    else:
        df = pd.read_csv(download_url)

    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['Date', 'metric']

    df = df.set_index('Date')
    indexer = (df.index.year >=year_start) & (df.index.year <= year_end)
    return df.loc[indexer]


